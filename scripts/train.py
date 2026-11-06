"""Train LiteISPNet / RawNet. Writes runs/<name>/{best.pt,last.pt,log.csv}.

Examples:
  python -m scripts.train --model rawnet --name rawnet --epochs 30
  python -m scripts.train --model liteisp --width 56 --name teacher --epochs 40
  python -m scripts.train --model liteisp --name student --teacher runs/teacher/best.pt
  python -m scripts.train --model liteisp --name ft --finetune runs/student/best.pt --lr 1e-4

Dataset contract (isp.data.datasets.SyntheticRawDataset(split, crop, seed)): yields
(RawImage, target float32 (crop, crop, 3) sRGB in [0, 1]). Only "train" and "val" are ever requested:
the Kodak24 test split is never touched here. Checkpoint selection uses val loss.
"""
from __future__ import annotations

import argparse
import csv
import functools
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from isp.learned.checkpoint import DEFAULT_WIDTH, build_model, load_checkpoint, save_checkpoint
from isp.learned.inputs import liteisp_input, rawnet_input
from isp.learned.losses import isp_loss

LOG_FIELDS = ["epoch", "train_loss", "val_loss", "val_psnr", "lr"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(DEFAULT_WIDTH), required=True)
    p.add_argument("--name", required=True, help="run name -> runs/<name>/")
    p.add_argument("--width", type=int, default=None, help="default: student/RawNet width")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=128, help="training crop; 128 x batch 8 fits 6 GB with AMP")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher", help="liteisp checkpoint for knowledge distillation")
    p.add_argument("--kd-weight", type=float, default=0.5)
    p.add_argument("--finetune", help="checkpoint whose weights initialise this run")
    p.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None, help="cap training steps per epoch")
    p.add_argument("--runs-dir", default="runs")
    args = p.parse_args(argv)
    if args.teacher and args.model != "liteisp":
        p.error("--teacher (distillation) is only supported for --model liteisp")
    if args.finetune and args.teacher:
        p.error("use either --finetune or --teacher in one run")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_dataset_factory(split: str, crop: int, seed: int):
    from isp.data.datasets import SyntheticRawDataset  # W2 contract; imported lazily

    return SyntheticRawDataset(split, crop, seed)


def default_front_end():
    from isp.classical.pipeline import front_end  # W1 contract; only liteisp needs it

    return front_end


def collate(batch, model_name: str, front_end):
    xs, ys = [], []
    for raw, target in batch:
        if model_name == "rawnet":
            xs.append(rawnet_input(raw))
        else:
            rgb, _, info = front_end(raw, False)
            xs.append(liteisp_input(rgb, info["noise_a"], info["noise_b"], info["exposure"]))
        ys.append(torch.as_tensor(np.asarray(target, dtype=np.float32)).permute(2, 0, 1))
    return torch.stack(xs), torch.stack(ys)


def make_loader(ds, args, front_end, shuffle: bool, epoch: int = 0):
    gen = torch.Generator().manual_seed(args.seed * 1000 + epoch)
    fn = functools.partial(collate, model_name=args.model, front_end=front_end)
    return torch.utils.data.DataLoader(ds, args.batch_size, shuffle=shuffle and hasattr(ds, "__len__"),
                                       collate_fn=fn, num_workers=args.workers, generator=gen)


def lr_at(args, epoch: int) -> float:
    return args.lr * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target).item()
    return 10 * math.log10(1.0 / max(mse, 1e-10))


def step_loss(model, teacher, x, y, kd_weight):
    pred = model(x).float()
    loss = isp_loss(pred, y)
    if teacher is not None:
        with torch.no_grad():
            soft = teacher(x).float()
        loss = loss + kd_weight * F.l1_loss(pred, soft)
    return loss


def train_epoch(model, teacher, loader, opt, scaler, args, device, amp) -> float:
    model.train()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if args.max_steps is not None and i >= args.max_steps:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = step_loss(model, teacher, x, y, args.kd_weight)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total, n = total + loss.item(), n + 1
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, amp) -> tuple[float, float]:
    model.eval()
    losses, psnrs = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            pred = model(x).float()
        losses.append(isp_loss(pred, y).item())
        psnrs.append(psnr(pred, y))
    return float(np.mean(losses)), float(np.mean(psnrs))


def append_log(path: Path, row: dict, fresh: bool) -> None:
    with open(path, "w" if fresh else "a", newline="") as f:
        w = csv.DictWriter(f, LOG_FIELDS)
        if fresh:
            w.writeheader()
        w.writerow(row)


def load_state(args, model, opt, scaler, run_dir: Path, device) -> tuple[int, float]:
    """Returns (start_epoch, best_val). Handles --resume and --finetune."""
    last = run_dir / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        return ck["epoch"], ck["best_val"]
    if args.finetune:
        src, meta = load_checkpoint(args.finetune, expect=args.model)
        model.load_state_dict(src.state_dict())
    return 0, float("inf")


def main(argv=None, dataset_factory=default_dataset_factory, front_end=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    amp = device.type == "cuda" and not args.no_amp
    if args.model == "liteisp" and front_end is None:
        front_end = default_front_end()
    width = args.width or DEFAULT_WIDTH[args.model]
    model = build_model(args.model, width).to(device)
    teacher = load_checkpoint(args.teacher, expect="liteisp")[0].to(device).eval() if args.teacher else None
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    run_dir = Path(args.runs_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    start, best = load_state(args, model, opt, scaler, run_dir, device)
    train_ds = dataset_factory("train", args.crop, args.seed)
    val_loader = make_loader(dataset_factory("val", args.crop, args.seed), args, front_end, shuffle=False)
    for epoch in range(start, args.epochs):
        t0 = time.time()
        lr = lr_at(args, epoch)
        for g in opt.param_groups:
            g["lr"] = lr
        loader = make_loader(train_ds, args, front_end, shuffle=True, epoch=epoch)
        tr = train_epoch(model, teacher, loader, opt, scaler, args, device, amp)
        val, val_psnr = validate(model, val_loader, device, amp)
        append_log(run_dir / "log.csv", dict(epoch=epoch + 1, train_loss=tr, val_loss=val,
                                             val_psnr=val_psnr, lr=lr), fresh=epoch == 0)
        if val < best:
            best = val
            save_checkpoint(run_dir / "best.pt", model, args.model, width, val_loss=val, epoch=epoch + 1)
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "best_val": best}, run_dir / "last.pt")
        print(f"epoch {epoch + 1}/{args.epochs} train {tr:.4f} val {val:.4f} psnr {val_psnr:.2f} "
              f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
