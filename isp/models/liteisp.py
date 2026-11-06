"""LiteISPNet: depthwise-separable U-Net with a residual restoration branch and a global colour head.

Input  (B, 4, H, W): classical sRGB (3) + noise sigma map (1)      [F-DL-4]
Output (B, 3, H, W): colour_matrix @ (rgb + residual) + colour_bias [F-DL-1, F-DL-3]

Only Conv / ReLU / ReduceMean / Gemm / Resize / Concat / MatMul / Add are used, so the graph
exports to ONNX opset 17 and quantises to INT8 without custom ops. No normalisation layers.
Any H, W work: strided convs round up and upsampling resizes to the skip tensor's size.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

STUDENT_WIDTH = 24
TEACHER_WIDTH = 56
_LEVELS = (1, 2, 4, 8)  # channel multipliers per resolution level


class DSConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 + ReLU."""

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin)
        self.pw = nn.Conv2d(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.pw(self.dw(x)))


class LiteISPNet(nn.Module):
    def __init__(self, width: int = STUDENT_WIDTH, in_ch: int = 4) -> None:
        super().__init__()
        ch = [width * m for m in _LEVELS]
        self.head = nn.Sequential(nn.Conv2d(in_ch, ch[0], 3, 1, 1), nn.ReLU())
        self.enc = nn.ModuleList([DSConv(ch[0], ch[0])])
        self.down = nn.ModuleList()
        for i in range(1, len(ch)):
            self.down.append(DSConv(ch[i - 1], ch[i], stride=2))
            self.enc.append(DSConv(ch[i], ch[i]))
        self.dec = nn.ModuleList([DSConv(ch[i + 1] + ch[i], ch[i]) for i in range(len(ch) - 1)])
        self.out = nn.Conv2d(ch[0], 3, 1)
        self.colour_head = nn.Sequential(nn.Linear(ch[-1], 32), nn.ReLU(), nn.Linear(32, 12))
        # start as identity: zero residual, identity matrix, zero bias
        for layer in (self.out, self.colour_head[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = [self.enc[0](self.head(x))]
        for down, enc in zip(self.down, self.enc[1:]):
            feats.append(enc(down(feats[-1])))
        return feats

    def _decode(self, feats: list[torch.Tensor]) -> torch.Tensor:
        y = feats[-1]
        for i in range(len(self.dec) - 1, -1, -1):
            y = F.interpolate(y, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            y = self.dec[i](torch.cat([y, feats[i]], dim=1))
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._encode(x)
        restored = x[:, :3] + self.out(self._decode(feats))
        p = self.colour_head(feats[-1].mean(dim=(2, 3)))
        eye = torch.eye(3, dtype=x.dtype, device=x.device)
        mat = p[:, :9].reshape(-1, 3, 3) + eye
        b, _, h, w = restored.shape
        out = torch.bmm(mat, restored.reshape(b, 3, h * w)) + p[:, 9:].unsqueeze(-1)
        return out.reshape(b, 3, h, w)
