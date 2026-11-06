"""RawNet: PyNET-style end-to-end learned-only baseline (packed Bayer -> RGB).

Input  (B, 5, H/2, W/2): RGGB packed to 4 channels + noise sigma map
Output (B, 3, H, W): PixelShuffle(2) of a 12-channel half-resolution prediction.
Plain convs, ReLU, Resize, PixelShuffle (DepthToSpace): ONNX opset 17 and INT8 friendly.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

RAWNET_WIDTH = 32


def _block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, 1, 1), nn.ReLU())


class RawNet(nn.Module):
    def __init__(self, width: int = RAWNET_WIDTH, in_ch: int = 5) -> None:
        super().__init__()
        ch = [width, width * 2, width * 4]
        self.enc = nn.ModuleList([_block(in_ch, ch[0])])
        self.enc.extend(_block(ch[i - 1], ch[i], stride=2) for i in range(1, 3))
        self.dec = nn.ModuleList([_block(ch[i + 1] + ch[i], ch[i]) for i in range(2)])
        self.out = nn.Conv2d(ch[0], 12, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.enc[0](x)]
        for enc in self.enc[1:]:
            feats.append(enc(feats[-1]))
        y = feats[-1]
        for i in (1, 0):
            y = F.interpolate(y, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            y = self.dec[i](torch.cat([y, feats[i]], dim=1))
        return self.shuffle(self.out(y))
