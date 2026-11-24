"""Hand-written small UNet for 28x56 single-channel images (noise prediction).

Resolutions: 28x56 -> 14x28 -> 7x14. Self-attention only at 7x14 (optional).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, groups=8, dropout=0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(groups, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch, heads=4, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.out = nn.Conv2d(ch, ch, 1)
        self.heads = heads

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.out(o.transpose(-1, -2).reshape(b, c, h, w))


class SmallUNet(nn.Module):
    def __init__(self, ch=(32, 64, 128), blocks=2, attn_lowest=True, tdim=128):
        super().__init__()
        self.tdim = tdim
        self.temb = nn.Sequential(nn.Linear(tdim, tdim * 2), nn.SiLU(), nn.Linear(tdim * 2, tdim * 2))
        td = tdim * 2
        self.inp = nn.Conv2d(1, ch[0], 3, padding=1)

        self.down = nn.ModuleList()
        skip_ch = [ch[0]]
        cur = ch[0]
        for i, c in enumerate(ch):
            for _ in range(blocks):
                self.down.append(ResBlock(cur, c, td)); cur = c; skip_ch.append(cur)
            if i < len(ch) - 1:
                self.down.append(nn.Conv2d(cur, cur, 3, stride=2, padding=1)); skip_ch.append(cur)

        self.mid1 = ResBlock(cur, cur, td)
        self.mid_attn = SelfAttention(cur) if attn_lowest else nn.Identity()
        self.mid2 = ResBlock(cur, cur, td)

        self.up = nn.ModuleList()
        for i, c in reversed(list(enumerate(ch))):
            for _ in range(blocks + 1):
                self.up.append(ResBlock(cur + skip_ch.pop(), c, td)); cur = c
            if i > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
                self.up.append(nn.Conv2d(cur, cur, 3, padding=1))
        self.out = nn.Sequential(nn.GroupNorm(8, cur), nn.SiLU(), nn.Conv2d(cur, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.tdim))
        h = self.inp(x)
        hs = [h]
        for m in self.down:
            h = m(h, temb) if isinstance(m, ResBlock) else m(h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for m in self.up:
            if isinstance(m, ResBlock):
                h = m(torch.cat([h, hs.pop()], 1), temb)
            else:
                h = m(h)
        return self.out(h)
