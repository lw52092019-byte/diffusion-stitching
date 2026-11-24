"""Shared helpers: MNIST 0-4 data on GPU, classifier architectures, checkpoint paths.

Pixel convention everywhere in this project: float images in [-1, 1], shape (B, 1, H, W).
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
NUM_CLASSES = 5  # digits 0..4
DEVICE = "cuda"


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mnist04(train: bool, device=DEVICE):
    """Return (images in [-1,1] of shape (N,1,28,28), labels (N,)) for digits 0-4, on `device`."""
    ds = torchvision.datasets.MNIST(DATA_DIR, train=train, download=False)
    keep = ds.targets <= 4
    x = ds.data[keep].float().div(255.0).mul(2).sub(1).unsqueeze(1)
    y = ds.targets[keep].long()
    return x.to(device), y.to(device)


class JudgeCNN(nn.Module):
    """Small plain CNN (no BatchNorm). Used to score x0_hat during sampling."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.c4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.max_pool2d(F.relu(self.c2(x)), 2)   # 14x14
        x = F.relu(self.c3(x)); x = F.max_pool2d(F.relu(self.c4(x)), 2)   # 7x7
        x = self.drop(F.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)


class _ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.b1 = nn.BatchNorm2d(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.skip = None if (cin == cout and stride == 1) else nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        h = F.relu(self.b1(x))
        s = x if self.skip is None else self.skip(h)
        h = self.c1(h)
        h = self.c2(F.relu(self.b2(h)))
        return h + s


class EvalResNet(nn.Module):
    """Different architecture for the evaluator: small pre-activation ResNet with BatchNorm
    and global average pooling (vs. the judge's plain conv + max-pool + FC)."""

    def __init__(self, w=48):
        super().__init__()
        self.stem = nn.Conv2d(1, w, 3, 1, 1, bias=False)
        self.blocks = nn.Sequential(
            _ResBlock(w, w, 1), _ResBlock(w, 2 * w, 2),        # 14x14
            _ResBlock(2 * w, 2 * w, 1), _ResBlock(2 * w, 4 * w, 2),  # 7x7
        )
        self.bn = nn.BatchNorm2d(4 * w)
        self.fc = nn.Linear(4 * w, NUM_CLASSES)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        h = F.relu(self.bn(h)).mean(dim=(2, 3))
        return self.fc(h)


CLASSIFIERS = {"judge": JudgeCNN, "evaluator": EvalResNet}


def load_classifier(name: str, device=DEVICE):
    model = CLASSIFIERS[name]().to(device)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"{name}.pt"), map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
