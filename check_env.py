"""Milestone 1: check PyTorch sees the GPU and download MNIST to ./data."""
import time
import torch
import torchvision

print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available"
print("device:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))

# Small matmul + conv on GPU to make sure kernels actually run on this arch.
x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize(); t = time.time()
y = x @ x
torch.cuda.synchronize()
print(f"matmul ok, {time.time() - t:.3f}s, checksum {y.float().abs().mean().item():.3f}")
conv = torch.nn.Conv2d(1, 8, 3).cuda()
print("conv ok:", tuple(conv(torch.randn(2, 1, 28, 28, device="cuda")).shape))

tr = torchvision.datasets.MNIST("./data", train=True, download=True)
te = torchvision.datasets.MNIST("./data", train=False, download=True)
print("MNIST train", len(tr), "test", len(te))
keep = lambda ds: int((ds.targets <= 4).sum())
print("digits 0-4: train", keep(tr), "test", keep(te))
