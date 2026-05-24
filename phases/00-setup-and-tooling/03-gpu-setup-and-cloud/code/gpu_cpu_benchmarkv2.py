import time

import torch

size = 4000
runs = 10
device = torch.device("mps")

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

# CPU warmup
_ = a_cpu @ b_cpu

cpu_times = []
for _ in range(runs):
    start = time.perf_counter()
    _ = a_cpu @ b_cpu
    cpu_times.append(time.perf_counter() - start)

# Move to GPU before timing
a_gpu = a_cpu.to(device)
b_gpu = b_cpu.to(device)
torch.mps.synchronize()

# MPS warmup
_ = a_gpu @ b_gpu
torch.mps.synchronize()

gpu_times = []
for _ in range(runs):
    start = time.perf_counter()
    _ = a_gpu @ b_gpu
    torch.mps.synchronize()
    gpu_times.append(time.perf_counter() - start)

cpu_avg = sum(cpu_times[1:]) / (runs - 1)
gpu_avg = sum(gpu_times[1:]) / (runs - 1)

print(f"CPU avg: {cpu_avg:.4f}s")
print(f"MPS avg: {gpu_avg:.4f}s")
print(f"Speedup: {cpu_avg / gpu_avg:.1f}x")
