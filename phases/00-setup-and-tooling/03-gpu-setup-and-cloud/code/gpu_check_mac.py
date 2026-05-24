import platform
import time


def check_gpu():
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")
        return

    print("=== GPU Check ===\n")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.system()}")

    if not torch.backends.mps.is_available():
        print("Apple Metal (MPS) backend not available.")
        print("Make sure you're using Apple Silicon and a recent PyTorch version.")
        return

    print("Backend: Apple Metal (MPS)")
    print("GPU: Apple Silicon GPU")

    print("\n=== CPU vs GPU Benchmark ===\n")

    size = 4000

    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # CPU benchmark
    start = time.time()
    _ = a @ b
    cpu_time = time.time() - start

    print(f"CPU matrix multiply ({size}x{size}): {cpu_time:.3f}s")

    # MPS benchmark
    device = torch.device("mps")

    a_gpu = a.to(device)
    b_gpu = b.to(device)

    # Warm-up
    _ = a_gpu @ b_gpu
    torch.mps.synchronize()

    start = time.time()

    _ = a_gpu @ b_gpu

    torch.mps.synchronize()

    gpu_time = time.time() - start

    print(f"MPS matrix multiply ({size}x{size}): {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")

    print(
        "\nApple Silicon uses unified memory, so usable model size depends on total system RAM."
    )


if __name__ == "__main__":
    check_gpu()
