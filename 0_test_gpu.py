import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("No CUDA GPU detected. Check ROCm installation.")

# For ROCm
try:
    print("ROCm available:", torch.version.hip is not None)
    if torch.version.hip:
        print("ROCm version:", torch.version.hip)
except AttributeError:
    print("ROCm not detected.")