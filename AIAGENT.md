# TinyStories Gemma Training on Radeon 8060S (gfx1150) – Working Setup (Dec 2025)

## Hardware
- CPU: AMD Ryzen AI Max+ 395 (32 Zen 5 cores/threads)
- iGPU: Radeon 8060S Graphics (gfx1150, RDNA 3.5, ~31 GiB shared VRAM)
- System: Framework Desktop (AMD Ryzen AI Max 300 Series)

## Software (Dec 2025)
- OS: Fedora 43 (Workstation)
- Kernel: 6.17.8-300.fc43.x86_64
- ROCm: 6.4.3 (Fedora native)
- PyTorch: installed from the official rocm6.4 index
- Python: 3.12.7 (pyenv)

## Critical Working Environment Variables (ROCm 6.4)
These must be set before importing torch:

HSA_OVERRIDE_GFX_VERSION=11.0.0  
PYTORCH_ROCM_ARCH=gfx1100  
ROCM_FORCE_CDNA_MODE=0  
AMD_SERIALIZE_KERNEL=1  
TORCH_USE_HIP_DSA=1  
HIP_VISIBLE_DEVICES=0  
TORCHINDUCTOR_DISABLE=1  
HIP_ALLOC_CONF=expandable_segments:True  
HSA_FORCE_FINE_GRAIN_PCIE=1

In Python scripts, place the equivalent os.environ lines at the very top.

## Stable Training Settings (ROCm 6.4)
Start with very conservative values to guarantee no page faults:

per_device_train_batch_size = 1  
gradient_accumulation_steps = 32 (effective batch 32)  
gradient_checkpointing = True  
bf16 = True  
torch_compile = False  
dataloader_num_workers = 1  
dataloader_pin_memory = False

Once the run survives 500–1000 steps without crashing, gradually increase batch size to 2 → 4 (and lower accumulation steps accordingly).

## When ROCm 7.1.1 Arrives in Fedora 43 (or you move to Rawhide/Fedora 44)
Drop almost all environment variables (keep only HSA_OVERRIDE_GFX_VERSION=11.0.0).  
Reinstall PyTorch from the rocm7.1 index.  
You can then use batch sizes of 8–16, enable torch_compile, disable gradient checkpointing → expect 200–300 it/s.

## Useful Commands
rocminfo | grep -A5 gfx1150  
rocm-smi  
watch -n 1 rocm-smi (monitor VRAM during training)  
sudo dnf install libdrm-amdgpu (eliminates the libdrm.ids warning)

## Harmless Warnings (safe to ignore)
- “You are using a model of type gemma3_text to instantiate a model of type gemma”
- “/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory”

## Future Upgrade Path
When Fedora 43 finally ships ROCm 7.1.1 (expected Q1–Q2 2026):
1. Run normal system update  
2. Reinstall PyTorch for rocm7.1  
3. Remove most env vars  
4. Increase batch size dramatically for 2–3× speed

The actual working training script (the one that finally succeeded in Dec 2025) is saved as 1_train_english.py in this folder.

— Happy training!  
drujensen – December 2025
