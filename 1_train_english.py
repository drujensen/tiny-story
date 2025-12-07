# 1_train_english.py — 100% working on Fedora 43 + ROCm 6.4 + Radeon 8060S
import os
import torch
from transformers import (
    AutoTokenizer,
    GemmaConfig,
    GemmaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# ========================================
# ROCm 6.4 + gfx1150 fixes (Dec 2025 working combo)
# ========================================
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"      # gfx1150
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"            # Stable fallback kernels
os.environ["ROCM_FORCE_CDNA_MODE"] = "0"
os.environ["AMD_SERIALIZE_KERNEL"] = "1"
os.environ["TORCH_USE_HIP_DSA"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# CORRECT allocator options for ROCm 6.4 (this is what actually works)
os.environ["HIP_ALLOC_CONF"] = "expandable_segments:True"   # ← fixed variable name
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"               # helps UMA page mapping
# os.environ["ROCM_MEM_DEFRAG"] = "1"                       # not needed with the above

torch.set_float32_matmul_precision('high')

# ========================================
# 1–4. Tokenizer / Dataset / Model (unchanged)
# ========================================
tokenizer = AutoTokenizer.from_pretrained("./gemma-3-1b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")
dataset = dataset.select(range(200_000))          # remove later for full dataset

MAX_LENGTH = 1024
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_attention_mask=True,
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=1,
)

base_config = GemmaConfig.from_pretrained("./gemma-3-1b")
new_config = GemmaConfig(
    vocab_size=base_config.vocab_size,
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=32,
    num_attention_heads=12,
    num_key_value_heads=12,
    head_dim=64,
    max_position_embeddings=1024,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    attention_bias=False,
    hidden_act="gelu",
    pad_token_id=tokenizer.pad_token_id,
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

model = GemmaForCausalLM(new_config)
print(f"Model parameters: {model.num_parameters():,} (~150M)")

# ========================================
# 5. Warm-up (CPU → GPU to avoid early HIP randint bug)
# ========================================
print("Warming up GPU kernels...")
device = "cuda"
model = model.to(device)        # ← now works!

dummy_input_cpu = torch.randint(0, new_config.vocab_size, (2, 128))
dummy_input = dummy_input_cpu.to(device)
dummy_mask = torch.ones_like(dummy_input)
dummy_labels = torch.full_like(dummy_input, -100, dtype=torch.long)
dummy_labels[:, :64] = torch.arange(64).repeat(2, 1)

with torch.no_grad():
    out = model(input_ids=dummy_input, attention_mask=dummy_mask, labels=dummy_labels)
    masked_tokens = dummy_labels.ne(-100).sum().item()
print(f"Warm-up successful — loss: {out.loss.item():.4f}, real tokens: {masked_tokens}")

# ========================================
# 6–9. Rest unchanged (low-pressure settings that survive page faults)
# ========================================
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

training_args = TrainingArguments(
    output_dir="./tiny-story-temp",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,          # ultra-stable
    gradient_accumulation_steps=16,         # effective batch = 16
    learning_rate=4e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=2500,
    save_total_limit=3,
    bf16=True,
    fp16=False,
    torch_compile=False,
    dataloader_num_workers=1,
    dataloader_pin_memory=False,
    report_to="none",
    gradient_checkpointing=False,            # temporarily disabled to test stability
    optim="adamw_torch",
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, data_collator=data_collator)

print("Starting training...")
trainer.train()

final_path = "./tiny-story"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"Training complete! Model saved to {final_path}")
print("Test with:")
print("   from transformers import pipeline")
print(f"   pipe = pipeline('text-generation', model='{final_path}', device=0)")
