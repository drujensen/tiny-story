# 2_fine_tune_chat.py
# Fine-tune the trained TinyStory model on Hermes-2.5 chat dataset

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"
os.environ["ROCM_FORCE_CDNA_MODE"] = "0"
os.environ["AMD_SERIALIZE_KERNEL"] = "1"
os.environ["TORCH_USE_HIP_DSA"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
torch.set_float32_matmul_precision('high')

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# ========================================
# 1. Load tokenizer and model
# ========================================
tokenizer = AutoTokenizer.from_pretrained("./tiny-story")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("./tiny-story")
model = model.to('cuda')

# ========================================
# 2. Load Hermes-2.5 dataset
# ========================================
print("Loading Hermes-2.5 dataset...")
dataset = load_dataset("teknium/OpenHermes-2.5", split="train")

# Optional: use a subset for quick testing
dataset = dataset.select(range(50_000))  # balanced data for improvement

# ========================================
# 3. Tokenization with chat template
# ========================================
MAX_LENGTH = 512

def tokenize_function(examples):
    texts = []
    for conv in examples["conversations"]:
        # Convert to standard chat format
        messages = []
        for msg in conv:
            role = msg["from"]
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            elif role == "system":
                # Skip system messages or prepend to first user message
                continue
            messages.append({"role": role, "content": msg["value"]})
        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    outputs = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,  # let data collator handle
        return_attention_mask=True,
    )
    return outputs

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=1,
)

# ========================================
# 4. Data collator
# ========================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal LM
    pad_to_multiple_of=8,
)

# ========================================
# 5. Training arguments
# ========================================
training_args = TrainingArguments(
    output_dir="./tiny-story-chat",
    overwrite_output_dir=True,
    num_train_epochs=2,  # balanced epochs for improvement
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,  # slightly higher for better adaptation
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=20,
    save_steps=1000,
    save_total_limit=2,
    bf16=False,
    fp16=False,
    torch_compile=False,
    dataloader_num_workers=1,
    dataloader_pin_memory=False,
    report_to="none",
    gradient_checkpointing=False,
    optim="adamw_torch",
)

# ========================================
# 6. Trainer
# ========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting fine-tuning...")
trainer.train()

# ========================================
# 7. Save final model
# ========================================
final_path = "./tiny-story-chat"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

print(f"Fine-tuning complete! Model saved to {final_path}")
