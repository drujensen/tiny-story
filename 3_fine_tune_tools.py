# 3_fine_tune_tools.py
# Fine-tune the chat model on tool calling dataset

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
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
tokenizer = AutoTokenizer.from_pretrained("./tiny-story-chat")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("./tiny-story-chat")
model = model.to('cuda')

# ========================================
# 2. Load tool calling dataset
# ========================================
print("Loading tool calling dataset...")
dataset = load_dataset("BitAgent/tool_calling", split="train")

# Optional: use a subset
dataset = dataset.select(range(50_000))

# ========================================
# 3. Tokenization with chat template
# ========================================
MAX_LENGTH = 1024

def tokenize_function(examples):
    texts = []
    for conv in examples["conversations"]:
        text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    
    outputs = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_attention_mask=True,
    )
    return outputs

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=4,
)

# ========================================
# 4. Data collator
# ========================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# ========================================
# 5. Training arguments
# ========================================
training_args = TrainingArguments(
    output_dir="./tiny-story-tools",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=20,
    save_steps=1000,
    save_total_limit=2,
    bf16=True,
    torch_compile=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
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

print("Starting tool fine-tuning...")
trainer.train()

# ========================================
# 7. Save final model
# ========================================
final_path = "./tiny-story-tools"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

print(f"Tool fine-tuning complete! Model saved to {final_path}")