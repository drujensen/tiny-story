# 1_train_english.py — 100% working on Fedora 43 + ROCm 6.4 + Radeon 8060S
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    GemmaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

torch.set_float32_matmul_precision('high')
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'

tokenizer = AutoTokenizer.from_pretrained("./gemma-3-1b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")
dataset = dataset.select(range(200_000))

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

config = AutoConfig.from_pretrained("./gemma-3-1b")
model = GemmaForCausalLM(config)
model = model.to('cuda')


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

training_args = TrainingArguments(
    output_dir="./tiny-story-temp",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=2500,
    save_total_limit=3,
    bf16=False,
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
