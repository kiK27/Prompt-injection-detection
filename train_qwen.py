from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from trl import SFTTrainer

import torch
import os

# -----------------------------
# Disable wandb
# -----------------------------

os.environ["WANDB_DISABLED"] = "true"

# -----------------------------
# Model
# -----------------------------

model_name = "Qwen/Qwen2-0.5B-Instruct"

# -----------------------------
# Load Dataset
# -----------------------------

print("Loading dataset...")

dataset = load_dataset(
    "cyberec/Prompt-injection-dataset",
    "core"
)

train_data = dataset["train"]

# Smaller subset for Colab stability
train_data = train_data.select(range(2000))

# -----------------------------
# Format Dataset
# -----------------------------

def format_example(example):

    label_map = {
        0: "SAFE",
        1: "MALICIOUS"
    }

    return {
        "text":
f"""### Instruction:
Analyze the following input for prompt injection attacks.

### Input:
{example['text']}

### Response:
{label_map[example['label']]}
"""
    }

train_data = train_data.map(format_example)

# -----------------------------
# Tokenizer
# -----------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Quantization Config
# -----------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# -----------------------------
# Load Model
# -----------------------------

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager"
)

# -----------------------------
# Important Stability Fixes
# -----------------------------

model.config.use_cache = False

model.gradient_checkpointing_enable()

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# -----------------------------
# LoRA Config
# -----------------------------

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Attach LoRA adapters
model = get_peft_model(
    model,
    peft_config
)

# Verify trainable parameters
model.print_trainable_parameters()

# -----------------------------
# Training Arguments
# -----------------------------

training_args = TrainingArguments(
    output_dir="models/qwen-prompt-detector",

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    num_train_epochs=1,

    learning_rate=2e-4,

    logging_steps=10,

    save_steps=100,

    fp16=True,

    report_to="none"
)

# -----------------------------
# Trainer
# -----------------------------

trainer = SFTTrainer(
    model=model,

    train_dataset=train_data,

    args=training_args,

    processing_class=tokenizer,

    formatting_func=lambda x: x["text"]
)

# -----------------------------
# Train
# -----------------------------

print("Training started...")

trainer.train()

# -----------------------------
# Save Model
# -----------------------------

print("Saving model...")

trainer.save_model(
    "models/qwen-prompt-detector"
)

tokenizer.save_pretrained(
    "models/qwen-prompt-detector"
)

print("Training complete!")