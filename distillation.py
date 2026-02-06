# distill_cpu_local.py
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
from tqdm import tqdm
import bitsandbytes as bnb  # 仅 GPU 使用

# ----------------------------
# 配置（CPU 优化版）
# ----------------------------
LOCAL_MODEL_PATH = "./local_model/qwen1_5_0_5b"
LOCAL_DATASET_PATH = "./local_data/wikitext2_test200.json"

BATCH_SIZE = 1          # CPU 必须小 batch
MAX_LENGTH = 64         # 缩短序列长度（原128 → 64）
TEMPERATURE = 2.0
ALPHA = 0.5
LEARNING_RATE = 5e-5
EPOCHS = 1              # 减少 epoch（原2 → 1）
SEED = 42

set_seed(SEED)
device = torch.device("cpu")  # 强制使用 CPU
print(f"Using device: {device}")

# ----------------------------
# 模型加载（CPU 安全）
# ----------------------------
print("Loading models from local path...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ⚠️ CPU 必须用 float32！
teacher_model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.float32,  # 关键修改
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to(device).eval()

student_model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.float32,  # 关键修改
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to(device)
student_model.train()

# ----------------------------
# 数据集加载
# ----------------------------
print(f"Loading dataset from {LOCAL_DATASET_PATH}...")
raw_dataset = load_dataset(
    "json",
    data_files=LOCAL_DATASET_PATH,
    split="train"
)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,  # 使用缩短的长度
    )

tokenized_ds = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

train_loader = DataLoader(
    tokenized_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=default_data_collator
)

# ----------------------------
# 优化器（CPU 兼容）
# ----------------------------
if device.type == "cuda":
    optimizer = bnb.optim.AdamW8bit(student_model.parameters(), lr=LEARNING_RATE)
else:
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)  # CPU 标准优化器

kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
ce_loss_fn = torch.nn.CrossEntropyLoss()

# ----------------------------
# 训练循环
# ----------------------------
print(f"\n🚀 Starting Distillation on CPU (16GB RAM)...")
print(f"Dataset size: {len(tokenized_ds)} examples")
print(f"Batch size: {BATCH_SIZE}, Max length: {MAX_LENGTH}, Epochs: {EPOCHS}")

for epoch in range(EPOCHS):
    total_loss = 0.0
    student_model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits

        student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Shift logits and labels
        shift_t = teacher_logits[..., :-1, :].contiguous()
        shift_s = student_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Losses
        soft_targets = torch.softmax(shift_t / TEMPERATURE, dim=-1)
        soft_probs = torch.log_softmax(shift_s / TEMPERATURE, dim=-1)
        kl_loss = kl_loss_fn(soft_probs, soft_targets) * (TEMPERATURE ** 2)
        ce_loss = ce_loss_fn(shift_s.view(-1, shift_s.size(-1)), shift_labels.view(-1))
        loss = ALPHA * kl_loss + (1 - ALPHA) * ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(train_loader):.4f}")

# ----------------------------
# 保存模型
# ----------------------------
student_model.save_pretrained("./distilled_student_cpu")  # 已是 float32
tokenizer.save_pretrained("./distilled_student_cpu")
print("✅ Saved distilled model locally!")