# -*- coding: utf-8 -*-
"""
📥 完整离线缓存脚本（运行一次即可）
✅ 缓存 Qwen1.5-0.5B 模型（含 tokenizer）
✅ 缓存 wikitext-2-raw-v1 的 train[:200]（用于蒸馏训练）
✅ 缓存 wikitext-2-raw-v1 的 test[:200]（用于评估）
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os

# 设置本地路径
MODEL_DIR = "./local_model/qwen1_5_0_5b"
DATA_DIR = "./local_data"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("1. Downloading and caching model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B",
    torch_dtype="auto",
    trust_remote_code=True
)

# 保存模型和 tokenizer 到本地
tokenizer.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR)
print(f"✅ Model saved to: {MODEL_DIR}")

print("\n2. Downloading and caching TRAIN dataset (train[:200])...")
train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:200]")
train_ds = train_ds.filter(lambda x: len(x["text"].strip()) > 20)  # 过滤太短的
train_ds.save_to_disk(os.path.join(DATA_DIR, "wikitext2_train200"))
print("✅ Train dataset cached.")

print("\n3. Downloading and caching TEST dataset (test[:200])...")
test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:200]")
test_ds = test_ds.filter(lambda x: len(x["text"].strip()) > 10)
test_ds.save_to_disk(os.path.join(DATA_DIR, "wikitext2_test200"))
print("✅ Test dataset cached.")

print("\n🎉 All assets cached locally! You can now run offline experiments.")