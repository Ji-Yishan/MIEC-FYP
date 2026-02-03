# -*- coding: utf-8 -*-
"""
安全结构化剪枝 Qwen1.5-0.5B + 正确评估（含 attention_mask 修复）
✅ 修复 PPL 计算：正确忽略 padding token
✅ 修复 Windows tempfile 问题
✅ 保留生成质量、速度、大小对比
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
import time
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# 1. 设置
# ----------------------------
device = "cpu"
model_name = "Qwen/Qwen1.5-0.5B"
prune_ratio = 0.2
batch_size = 4

print(f"Device: {device} | Prune ratio: {prune_ratio * 100:.1f}%")

# ----------------------------
# 2. 加载 tokenizer 和模型
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nLoading original model for comparison...")
original_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, trust_remote_code=True
).to(device)
original_model.eval()

print("Loading model for pruning...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, trust_remote_code=True
).to(device)
model.eval()


# ----------------------------
# 3. 安全剪枝函数
# ----------------------------
def prune_mlp_block(up_proj, gate_proj, down_proj, prune_ratio, device):
    importance_up = torch.norm(up_proj.weight.data, p=1, dim=1)
    importance_gate = torch.norm(gate_proj.weight.data, p=1, dim=1)
    importance = importance_up + importance_gate

    num_inter = importance.numel()
    num_prune = int(prune_ratio * num_inter)
    if num_prune <= 0:
        return up_proj, gate_proj, down_proj

    _, keep_indices = torch.topk(importance, num_inter - num_prune, largest=True)
    keep_indices = keep_indices.sort().values

    new_up = nn.Linear(up_proj.in_features, len(keep_indices), bias=up_proj.bias is not None)
    new_up.weight.data = up_proj.weight.data[keep_indices]
    if up_proj.bias is not None:
        new_up.bias.data = up_proj.bias.data[keep_indices]

    new_gate = nn.Linear(gate_proj.in_features, len(keep_indices), bias=gate_proj.bias is not None)
    new_gate.weight.data = gate_proj.weight.data[keep_indices]
    if gate_proj.bias is not None:
        new_gate.bias.data = gate_proj.bias.data[keep_indices]

    new_down = nn.Linear(len(keep_indices), down_proj.out_features, bias=down_proj.bias is not None)
    new_down.weight.data = down_proj.weight.data[:, keep_indices]
    if down_proj.bias is not None:
        new_down.bias.data = down_proj.bias.data

    return new_up.to(device), new_gate.to(device), new_down.to(device)


# ----------------------------
# 4. 执行剪枝
# ----------------------------
for i, layer in enumerate(model.model.layers):
    up, gate, down = layer.mlp.up_proj, layer.mlp.gate_proj, layer.mlp.down_proj
    new_up, new_gate, new_down = prune_mlp_block(up, gate, down, prune_ratio, device)
    layer.mlp.up_proj, layer.mlp.gate_proj, layer.mlp.down_proj = new_up, new_gate, new_down

model.eval()
print("✅ Pruning completed.")


# ----------------------------
# 5. 工具函数（Windows-safe）
# ----------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model):
    fd, temp_path = tempfile.mkstemp(suffix=".pt")
    try:
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    finally:
        os.close(fd)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return size_mb


# ----------------------------
# 6. 加载测试数据（保留 attention_mask！）
# ----------------------------

print("\nLoading wikitext test set (test[:200]) for evaluation...")
test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:200]")

test_dataset = test_dataset.filter(lambda x: len(x["text"].strip()) > 10)


def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
        return_attention_mask=True  # 👈 关键：必须返回 attention_mask
    )


tokenized_test = test_dataset.map(
    tokenize, batched=True, remove_columns=["text"]
)
# 注意：必须包含 attention_mask
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
test_loader = DataLoader(tokenized_test, batch_size=batch_size, shuffle=False)


# ----------------------------

# 7. 修复后的 PPL 计算（正确忽略 padding）

# ----------------------------
def compute_perplexity(model, dataloader, device):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # 创建 labels，并将 padding 位置设为 -100（Hugging Face 忽略这些位置）
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            num_tokens = (labels != -100).sum().item()
            if num_tokens > 0:
                total_nll += outputs.loss.item() * num_tokens
                total_tokens += num_tokens

    if total_tokens == 0:
        return float('inf')
    ppl = torch.exp(torch.tensor(total_nll / total_tokens))
    return ppl.item()


print("\nComputing Perplexity (with attention_mask)...")
ppl_original = compute_perplexity(original_model, test_loader, device)
ppl_pruned = compute_perplexity(model, test_loader, device)


# ----------------------------
# 8. 推理速度测试
# ----------------------------
def measure_inference_speed(model, dataloader, device, num_runs=20):
    model.eval()
    latencies = []
    for _ in range(5):  # warmup
        for batch in dataloader:
            input_ids = batch["input_ids"][:1].to(device)
            with torch.no_grad():
                _ = model(input_ids)
            break
    for _ in range(num_runs):
        for batch in dataloader:
            input_ids = batch["input_ids"][:1].to(device)
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            break
    avg_latency = sum(latencies) / len(latencies)
    throughput = 1000 / avg_latency
    return avg_latency, throughput


print("Measuring inference speed...")
latency_orig, throughput_orig = measure_inference_speed(original_model, test_loader, device)
latency_pruned, throughput_pruned = measure_inference_speed(model, test_loader, device)

# ----------------------------
# 9. 模型大小
# ----------------------------
params_orig = count_parameters(original_model)
params_pruned = count_parameters(model)
size_orig = get_model_size_mb(original_model)
size_pruned = get_model_size_mb(model)

# ----------------------------
# 10. 生成样例
# ----------------------------
prompts = ["Natural language processing", "The future of AI is"]


def generate_text(model, prompt, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ----------------------------
# 11. 打印报告
# ----------------------------
print("\n" + "=" * 80)
print("📊 PRUNING EVALUATION REPORT (FIXED: attention_mask + labels=-100)")
print("=" * 80)

print(f"\n📦 MODEL SIZE:")
print(f"  Original Params: {params_orig:,} ({params_orig / 1e6:.2f}M)")
print(
    f"  Pruned   Params: {params_pruned:,} ({params_pruned / 1e6:.2f}M) → ↓{(1 - params_pruned / params_orig) * 100:.1f}%")
print(f"  Disk Size: {size_orig:.1f} MB → {size_pruned:.1f} MB (↓{(1 - size_pruned / size_orig) * 100:.1f}%)")

print(f"\n⚡ INFERENCE SPEED (per sample):")
print(f"  Original: {latency_orig:.2f} ms | {throughput_orig:.2f} samples/sec")
print(f"  Pruned:   {latency_pruned:.2f} ms | {throughput_pruned:.2f} samples/sec")
print(f"  Speedup:  {latency_orig / latency_pruned:.2f}x")

print(f"\n🎯 ACCURACY (Perplexity on wikitext test):")
print(f"  Original PPL: {ppl_original:.2f}")
print(f"  Pruned   PPL: {ppl_pruned:.2f}")
if ppl_original > 0:
    print(f"  Δ PPL:        {ppl_pruned - ppl_original:+.2f} ({(ppl_pruned / ppl_original - 1) * 100:+.1f}%)")

print(f"\n📝 GENERATION QUALITY:")
for prompt in prompts:
    orig_gen = generate_text(original_model, prompt, tokenizer, device)
    pruned_gen = generate_text(model, prompt, tokenizer, device)
    print(f"\nPrompt: {prompt}")
    print(f"  Original: {orig_gen}")
    print(f"  Pruned:   {pruned_gen}")

print("\n✅ Evaluation complete with CORRECT PPL calculation!")