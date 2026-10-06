"""
Test script for INT8 weight-only quantization in nano-vllm.
Run from the nano-vllm directory:
    python test_int8.py
"""
import os
import torch
from nanovllm import LLM, SamplingParams

MODEL_PATH = os.path.expanduser("C:/Users/Mizan/huggingface/Qwen3-0.6B")

# ── 1. FP16 baseline ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Loading FP16 model...")
print("="*60)

llm_fp16 = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1)
fp16_kv_blocks = llm_fp16.model_runner.config.num_kvcache_blocks

# Check first linear layer dtype
from nanovllm.layers.linear import LinearBase
for name, module in llm_fp16.model_runner.model.named_modules():
    if isinstance(module, LinearBase):
        print(f"  First linear layer : {name}")
        print(f"  Weight dtype       : {module.weight.dtype}")
        print(f"  Weight size        : {module.weight.nbytes / 1e6:.2f} MB")
        break

print(f"  KV cache blocks    : {fp16_kv_blocks}")
print(f"  Total VRAM used    : {torch.cuda.memory_allocated() / 1e9:.3f} GB")

# Run inference
sp = SamplingParams(temperature=0.1, max_tokens=60)
out_fp16 = llm_fp16.generate(["What is the capital of France?"], sp)
print(f"\n  Output: {out_fp16[0]['text']!r}")

llm_fp16.exit()
del llm_fp16
torch.cuda.empty_cache()
import gc; gc.collect()


# ── 2. INT8 quantized ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Loading INT8 model...")
print("="*60)

# Reset the flag between runs (important when running both in same process)
import nanovllm.layers.linear as _lm
_lm._INT8_QUANTIZED = False

llm_int8 = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1, quantization="int8")
int8_kv_blocks = llm_int8.model_runner.config.num_kvcache_blocks

for name, module in llm_int8.model_runner.model.named_modules():
    if isinstance(module, LinearBase):
        print(f"  First linear layer : {name}")
        print(f"  Weight dtype       : {module.weight.dtype}")       # expect torch.int8
        print(f"  Weight size        : {module.weight.nbytes / 1e6:.2f} MB")  # expect ~half
        print(f"  Scale shape        : {module.weight_scale.shape}") # expect [N, 1]
        break

print(f"  KV cache blocks    : {int8_kv_blocks}")
print(f"  Total VRAM used    : {torch.cuda.memory_allocated() / 1e9:.3f} GB")

out_int8 = llm_int8.generate(["What is the capital of France?"], sp)
print(f"\n  Output: {out_int8[0]['text']!r}")

del llm_int8
torch.cuda.empty_cache()


# ── 3. Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  FP16 KV cache blocks : {fp16_kv_blocks}")
print(f"  INT8 KV cache blocks : {int8_kv_blocks}")
print(f"  KV cache increase    : {int8_kv_blocks / fp16_kv_blocks:.2f}x")
print(f"  FP16 output : {out_fp16[0]['text'][:80]!r}")
print(f"  INT8 output : {out_int8[0]['text'][:80]!r}")
