import subprocess
subprocess.run(["pip", "install", "autoawq", "transformers", "torch"])

import os
import torch
import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

# Parse arguments (local paths)
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./model")
parser.add_argument("--quant_path", type=str, default="./quantized_model")
#parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face authentication token")

args = parser.parse_args()

# Point to your local file
calib_dataset = load_dataset("json", data_files="./calib_data/val.jsonl.zst", split="train")
# Extract the text samples (adjust the key if needed)
calib_data = [sample["text"] for sample in calib_dataset.select(range(512))]

# Define quantization config
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

print(f"Loading model from {args.model_path}...")
model = AutoAWQForCausalLM.from_pretrained(
    args.model_path,
    low_cpu_mem_usage=True,
    use_cache=False,
    torch_dtype=torch.float16,
    #use_auth_token=args.hf_token
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
    #use_auth_token=args.hf_token,
    trust_remote_code=True
)

# Perform quantization
print("Quantizing model...")
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

# Save quantized model locally
os.makedirs(args.quant_path, exist_ok=True)
print(f"Saving quantized model at {args.quant_path}...")
model.save_quantized(args.quant_path)
tokenizer.save_pretrained(args.quant_path)

print("Quantization complete and saved locally.")
