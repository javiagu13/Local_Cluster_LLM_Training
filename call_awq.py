from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "./quantized_model"
# Load your model in floating point first.
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# (Optional) If you want to quantize calibration on your own data:
# calib_data = [ ... list of calibration texts ... ]
# quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
# model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

# Then run inference just like with Transformers:
prompt = "[|user|]F/11mon  2019.12.22 발열 없이 기침, 가래, 콧물 시작됨  2019.12.20 점심부터 발열 발생 local PED 내원하여 모세기관지염 의심 소견 듣고 항생제(clarithromycin) 포함 약 처방받아 복용함. 독감검사 시행하지 않음. 저녁에 40.3도 고열 발생함  2019.12.24 발열 지속되어 X-ray 촬영 권유받고 ER 내원함 \n[|assistant|]"

device = next(model.parameters()).device

inputs = tokenizer(prompt, return_tensors="pt").to(device)
output_ids = model.generate(inputs.input_ids, max_new_tokens=1280)
response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print(response)
