import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 토크나이저 및 모델 로드
model_name = "JkhTV/UA-N1-Queen"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. 모델 테스트
while True:
    input_text = input("User: ")
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    # 3. 추론 실행
    output = model.generate(input_ids, max_length=100)
    print("UA: ", tokenizer.decode(output[0], skip_special_tokens=True))