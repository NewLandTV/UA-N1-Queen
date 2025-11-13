import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# 1. 데이터 로드
dataset = load_dataset(
    path="dataset",
    data_files="train.jsonl",
    split="train"
)
print(dataset)

# 2. 토크나이저 및 모델 로드
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto"
)
print(f"{model_name} 모델이 성공적으로 로드되었습니다.")

# 3. 모델 양자화
model = prepare_model_for_kbit_training(model)

# 4. LoRA 설정
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)

# 5. 데이터 전처리 (tokenize)
def preprocess(example):
    prompt = f"User: {example['user']}\nAssistant: {example['assistant']}"
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(
    preprocess,
    batched=False,
    remove_columns=["user", "assistant"]
)

# 6. DataCollator 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 7. 학습 설정
output_dir = "./output"
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    logging_dir="./logs",
    output_dir=output_dir,
    save_strategy="epoch",
    save_total_limit=2,
    save_steps=300,
    remove_unused_columns=False
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 8. 모델 학습
trainer.train()

# 9. 모델 저장
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"모델이 {output_dir}에 저장되었습니다.")