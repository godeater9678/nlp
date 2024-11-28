from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from util.transrator import korean_to_english, english_to_korean

# 데이터 준비
data = [
    {"text": "질문: 현범이는 누구야? 답변: 네, 그 사람은 개발자에요."},
    {"text": "질문: 현범이는 무엇을 좋아하나요? 답변: 그는 코딩과 새로운 기술을 배우는 것을 좋아해요."},
]

for item in data:
    item["text"] = korean_to_english(item["text"].strip())

# 데이터셋 생성
dataset = Dataset.from_list(data)

# 토크나이저 및 모델 로드
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="./models/gpt2")
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정
model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="./models/gpt2")
model.config.pad_token_id = tokenizer.pad_token_id

# 토큰화 함수
def tokenize(batch):
    encoding = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
    encoding["labels"] = encoding["input_ids"].clone()  # labels 추가
    return encoding

# 데이터셋 토큰화
tokenized_dataset = dataset.map(tokenize, batched=True)

# 훈련 설정
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=2,
    prediction_loss_only=True,
    learning_rate=5e-5  # 기본 학습률을 낮게 설정
)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-Tuning 실행
trainer.train()
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")
