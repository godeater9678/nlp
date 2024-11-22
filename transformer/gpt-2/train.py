from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# 모델과 토크나이저 로드
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="./models/gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="./models/gpt2")

# 패딩 토큰 설정
tokenizer.pad_token = tokenizer.eos_token

# 데이터셋 준비
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size  # 학습 데이터의 최대 토큰 길이
    )
    return dataset

# 데이터 로드
train_file = "./data/custom_data.txt"  # 학습 데이터 경로
train_dataset = load_dataset(train_file, tokenizer)

# 데이터 처리기
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2는 언어 모델링(MLM) 대신 일반 LM을 사용
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./models/gpt2-finetuned",  # 출력 디렉토리
    overwrite_output_dir=True,
    num_train_epochs=3,  # 학습 에포크 수
    per_device_train_batch_size=4,  # 배치 크기
    save_steps=500,  # 체크포인트 저장 간격
    save_total_limit=2,  # 최대 저장 체크포인트 수
    logging_dir="./logs",  # 로그 디렉토리
    logging_steps=100,
    prediction_loss_only=True,
)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# 모델 학습
trainer.train()

# Fine-Tuned 모델 저장
trainer.save_model("./models/gpt2-finetuned")
tokenizer.save_pretrained("./models/gpt2-finetuned")
