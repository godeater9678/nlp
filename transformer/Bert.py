from datasets import load_dataset

# IMDb 데이터셋 로드 - 2.5만개 리뷰 데이터
dataset = load_dataset("imdb")
#print(dataset)  # train/test 데이터셋 확인


from transformers import BertTokenizer

# BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 데이터 전처리 함수 정의
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# 데이터셋 전처리
encoded_dataset = dataset.map(preprocess_function, batched=True)


import torch
from torch.utils.data import DataLoader

# 데이터셋 변환
encoded_dataset = encoded_dataset.rename_column("label", "labels")  # 라벨 이름 변경
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# DataLoader 생성
train_loader = DataLoader(encoded_dataset["train"], batch_size=16, shuffle=True)
test_loader = DataLoader(encoded_dataset["test"], batch_size=16)


from transformers import BertForSequenceClassification

# BERT 모델 로드
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # 긍정/부정 분류

from torch.optim import AdamW
from transformers import get_scheduler

# 옵티마이저와 학습률 스케줄러 설정
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 3  # 3 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# GPU 사용 설정
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# 학습 루프
from tqdm import tqdm

epochs = 3
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Progress bar 업데이트
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())


from sklearn.metrics import accuracy_score

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

# 정확도 계산
accuracy = accuracy_score(all_labels, all_predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# 로컬에 저장
model.save_pretrained("./imdb-bert-classifier")
tokenizer.save_pretrained("./imdb-bert-classifier")


# 저장된 모델과 토크나이저 로드
model_path = "./imdb-bert-classifier"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 예측 함수 정의
def predict_sentiment(review_text):
    # 텍스트를 토큰화
    inputs = tokenizer(review_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # GPU/CPU 이동

    # 모델 추론
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)  # 가장 높은 점수의 클래스로 예측

    # 라벨 매핑 (0: 부정, 1: 긍정)
    labels = ["Negative", "Positive"]
    return labels[prediction.item()]

# 테스트 리뷰
review_1 = "The movie was absolutely fantastic! I loved it."
review_2 = "I didn't enjoy the movie. It was boring and too long."

# 예측
print(f"Review 1 Sentiment: {predict_sentiment(review_1)}")  # Positive
print(f"Review 2 Sentiment: {predict_sentiment(review_2)}")  # Negative