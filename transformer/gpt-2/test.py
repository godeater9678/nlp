from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
from deep_translator import GoogleTranslator

nltk.download('punkt')
nltk.download('punkt_tab')

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="./models/gpt2")
#model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="./models/gpt2")
# Fine-Tuned 모델과 토크나이저 로드
model_path = "./models/gpt2-finetuned"  # Fine-Tuned 모델 경로
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# `pad_token_id`를 `eos_token_id`로 설정
tokenizer.pad_token = tokenizer.eos_token

ask = "what is PYUN"
input_text = GoogleTranslator(source='ko', target='en').translate(
    ask
)
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# 텍스트 생성
outputs = model.generate(
    input_ids=inputs["input_ids"]
)

print("질문: ", ask)
for out in outputs:
    answer = GoogleTranslator(source='en', target='ko').translate(
        tokenizer.decode(out, skip_special_tokens=True)
    )
    # 문장 단위로 나누기
    sentences = sent_tokenize(answer)
    for sentence in sentences:
        print("답변: ", sentence)
        break
    break


