from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
from deep_translator import GoogleTranslator

nltk.download('punkt')
nltk.download('punkt_tab')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="./models/gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="./models/gpt2")

# `pad_token_id`를 `eos_token_id`로 설정
tokenizer.pad_token = tokenizer.eos_token

ask = "현범이는 배가 아파요. 어떻게 해야해요?"
input_text = GoogleTranslator(source='ko', target='en').translate(
    ask
)
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# 텍스트 생성
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],  # attention_mask 추가
    max_length=50,
    num_return_sequences=10,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.pad_token_id  # pad_token_id 설정
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


