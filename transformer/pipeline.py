from transformers import pipeline
from konlpy.tag import Mecab

tokenizer = Mecab()

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
context = "현범이는 서울에 사는 잘생긴 남자다."

# 형태소 분석 후 조사 제거
tokens = tokenizer.pos(context)
cleaned_tokens = [word for word, tag in tokens if tag not in ["JKS", "JKB", "JKO", "JKG", "JC"]]
# 결과 문장 생성
cleaned_sentence = ' '.join(cleaned_tokens)

question = "잘생긴 남자의 이름은?"
result = qa_pipeline(question=question, context=cleaned_sentence)
print(result["answer"])  # "Facebook"

