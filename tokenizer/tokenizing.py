from konlpy.tag import Mecab

#https://resultofeffort.tistory.com/120 참고해서 Mecab 설치

# Mecab 형태소 분석기 초기화
tokenizer = Mecab()

# 예제 문장
sentence = "파이토치를 이용하여 자연어 처리를 진행합니다."

# Tokenization
tokens = tokenizer.morphs(sentence)  # 형태소 단위로 분리
print("Tokens:", tokens)



