import re
import numpy as np
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger_eng')
import spacy

# spaCy 영어 모델 로드
nlp = spacy.load("en_core_web_sm")
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from util.translator import korean_to_english, english_to_korean


def read_from_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def read_from_txt_korean(file_path) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        kor_lines = file.readlines()
    lines = []
    for line in kor_lines:
        if line.strip() != '':
            lines.append(korean_to_english(line.strip()))
    return lines


def split_questions_and_answers(lines):
    questions = []
    answers = []
    current_question = None
    current_answer = []

    for line in lines:
        line = line.strip()
        if not line:  # 빈 줄은 무시
            continue
        if line.endswith("?"):  # 질문인지 확인
            if current_question and current_answer:
                answers.append(" ".join(current_answer))  # 이전 답변 저장
            current_question = line
            questions.append(current_question)
            current_answer = []  # 답변 초기화
        elif current_question:  # 답변으로 간주
            current_answer.append(line)

    # 마지막 답변 추가
    if current_answer:
        answers.append(" ".join(current_answer))

    return questions, answers


def is_question(sentence):
    # 의문사로 시작하거나 '?'로 끝나면 질문으로 판단
    question_words = ("how", "what", "why", "where", "when", "who", "which")
    sentence = sentence.strip().lower()

    # 1. 구두점 확인
    if sentence.endswith("?"):
        return True

    # 2. 의문사 확인
    if sentence.startswith(question_words):
        return True

    # 3. 조동사 확인
    if sentence.startswith(("is", "are", "does", "do", "can", "will", "should")):
        return True

    # 기본적으로 질문이 아닌 것으로 판단
    return False


#질문듦로 나누기
def split_questions_no_punctuation(text):
    # 구두점이 없는 경우 문장 토큰화 (NLTK 기반)
    sentences = sent_tokenize(text)
    questions = [sentence.strip() for sentence in sentences if sentence]
    return questions


#고유명사 가져오기
def extract_proper_nouns(sentence):
    # 문장 분석
    doc = nlp(sentence)
    proper_nouns = []

    # 토큰 중 고유명사(PROPN) 필터링
    for token in doc:
        if token.pos_ == "PROPN":  # 고유명사
            proper_nouns.append(token.text)

    return proper_nouns if len(proper_nouns) > 0 else ['']


#질문과 답으로 분리
def split_questions_and_answers(lines):
    questions = []
    answers = []
    current_question = None
    current_answer = []

    for line in lines:
        line = line.strip()
        if not line:  # 빈 줄은 무시
            continue
        if line.endswith("?"):  # 질문인지 확인
            if current_question and current_answer:
                answers.append(" ".join(current_answer))  # 이전 답변 저장
            current_question = line
            questions.append(current_question)
            current_answer = []  # 답변 초기화
        elif current_question:  # 답변으로 간주
            current_answer.append(line)

    # 마지막 답변 추가
    if current_answer:
        answers.append(" ".join(current_answer))

    return questions, answers


#질문인가?
def is_question(sentence) -> bool:
    # 의문사로 시작하거나 '?'로 끝나면 질문으로 판단
    question_words = ("how", "what", "why", "where", "when", "who", "which")
    sentence = sentence.strip().lower()

    # 1. 구두점 확인
    if sentence.endswith("?"):
        return True

    # 2. 의문사 확인
    if sentence.startswith(question_words):
        return True

    # 3. 조동사 확인
    if sentence.startswith(("is", "are", "does", "do", "can", "will", "should")):
        return True

    # 기본적으로 질문이 아닌 것으로 판단
    return False


#문장으로 나누기
def split_by_sentence(text: str) -> list[str]:
    sentences = []
    # 텍스트 분석 및 문장 분리
    doc = nlp(text)
    # 결과 출력
    for idx, sentence in enumerate(doc.sents, 1):
        sentences.append(str(sentence))

    return sentences


# 불완전하거나 소개 문장으로 보이는 패턴
def is_incomplete_or_introductory(sentence):
    patterns = [
        r"^\d+\.\s",  # 번호로 시작하는 문장 (예: "1. ", "3. ")
        r".*:\s*$",  # 콜론으로 끝나는 문장
        r".*\-\s*$",  # 대시로 끝나는 문장
        r".*\.\.\.\s*$",  # 점 세 개로 끝나는 문장
        r".*\b(strategy|approach|method|plan|overview|introduction)\b.*\.\s*$"  # 특정 단어 포함
    ]

    # 각 패턴에 대해 일치 여부 확인
    for pattern in patterns:
        if re.match(pattern, sentence.strip(), re.IGNORECASE):
            return True

    return False


def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # 특수문자 제거
    text = text.lower()  # 소문자로 변환
    tokens = word_tokenize(text)  # 단어 토큰화
    return tokens


# 3. 형태소 분석 및 주요 품사 태그 확인
def analyze_sentence(tokens):
    tagged = pos_tag(tokens)  # 품사 태깅
    noun_count = sum(1 for word, pos in tagged if pos in ["NN", "NNS", "NNP", "JJ"])  # 명사와 형용사 수
    verb_count = sum(1 for word, pos in tagged if pos.startswith("VB"))  # 동사 수
    return noun_count, verb_count


# 4. 주제 여부 예측 함수
def is_topic_sentence(sentence):
    tokens = preprocess_text(sentence.lower().strip())
    noun_count, verb_count = analyze_sentence(tokens)

    # 주제 문장 기준: 명사/형용사 비중 > 동사, 연결 단어 포함 여부
    has_connector = any(word in tokens for word in ["and", "with",
    "about", "for", "their", "why","why", "how", "steps", "ways",
    "justifications", "introduction", "services", "strategy"])
    return noun_count > verb_count and has_connector


# 두 문장 간 관련성 판단
def cosine_similarity_tfidf(sentence1, sentence2):
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]  # 유사도 점수 반환


semantic_similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # 간단한 모델
def semantic_similarity(sentence1, sentence2):
    # 사전 훈련된 Sentence Transformer 모델 사용

    embeddings = semantic_similarity_model.encode([sentence1, sentence2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return similarity[0][0]  # 유사도 점수 반환

