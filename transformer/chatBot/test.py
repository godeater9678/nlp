import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import spacy

from preprocess.preprocessor import split_questions_no_punctuation, extract_proper_nouns
from util.translator import korean_to_english, english_to_korean

model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")


def generate_with_uncertainty(model, tokenizer, input_text, confidence_threshold=0.2):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=inputs["attention_mask"],
        max_length=200,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.2,  # 반복 억제
    )

    # 출력 토큰 및 점수 가져오기
    generated_ids = outputs.sequences[0]
    scores = outputs.scores

    # 토큰별 확률 계산
    probabilities = torch.softmax(torch.cat(scores, dim=0), dim=-1)
    mean_confidence = probabilities.max(dim=-1).values.mean().item()

    # 확률 기반으로 확신 여부 판단
    if mean_confidence < confidence_threshold:
        return None
    else:
        return tokenizer.decode(generated_ids, skip_special_tokens=True)


input_text = korean_to_english("현범이는 누구에요?")
spl = split_questions_no_punctuation(input_text) #문장 나누기
spc = extract_proper_nouns(input_text)[0] if len(spl) > 0 else '' #명사 가져오기
for question in spl:
    question = question if spc in question else f'{spc}, {question}'
    response = generate_with_uncertainty(model, tokenizer, question)
    print(f'Question: {question}')
    if response is not None:
        print(english_to_korean(response))
    else:
        print("죄송합니다. 그 질문에 대해 잘 모르겠습니다.")
