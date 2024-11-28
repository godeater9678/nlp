import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util.transrator import korean_to_english, english_to_korean

model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")

input_text = korean_to_english("현범이는 누구? 좋아하는건?")
#input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]


def generate_with_uncertainty(model, tokenizer, input_text, confidence_threshold=0.5):
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
    outputs = model.generate(
        input_ids=input_ids,
        max_length=200,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id
    )

    # 출력 토큰 및 점수 가져오기
    generated_ids = outputs.sequences[0]
    scores = outputs.scores

    # # 각 답변 출력
    # for idx, output in enumerate(outputs):
    #     aa = tokenizer.decode(output, skip_special_tokens=True)
    #     print(f"답변 {idx + 1}: {aa}")


    # 토큰별 확률 계산
    probabilities = torch.softmax(torch.cat(scores, dim=0), dim=-1)
    mean_confidence = probabilities.max(dim=-1).values.mean().item()

    # 확률 기반으로 확신 여부 판단
    if mean_confidence < confidence_threshold:
        return korean_to_english("죄송합니다. 그 질문에 대해 잘 모르겠습니다.")
    else:
        return tokenizer.decode(generated_ids, skip_special_tokens=True)


#output = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.pad_token_id)
#response = tokenizer.decode(output[0], skip_special_tokens=True)
response = generate_with_uncertainty(model, tokenizer, input_text)
print(english_to_korean(response))
