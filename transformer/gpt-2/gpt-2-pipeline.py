from transformers import pipeline

# Q&A 파이프라인 생성 (한국어 지원 모델 로드)
qa_model = pipeline(
    "question-answering",
    model="deepset/xlm-roberta-large-squad2",
    tokenizer="deepset/xlm-roberta-large-squad2",
    ignore_mismatched_sizes=True  # 경고 억제
)


def answer_question(question, context):
    """
    Q&A 특화 모델을 사용해 질문에 대한 답변 생성
    Args:
        question (str): 사용자 질문
        context (str): 문맥
    Returns:
        str: 생성된 답변
    """
    result = qa_model(question=question, context=context)
    return result['answer']


if __name__ == "__main__":
    # 문맥을 준비합니다.
    context = """
    인공지능은 인간의 지능을 모방하여 학습, 추론, 문제 해결을 수행하는 기술입니다.
    특히 자연어 처리와 컴퓨터 비전에서 뛰어난 성과를 보이고 있습니다.
    """

    print("Q&A 프로그램입니다. '종료'를 입력하면 종료됩니다.")
    while True:
        user_question = input("질문: ")
        if user_question.strip().lower() in ["종료", "exit", "quit"]:
            print("프로그램을 종료합니다.")
            break
        response = answer_question(user_question, context)
        print(f"답변: {response}")
