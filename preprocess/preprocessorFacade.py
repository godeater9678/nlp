from preprocess.preprocessor import (
    split_by_sentence,
    read_from_txt_korean,
    is_incomplete_or_introductory,
    split_questions_and_answers,
    is_question,
    is_topic_sentence,
    semantic_similarity,
    is_topic_sentence
)



def preprocessorForDocumentText(lines: list[str]) -> list[any]:
    #lines = read_from_txt_korean('./transformer/chatBot/contents.txt')
    sentences = []
    question = ''
    answer = ''
    for line in lines:
        for sentence in split_by_sentence(line):
            isQuestion: bool = is_question(sentence)
            isDefine: bool = is_incomplete_or_introductory(sentence) or is_topic_sentence(sentence)

            if isQuestion and answer == '':  # 답변이 없는 상태에서 질문이 왔다면 연속된 질문이다.
                question = sentence if question == '' else f'{question} {sentence}'
            elif isQuestion and answer != '':  # 답변이 있는 상태에서 새로운 질문이라면 이전 질문을 종료하고 새로운 질문으로 간다.
                sentences.append({'text': f'{question} {answer}'})
                question = sentence
                answer = ''
            elif isDefine and question == '':
                question = sentence
            elif semantic_similarity(question + answer, sentence) > 0.2:  # 기타 답변문장이 질답과 연관성이 있다고 판단돼야 연속적인 답변이 된다.
                answer = sentence if answer == '' else f'{answer} {sentence}'
            else:
                if question != '' and answer != '':
                    sentences.append({'text': f'{question} {answer}'})
                    question = ''
                    answer = ''
                else:
                    print("버림: " + sentence)
    if question != '' and answer == '':
        sentences.append({'text': f'{question} {answer}'})

    return sentences