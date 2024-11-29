from preprocess.preprocessor import (
    split_by_sentence,
    read_from_txt_korean,
    is_incomplete_or_introductory,
    split_questions_and_answers,
    is_question,
    is_topic_sentence,
    semantic_similarity
)
from preprocess.preprocessorFacade import preprocessorForDocumentText

sentences = preprocessorForDocumentText(read_from_txt_korean('./transformer/chatBot/contents.txt'))


print(sentences)

#
# # 예제 문장
# sentences = [
#     "3. Initial Content Seeding Strategy.",
#     "This is the final conclusion.",
#     "The key benefits include:",
#     "Benefits -",
#     "An overview of the process...",
#     "Hyunbeom is learning Python."
# ]
#
# # 판단
# for sentence in sentences:
#     result = is_incomplete_or_introductory(sentence)
#     print(f"'{sentence}' -> {'Introductory or incomplete' if result else 'Complete'}")
