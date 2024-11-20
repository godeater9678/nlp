#https://github.com/google/sentencepiece#installation

#영어를 불어로

from transformers import MarianMTModel, MarianTokenizer

tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # "Bonjour, comment ça va?"
