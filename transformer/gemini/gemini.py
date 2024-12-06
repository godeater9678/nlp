import requests
# import google.auth
# from google.auth.transport.requests import Request
#
# # Google Cloud 프로젝트 정보
# PROJECT_ID = "chatbottonkni"
# REGION = "us-central1"
#
# # URL 구성
# BASE_URL = f"https://{REGION}-aiplatform.googleapis.com/v1"
# MODEL_NAME = "publishers/google/models/gemini-1.5-pro:generateText"
# URL = f"{BASE_URL}/projects/{PROJECT_ID}/locations/{REGION}/{MODEL_NAME}"

import google.generativeai as genai
#GOOGLE_API_KEY='AIzaSyAkfbQcjaVxAMz2MjP-jkz8wmroiXVjKOk' #gemini.api.key.txt
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("가위바위보 파이썬 코드 작성해줘")
print(response.text)