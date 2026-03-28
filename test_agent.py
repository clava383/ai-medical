import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are a medical assistant."},
        {"role": "user", "content": "Summarize: 65M with DM, admitted for chest pain."}
    ]
)

print(response.choices[0].message.content)
