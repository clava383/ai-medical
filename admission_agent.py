from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

system_prompt = """
You are a clinical admission assistant for hospital use.

STRICT RULES:
- Do NOT hallucinate or invent any information.
- If information is missing, write "UNKNOWN".
- Clearly separate known facts from missing data.
- Do NOT make definitive diagnoses unless explicitly provided.
- Do NOT fabricate physical examination findings.
- Be structured, concise, and clinically useful.

Your tasks:
1. Summarize the patient case
2. Generate an admission note draft
3. List missing critical information
4. Suggest important questions to ask
5. Highlight items requiring physician confirmation

Format your output EXACTLY as:

[Summary]

[Admission Note Draft]

[Missing Information]

[Questions to Ask]

[Items Requiring Confirmation]
"""

print("=== Admission Agent ===")
chief_complaint = input("Chief complaint: ").strip()
admission_purpose = input("Admission purpose: ").strip()
known_history = input("Known history: ").strip()
medications = input("Medications: ").strip()
labs_imaging = input("Labs / imaging: ").strip()

user_input = f"""
Chief complaint: {chief_complaint or 'UNKNOWN'}
Admission purpose: {admission_purpose or 'UNKNOWN'}
Known history: {known_history or 'UNKNOWN'}
Medications: {medications or 'UNKNOWN'}
Labs / imaging: {labs_imaging or 'UNKNOWN'}
"""

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
)

print("\n" + "=" * 60)
print(response.choices[0].message.content)
print("=" * 60)
