from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """
You are a clinical assistant generating weekly summaries.

STRICT RULES:
- Do NOT hallucinate
- Only use provided information
- Do NOT include events before admission

TASK:
Generate TWO versions:

========================
1. Structured Weekly Summary
========================

Format:

【Weekly Summary】

Since admission, the patient has undergone the following clinical course:

- timeline events

During this period, the clinical course was characterized by:
- key clinical features

Currently, the patient is:
- current status


========================
2. Narrative Clinical Course
========================

Format:

【Narrative Clinical Course】

- DO NOT include pre-admission history
- START DIRECTLY with hospitalization course

- MUST follow this structure:

During hospitalization, the clinical course was characterized by ...

Subsequently, ...

Overall, the patient's condition ...

RULES:
- Paragraph form
- Logical and chronological
- Highlight treatment → response → complications
- No unnecessary background
- Professional clinical English
- Concise and smooth
"""

def generate_weekly(events, previous_weekly):

    user_input = f"""
Weekly events and timeline:
{events}

Previous weekly summary (if any):
{previous_weekly}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )

    return response.choices[0].message.content


demo = gr.Interface(
    fn=generate_weekly,
    inputs=[
        gr.Textbox(label="This week's events & timeline", lines=15),
        gr.Textbox(label="Previous weekly summary (optional)", lines=10),
    ],
    outputs=gr.Textbox(label="Weekly Summary", lines=25),
    title="Weekly Summary Generator (Clinical)",
    description="輸入本週事件 → 生成 structured + narrative weekly summary"
)

if __name__ == "__main__":
    demo.launch()
