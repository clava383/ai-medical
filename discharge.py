from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """
You are a clinical assistant generating discharge summaries.

STRICT RULES:
- Do NOT hallucinate
- Only use provided information
- Do NOT include pre-admission history

TASK:
Generate ONLY:

【Course and Treatment】

FORMAT:

【Course and Treatment】

- Narrative paragraph
- Start directly from hospitalization
- Chronological and logical
- Emphasize:
    - treatments
    - responses
    - complications
- No bullet points
- No extra sections

STYLE:
- Professional clinical English
- Smooth and natural
- Similar to discharge summary or case report
- Concise but complete

STRUCTURE GUIDELINE:

During hospitalization, ...

Subsequently, ...

During the hospital course, ...

At the time of discharge, ...
"""

def generate_discharge(weekly, final_events):

    user_input = f"""
Weekly summaries:
{weekly}

Final week events and timeline:
{final_events}
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
    fn=generate_discharge,
    inputs=[
        gr.Textbox(label="All weekly summaries (optional)", lines=15),
        gr.Textbox(label="Final week events & timeline", lines=10),
    ],
    outputs=gr.Textbox(label="Course and Treatment", lines=20),
    title="Discharge Note Generator (Course & Treatment)",
    description="輸入 weekly + 最後一週事件 → 生成 narrative discharge course"
)

if __name__ == "__main__":
    demo.launch()
