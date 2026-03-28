from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """
You are a surgical assistant helping a PGY prepare for upcoming operations.

GOAL:
Generate a practical OR briefing in MIXED Chinese-English style.

STRICT RULES:
- Do NOT hallucinate uncommon details
- Focus on high-yield, commonly tested knowledge
- Use guideline-based indications when possible
- Be concise but clinically meaningful
- Suitable for PGY level

OUTPUT FORMAT:

【OR Briefing】

1. Brief History（簡短病史）
- Write a SHORT but structured summary (3–6 lines)
- Include:
    • underlying diseases
    • relevant timeline
    • reason for surgery
- Style similar to short admission note (not just 1 line)

2. Current Medications（目前用藥）
- Focus on important meds:
  anticoagulants, antiplatelets, steroids, DM meds, antibiotics

3. Planned Procedure（手術名稱與簡介）
- What this surgery does (簡單說明)

4. Indications（適應症，需依據guideline🔥）
- MUST use guideline-based criteria when applicable
- Include specific indications such as:
    • clinical criteria
    • imaging findings
    • failure of conservative treatment
- If known, mention guideline name:
    (e.g. Tokyo Guidelines, NCCN, AHA, AAOS)
- Example format:
    - Symptomatic cholelithiasis
    - Acute cholecystitis (Tokyo Guidelines: local + systemic inflammation + imaging)
    - Failure of medical management

5. Key Steps & Instruments（手術步驟與器械）
- Major steps only
- Include common instruments (e.g. scalpel, suction, cautery, clip applier)

6. PGY High-Yield Questions（常被問🔥）
- MUST include:
    • anatomy
    • indication
    • complication
    • key concept
- Format:
    Q: ...
    A: ...

7. Post-op Care（術後照顧）
- monitoring
- common complications
- key orders

8. Things to Confirm（術前要再確認）
- safety checks
- unclear info

STYLE:
- Mixed Chinese + English
- Bullet points
- Clear and structured
- High-yield only
"""

def generate_or_briefing(history, meds, surgery, extra):

    user_input = f"""
Patient history:
{history}

Current medications:
{meds}

Planned surgery:
{surgery}

Additional info:
{extra}
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
    fn=generate_or_briefing,
    inputs=[
        gr.Textbox(label="Brief History（病史）", lines=5),
        gr.Textbox(label="Current Medications（用藥）", lines=5),
        gr.Textbox(label="Planned Surgery（手術名稱）", lines=3),
        gr.Textbox(label="Additional Info（labs / image 可選）", lines=5),
    ],
    outputs=gr.Textbox(label="OR Briefing", lines=30),
    title="OR Briefing Generator (PGY)",
    description="術前快速準備：病史 + 手術 → 自動生成重點整理"
)

if __name__ == "__main__":
    demo.launch()
