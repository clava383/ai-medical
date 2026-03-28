from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv()
client = OpenAI()

# =====================
# Stage 1 Prompt
# =====================
STAGE1_PROMPT = """
You are a clinical admission assistant.

Your task is NOT to write the admission note yet.

You must generate a pre-admission checklist in MIXED Chinese-English style.

GOALS:
1. Identify missing critical information
2. Generate practical questions to ask the patient
3. Suggest focused physical examination items

STRICT RULES:
- Do NOT hallucinate
- Only use clinically relevant items
- Be concise, practical, and checklist-oriented
- Use mixed Chinese-English wording
- Output should be easy for a PGY to use during bedside history taking
- Do NOT write long paragraphs

OUTPUT FORMAT EXACTLY:

[Missing Information 待補資訊]
[ ] item 1
[ ] item 2
[ ] item 3

[Questions to Ask 建議追問]
[ ] item 1
[ ] item 2
[ ] item 3

[Focused PE Checklist 重點身體診察]
[ ] item 1
[ ] item 2
[ ] item 3

STYLE RULES:
- Each line must begin with [ ]
- Keep each item short
- Use mixed Chinese-English
- Prioritize high-yield admission questions
- PE section should focus on meaningful positive findings to look for
"""

# =====================
# Stage 2 Prompt
# =====================
STAGE2_PROMPT = """
You are a clinical admission assistant.

STRICT RULES:
- Do NOT hallucinate
- If information is missing → write UNKNOWN
- DO NOT fabricate physical exam or lab
- Use formal clinical English
- Follow EXACT formatting rules

========================
FORMAT REQUIREMENTS
========================

一. 主訴(Chief Complaint)
→ Use given chief complaint

二. 病史(Brief History)

【Present illness】
- MUST start with:
"This is a XX-year-old man/woman with the following underlying diseases:"
- Each disease must start with "#"
- Treatment history must use "- status post ..."
- Chronological narrative
- Match real admission style

【Past History】
→ DO NOT generate (leave blank or minimal)

三. 系統性回顧(Review of Systems)

→ DO NOT output full ROS
→ ONLY output:

[ROS Positive Findings]
- list only positive symptoms

四. 身體診察(Physical Examination)

→ DO NOT output full exam
→ ONLY output:

[Physical Examination Positive Findings]
- list only abnormal findings

五. 檢驗紀錄(Laboratory Report)
→ DO NOT generate

六. 檢查紀錄(Examination Report)
→ DO NOT generate

七. 影像報告(Imaging Report)
→ DO NOT generate

八. 病理報告(Pathology Report)
→ DO NOT generate

九. 臆斷(Tentative Diagnosis)

→ COPY the disease list from Present illness (DO NOT MODIFY)

【Actives】
- list currently active diseases

【Underlyings】
- list chronic diseases

十. 醫療需求與治療計畫(Medical Needs and Care Plan)

【Subjective】
→ Chief complaint

【Objective】

[Physical examination]
→ key findings only

[Image]
→ summarize key imaging

[Lab]

→ MUST include TWO parts:

1. Interpretation (brief summary)

2. Structured lab format EXACTLY as below:

Hb/Plt/WBC: Hb_value/Plt_value/WBC_value
Alb: value
AST/ALT: value/value
BUN/CRE: value/value
Na/K: value/value

PT/aPTT/INR: value/value/value

RULES:
- Follow EXACT order and format
- If any value is missing → write UNKNOWN
- Do NOT change labels
- Do NOT add units
- Prefer most recent lab values

【Assessment】
→ SAME as Tentative Diagnosis (copy)

【Plan】
→ ONLY for Actives
→ FORMAT:

# Disease name
- treatment
- monitoring
- supportive care

========================
STYLE
========================
- Match hematology admission style
- Structured, clean, professional
"""
# =====================
# Stage 1 Function
# =====================
def stage1(chief_complaint, admission_purpose, history_text, labs):

    user_input = f"""
Chief complaint: {chief_complaint}
Admission purpose: {admission_purpose}

Known history:
{history_text}

Labs / imaging:
{labs}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": STAGE1_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )

    return response.choices[0].message.content


# =====================
# Stage 2 Function
# =====================
def stage2(chief_complaint, admission_purpose, history_text, labs,
           additional_history, pe_findings, extra_data):

    user_input = f"""
Chief complaint: {chief_complaint}
Admission purpose: {admission_purpose}

Known history:
{history_text}

Labs / imaging:
{labs}

Additional history obtained:
{additional_history}

Physical examination findings:
{pe_findings}

Additional data:
{extra_data}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": STAGE2_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )

    return response.choices[0].message.content


# =====================
# GUI
# =====================
with gr.Blocks() as demo:

    gr.Markdown("## Clinical Admission Copilot (Two-Stage Version)")

    with gr.Tab("Stage 1: Interview Guide"):
        chief1 = gr.Textbox(label="Chief Complaint")
        purpose1 = gr.Textbox(label="Admission Purpose")
        history1 = gr.Textbox(label="Known History / Previous Notes (中英皆可)", lines=10)
        labs1 = gr.Textbox(label="Labs / Imaging (brief)", lines=5)

        btn1 = gr.Button("Generate Interview Guide")

        output1 = gr.Textbox(label="Stage 1 Checklist", lines=28)

        btn1.click(
            stage1,
            inputs=[chief1, purpose1, history1, labs1],
            outputs=output1
        )

    with gr.Tab("Stage 2: Final Admission"):
        chief2 = gr.Textbox(label="Chief Complaint")
        purpose2 = gr.Textbox(label="Admission Purpose")
        history2 = gr.Textbox(label="Known History / Previous Notes", lines=10)
        labs2 = gr.Textbox(label="Labs / Imaging", lines=5)

        add_hist = gr.Textbox(label="Additional History Obtained", lines=5)
        pe = gr.Textbox(label="Physical Examination Findings", lines=5)
        extra = gr.Textbox(label="Additional Data", lines=5)

        btn2 = gr.Button("Generate Final Admission Note")

        output2 = gr.Textbox(label="Final Admission Note", lines=30)

        btn2.click(
            stage2,
            inputs=[chief2, purpose2, history2, labs2, add_hist, pe, extra],
            outputs=output2
        )

if __name__ == "__main__":
    demo.launch()
