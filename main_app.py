from dotenv import load_dotenv
import os
from openai import OpenAI
import gradio as gr
from datetime import datetime
from pathlib import Path
import re

load_dotenv()

def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    return OpenAI(api_key=api_key)

OUTPUT_DIR = Path.home() / "ai-medical" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================
# Helpers
# =====================
def sanitize_filename(text: str, max_len: int = 40) -> str:
    if not text:
        return "untitled"
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len] if text else "untitled"


def save_output(tool_name: str, title_hint: str, content: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_hint = sanitize_filename(title_hint)
    filename = f"{timestamp}_{tool_name}_{safe_hint}.txt"
    path = OUTPUT_DIR / filename
    path.write_text(content or "", encoding="utf-8")
    return str(path)


def ask_model(system_prompt: str, user_input: str) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content


# =====================
# Prompts
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

WEEKLY_PROMPT = """
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

DISCHARGE_PROMPT = """
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

OR_PROMPT = """
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


# =====================
# Core functions
# =====================
def admission_stage1(chief_complaint, admission_purpose, history_text, labs):
    user_input = f"""
Chief complaint: {chief_complaint}

Admission purpose: {admission_purpose}

Known history:
{history_text}

Labs / imaging:
{labs}
"""
    result = ask_model(STAGE1_PROMPT, user_input)
    path = save_output("admission_stage1", chief_complaint, result)
    return result, path


def admission_stage2(
    chief_complaint,
    admission_purpose,
    history_text,
    labs,
    additional_history,
    pe_findings,
    extra_data
):
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
    result = ask_model(STAGE2_PROMPT, user_input)
    path = save_output("admission_stage2", chief_complaint, result)
    return result, path


def weekly_summary(events, previous_weekly):
    user_input = f"""
Weekly events and timeline:
{events}

Previous weekly summary (if any):
{previous_weekly}
"""
    result = ask_model(WEEKLY_PROMPT, user_input)
    path = save_output("weekly_summary", "weekly", result)
    return result, path


def discharge_note(weekly, final_events):
    user_input = f"""
Weekly summaries:
{weekly}

Final week events and timeline:
{final_events}
"""
    result = ask_model(DISCHARGE_PROMPT, user_input)
    path = save_output("discharge_note", "discharge", result)
    return result, path


def or_briefing(history, meds, surgery, extra):
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
    result = ask_model(OR_PROMPT, user_input)
    path = save_output("or_briefing", surgery, result)
    return result, path


# =====================
# Cross-fill helpers
# =====================
def copy_stage1_to_stage2(chief, purpose, history, labs):
    return chief, purpose, history, labs


def copy_weekly_to_discharge(weekly_output):
    return weekly_output


def copy_history_to_or(history_text):
    return history_text


# =====================
# Clear helpers
# =====================
def clear_admission_stage1():
    return "", "", "", "", "", None


def clear_admission_stage2():
    return "", "", "", "", "", "", "", "", None


def clear_weekly():
    return "", "", "", None


def clear_discharge():
    return "", "", "", None


def clear_or():
    return "", "", "", "", "", None


# =====================
# GUI
# =====================
with gr.Blocks(title="Clinical AI Workspace") as demo:
    gr.Markdown("# Clinical AI Workspace")
    gr.Markdown("Admission / Weekly / Discharge / OR Briefing")

    with gr.Tab("Admission Copilot"):
        gr.Markdown("## Stage 1: Interview Guide")
        with gr.Row():
            chief1 = gr.Textbox(label="Chief Complaint")
            purpose1 = gr.Textbox(label="Admission Purpose")
        history1 = gr.Textbox(label="Known History / Previous Notes (中英皆可)", lines=10)
        labs1 = gr.Textbox(label="Labs / Imaging (brief)", lines=5)

        with gr.Row():
            btn1 = gr.Button("Generate Interview Guide")
            btn1_clear = gr.Button("Clear")

        output1 = gr.Textbox(label="Stage 1 Checklist", lines=24)
        file1 = gr.File(label="Download Stage 1 Output")

        btn1.click(
            admission_stage1,
            inputs=[chief1, purpose1, history1, labs1],
            outputs=[output1, file1]
        )
        btn1_clear.click(
            clear_admission_stage1,
            outputs=[chief1, purpose1, history1, labs1, output1, file1]
        )

        gr.Markdown("## Stage 2: Final Admission Note")
        with gr.Row():
            chief2 = gr.Textbox(label="Chief Complaint")
            purpose2 = gr.Textbox(label="Admission Purpose")
        history2 = gr.Textbox(label="Known History / Previous Notes", lines=10)
        labs2 = gr.Textbox(label="Labs / Imaging", lines=5)
        add_hist = gr.Textbox(label="Additional History Obtained", lines=5)
        pe2 = gr.Textbox(label="Physical Examination Findings", lines=5)
        extra2 = gr.Textbox(label="Additional Data", lines=5)

        with gr.Row():
            btn_copy_to_stage2 = gr.Button("Copy Stage 1 Inputs to Stage 2")
            btn2 = gr.Button("Generate Final Admission Note")
            btn2_clear = gr.Button("Clear")

        output2 = gr.Textbox(label="Final Admission Note", lines=30)
        file2 = gr.File(label="Download Final Admission Note")

        btn_copy_to_stage2.click(
            copy_stage1_to_stage2,
            inputs=[chief1, purpose1, history1, labs1],
            outputs=[chief2, purpose2, history2, labs2]
        )
        btn2.click(
            admission_stage2,
            inputs=[chief2, purpose2, history2, labs2, add_hist, pe2, extra2],
            outputs=[output2, file2]
        )
        btn2_clear.click(
            clear_admission_stage2,
            outputs=[chief2, purpose2, history2, labs2, add_hist, pe2, extra2, output2, file2]
        )

    with gr.Tab("Weekly Summary"):
        weekly_events = gr.Textbox(label="This week's events & timeline", lines=15)
        weekly_prev = gr.Textbox(label="Previous weekly summary (optional)", lines=10)

        with gr.Row():
            weekly_btn = gr.Button("Generate Weekly Summary")
            weekly_clear = gr.Button("Clear")

        weekly_out = gr.Textbox(label="Weekly Summary", lines=25)
        weekly_file = gr.File(label="Download Weekly Summary")

        weekly_btn.click(
            weekly_summary,
            inputs=[weekly_events, weekly_prev],
            outputs=[weekly_out, weekly_file]
        )
        weekly_clear.click(
            clear_weekly,
            outputs=[weekly_events, weekly_prev, weekly_out, weekly_file]
        )

    with gr.Tab("Discharge Note"):
        discharge_weekly = gr.Textbox(label="All weekly summaries (optional)", lines=15)
        discharge_events = gr.Textbox(label="Final week events & timeline", lines=10)

        with gr.Row():
            discharge_copy = gr.Button("Copy Weekly Output Here")
            discharge_btn = gr.Button("Generate Course and Treatment")
            discharge_clear = gr.Button("Clear")

        discharge_out = gr.Textbox(label="Course and Treatment", lines=20)
        discharge_file = gr.File(label="Download Course and Treatment")

        discharge_copy.click(
            copy_weekly_to_discharge,
            inputs=[weekly_out],
            outputs=[discharge_weekly]
        )
        discharge_btn.click(
            discharge_note,
            inputs=[discharge_weekly, discharge_events],
            outputs=[discharge_out, discharge_file]
        )
        discharge_clear.click(
            clear_discharge,
            outputs=[discharge_weekly, discharge_events, discharge_out, discharge_file]
        )

    with gr.Tab("OR Briefing"):
        or_history = gr.Textbox(label="Brief History（病史）", lines=6)
        or_meds = gr.Textbox(label="Current Medications（用藥）", lines=5)
        or_surgery = gr.Textbox(label="Planned Surgery（手術名稱）", lines=3)
        or_extra = gr.Textbox(label="Additional Info（labs / image 可選）", lines=5)

        with gr.Row():
            or_copy_history = gr.Button("Copy Admission History Here")
            or_btn = gr.Button("Generate OR Briefing")
            or_clear = gr.Button("Clear")

        or_out = gr.Textbox(label="OR Briefing", lines=30)
        or_file = gr.File(label="Download OR Briefing")

        or_copy_history.click(
            copy_history_to_or,
            inputs=[history2],
            outputs=[or_history]
        )
        or_btn.click(
            or_briefing,
            inputs=[or_history, or_meds, or_surgery, or_extra],
            outputs=[or_out, or_file]
        )
        or_clear.click(
            clear_or,
            outputs=[or_history, or_meds, or_surgery, or_extra, or_out, or_file]
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    username = os.environ.get("CLINICAL_AI_USERNAME", "admin")
    password = os.environ.get("CLINICAL_AI_PASSWORD", "change-this-password")

    print("DEBUG OPENAI exists:", bool(os.environ.get("OPENAI_API_KEY")))
    print("DEBUG CLINICAL_AI_USERNAME:", repr(username))
    print("DEBUG CLINICAL_AI_PASSWORD length:", len(password))
    print("DEBUG matching env keys:", [k for k in os.environ.keys() if "CLINICAL" in k or "OPENAI" in k])

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        auth=(username, password)
    )
