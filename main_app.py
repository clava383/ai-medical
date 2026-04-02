from dotenv import load_dotenv
import os
import json
import uuid
import secrets
import hashlib
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from openai import OpenAI
import gradio as gr

load_dotenv()


# =========================================================
# Config
# =========================================================
APP_DATA_DIR = Path(os.environ.get("APP_DATA_DIR", "/tmp/ai-medical-data"))
USERS_DB_PATH = APP_DATA_DIR / "users.json"
USER_DATA_DIR = APP_DATA_DIR / "users"

APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

if not USERS_DB_PATH.exists():
    USERS_DB_PATH.write_text(json.dumps({"users": []}, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================================================
# LLM client
# =========================================================
def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    return OpenAI(api_key=api_key)


# =========================================================
# Helpers
# =========================================================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sanitize_filename(text: str, max_len: int = 40) -> str:
    if not text:
        return "untitled"
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len] if text else "untitled"


def empty_case_form():
    return "", "", "", ""


def empty_stage1():
    return "", "", "", "", "", "", "", ""


def empty_stage2():
    return "", "", "", "", "", "", "", "", "", "", ""


def empty_weekly():
    return "", "", ""


def empty_discharge():
    return "", "", ""


def empty_or():
    return "", "", "", "", ""


def empty_handoff():
    return "", "", "", ""


def default_case_record(mrn: str, name: str, age: str, sex: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "mrn": mrn.strip(),
        "name": name.strip(),
        "age": age.strip(),
        "sex": sex.strip(),
        "status": "active",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "discharged_at": "",
        "admission": {
            "stage1": {
                "chief_complaint": "",
                "admission_purpose": "",
                "history_text": "",
                "outpatient_notes": "",
                "emergency_notes": "",
                "labs": "",
                "timeline_output": "",
                "checklist_output": "",
                "output": "",
            },
            "stage2": {
                "chief_complaint": "",
                "admission_purpose": "",
                "history_text": "",
                "outpatient_notes": "",
                "emergency_notes": "",
                "timeline_text": "",
                "labs": "",
                "additional_history": "",
                "pe_findings": "",
                "extra_data": "",
                "output": "",
            },
        },
        "weekly": {
            "events": "",
            "previous_weekly": "",
            "output": "",
        },
        "discharge": {
            "weekly": "",
            "final_events": "",
            "output": "",
        },
        "or_briefing": {
            "history": "",
            "meds": "",
            "surgery": "",
            "extra": "",
            "output": "",
        },
        "handoff": {
            "problem": "",
            "assessment": "",
            "plan": "",
            "output": "",
        },
    }



def sanitize_username(username: str) -> str:
    username = (username or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
        raise gr.Error("username 需為 3-32 字元，只能包含英文、數字、底線、點或減號。")
    return username


def user_case_db_path(username: str) -> Path:
    safe_username = sanitize_username(username)
    user_dir = USER_DATA_DIR / safe_username
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / "cases.json"
    if not path.exists():
        path.write_text(json.dumps({"cases": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_users_db() -> dict:
    try:
        return json.loads(USERS_DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"users": []}


def save_users_db(db: dict) -> None:
    USERS_DB_PATH.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt.hex(), dk.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    _, candidate = hash_password(password, salt_hex)
    return secrets.compare_digest(candidate, hash_hex)


def get_user_record(username: str):
    db = load_users_db()
    for user in db.get("users", []):
        if user.get("username") == username:
            return user
    return None


def register_user(username: str, password: str, confirm_password: str):
    username = sanitize_username(username)
    password = (password or "").strip()
    confirm_password = (confirm_password or "").strip()

    if len(password) < 6:
        raise gr.Error("密碼至少需要 6 個字元。")
    if password != confirm_password:
        raise gr.Error("兩次輸入的密碼不一致。")
    if get_user_record(username):
        raise gr.Error("這個 username 已經被使用。")

    db = load_users_db()
    salt_hex, hash_hex = hash_password(password)
    db.setdefault("users", []).append(
        {
            "id": str(uuid.uuid4()),
            "username": username,
            "password_salt": salt_hex,
            "password_hash": hash_hex,
            "created_at": now_iso(),
        }
    )
    save_users_db(db)
    user_case_db_path(username)
    return f"註冊成功：{username}，現在可以登入。"


def authenticate_user(username: str, password: str) -> bool:
    username = (username or "").strip()
    password = (password or "").strip()
    user = get_user_record(username)
    if not user:
        return False
    return verify_password(password, user["password_salt"], user["password_hash"])


def admin_user_choices(keyword: str = ""):
    db = load_users_db()
    users = sorted(db.get("users", []), key=lambda x: x.get("created_at", ""), reverse=True)
    keyword = (keyword or "").strip().lower()
    results = []
    for u in users:
        username = u.get("username", "")
        if keyword and keyword not in username.lower():
            continue
        results.append((username, username))
    return results


def admin_case_choices(username: str, keyword: str = ""):
    if not username:
        return []
    db = load_db(username)
    cases = sorted(db.get("cases", []), key=lambda x: x.get("updated_at", ""), reverse=True)
    keyword = (keyword or "").strip().lower()
    choices = []
    for case in cases:
        label = f"{case.get('mrn','')} | {case.get('name','')} | {case.get('age','')} | {case.get('sex','')} | {case.get('status','')}"
        haystack = " ".join([
            str(case.get("mrn", "")),
            str(case.get("name", "")),
            str(case.get("age", "")),
            str(case.get("sex", "")),
            str(case.get("status", "")),
        ]).lower()
        if keyword and keyword not in haystack:
            continue
        choices.append((label, case.get("id", "")))
    return choices


def admin_case_preview_md(username: str, case_id: str) -> str:
    if not username or not case_id:
        return "### Case Preview\n\n_尚未選擇 case。_"
    db = load_db(username)
    case = next((c for c in db.get("cases", []) if c.get("id") == case_id), None)
    if not case:
        return "### Case Preview\n\n_找不到這個 case。_"

    lines = [
        "### Case Preview",
        "",
        f"- **MRN:** {case.get('mrn','')}",
        f"- **Name:** {case.get('name','')}",
        f"- **Age / Sex:** {case.get('age','')} / {case.get('sex','')}",
        f"- **Status:** {case.get('status','')}",
        f"- **Created:** {case.get('created_at','')}",
        f"- **Updated:** {case.get('updated_at','')}",
        "",
        "#### Available Outputs",
        f"- Admission Stage 1: {'Yes' if case.get('admission',{}).get('stage1',{}).get('output') else 'No'}",
        f"- Admission Stage 2: {'Yes' if case.get('admission',{}).get('stage2',{}).get('output') else 'No'}",
        f"- Weekly: {'Yes' if case.get('weekly',{}).get('output') else 'No'}",
        f"- Discharge: {'Yes' if case.get('discharge',{}).get('output') else 'No'}",
        f"- OR Briefing: {'Yes' if case.get('or_briefing',{}).get('output') else 'No'}",
        f"- Handoff: {'Yes' if case.get('handoff',{}).get('output') else 'No'}",
    ]
    return "\n".join(lines)


def refresh_admin_user_search(current_username: str, keyword: str):
    if not is_admin_user(current_username):
        raise gr.Error("只有 admin 可以使用後台。")
    return gr.update(choices=admin_user_choices(keyword), value=None), "### User Cases\n\n_尚未選擇使用者。_", "", "", "", gr.update(choices=[], value=None), "### Case Preview\n\n_尚未選擇 case。_"


def refresh_admin_case_list(current_username: str, target_username: str, keyword: str):
    if not is_admin_user(current_username):
        raise gr.Error("只有 admin 可以使用後台。")
    return gr.update(choices=admin_case_choices(target_username, keyword), value=None), "### Case Preview\n\n_尚未選擇 case。_"


def load_admin_case_preview(current_username: str, target_username: str, case_id: str):
    if not is_admin_user(current_username):
        raise gr.Error("只有 admin 可以使用後台。")
    return admin_case_preview_md(target_username, case_id)

def summarize_user_cases(username: str) -> str:
    if not username:
        return "### User Cases\n\n_尚未選擇使用者。_"
    db = load_db(username)
    cases = db.get("cases", [])
    lines = [f"### {username} 的病例", "", f"- 帳號：{username}", f"- 病例數：{len(cases)}", ""]
    if not cases:
        lines.append("_沒有病例資料。_")
        return "\n".join(lines)
    for case in cases[:50]:
        lines.append(
            f"- **{case.get('mrn','')}** | {case.get('name','')} | {case.get('age','')} | {case.get('sex','')}  \\n"
            f"  status: {case.get('status','')} · updated {case.get('updated_at','')}"
        )
    if len(cases) > 50:
        lines.append(f"\n_And {len(cases)-50} more cases..._")
    return "\n".join(lines)


def load_admin_user_data(current_username: str, target_username: str):
    if not is_admin_user(current_username):
        raise gr.Error("只有 admin 可以使用後台。")
    if not target_username:
        return (
            "### User Cases\n\n_尚未選擇使用者。_",
            "",
            "",
            "尚未選擇使用者。",
            gr.update(choices=[], value=None),
            "### Case Preview\n\n_尚未選擇 case。_",
        )
    user_record = get_user_record(target_username)
    user_db = load_db(target_username)
    pretty_user = json.dumps(user_record or {}, ensure_ascii=False, indent=2)
    pretty_cases = json.dumps(user_db, ensure_ascii=False, indent=2)
    return (
        summarize_user_cases(target_username),
        pretty_user,
        pretty_cases,
        f"已載入使用者資料：{target_username}",
        gr.update(choices=admin_case_choices(target_username), value=None),
        "### Case Preview\n\n_尚未選擇 case。_",
    )


def delete_registered_user(current_username: str, target_username: str):
    if not is_admin_user(current_username):
        raise gr.Error("只有 admin 可以刪除帳號。")
    target_username = (target_username or "").strip()
    if not target_username:
        raise gr.Error("請先選擇要刪除的帳號。")
    if is_admin_user(target_username):
        raise gr.Error("不能刪除 admin 帳號。")

    db = load_users_db()
    before = len(db.get("users", []))
    db["users"] = [u for u in db.get("users", []) if u.get("username") != target_username]
    if len(db["users"]) == before:
        raise gr.Error("找不到這個帳號。")
    save_users_db(db)

    import shutil
    user_dir = USER_DATA_DIR / sanitize_username(target_username)
    if user_dir.exists():
        shutil.rmtree(user_dir, ignore_errors=True)

    return (
        gr.update(choices=admin_user_choices(), value=None),
        "### User Cases\n\n_尚未選擇使用者。_",
        "",
        "",
        f"已刪除帳號：{target_username}",
        gr.update(choices=[], value=None),
        "### Case Preview\n\n_尚未選擇 case。_",
    )


def load_db(username: str) -> dict:
    try:
        return json.loads(user_case_db_path(username).read_text(encoding="utf-8"))
    except Exception:
        return {"cases": []}


def save_db(username: str, db: dict) -> None:
    user_case_db_path(username).write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")


def get_case_by_id(username: str, case_id: str):
    db = load_db(username)
    for case in db.get("cases", []):
        if case["id"] == case_id:
            return case
    return None


def update_case(username: str, case_id: str, mutate_fn):
    db = load_db(username)
    for idx, case in enumerate(db.get("cases", [])):
        if case["id"] == case_id:
            case_copy = deepcopy(case)
            mutate_fn(case_copy)
            case_copy["updated_at"] = now_iso()
            db["cases"][idx] = case_copy
            save_db(username, db)
            return case_copy
    raise ValueError("Case not found")


def ask_model(system_prompt: str, user_input: str) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content or ""


def split_stage1_output(raw_text: str) -> tuple[str, str]:
    raw_text = raw_text or ""
    timeline = ""
    checklist = raw_text.strip()

    pattern = re.compile(
        r"\[Current History Timeline 目前病史時間軸整理\](.*?)(?=\n\[Admission Checklist 入院前檢查清單\]|\Z)",
        re.DOTALL,
    )
    m = pattern.search(raw_text)
    if m:
        timeline = m.group(1).strip()

    pattern2 = re.compile(
        r"\[Admission Checklist 入院前檢查清單\](.*)$",
        re.DOTALL,
    )
    m2 = pattern2.search(raw_text)
    if m2:
        checklist = m2.group(1).strip()
    elif timeline:
        checklist = raw_text.replace(m.group(0), "").strip()

    return timeline, checklist




def extract_diagnosis_blocks(text: str) -> list[str]:
    text = text or ""
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []

    def flush():
        nonlocal current
        if current:
            cleaned = "\n".join([ln.rstrip() for ln in current]).strip()
            if cleaned:
                blocks.append(cleaned)
            current = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("#."):
            flush()
            current = [stripped]
            continue

        if not current:
            continue

        if stripped.startswith("#."):
            flush()
            current = [stripped]
        elif stripped.startswith("*") or stripped.startswith("-"):
            current.append(stripped)
        elif not stripped:
            flush()
        elif line.startswith(" ") or line.startswith("	"):
            current.append(stripped)
        elif stripped.lower().startswith(("status post", "with recurrence", "s/p", "post-op", "post op")):
            current.append(stripped)
        else:
            flush()

    flush()
    return blocks


def normalize_dx_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def get_block_key(block: str) -> str:
    first_line = ""
    for line in (block or "").splitlines():
        if line.strip().startswith("#."):
            first_line = line.strip()
            break
    return normalize_dx_line(first_line)


def collect_latest_diagnosis_blocks(*texts: str) -> list[str]:
    latest_map: dict[str, str] = {}
    order: list[str] = []

    for text in texts:
        for block in extract_diagnosis_blocks(text or ""):
            key = get_block_key(block)
            if not key:
                continue
            if key in order:
                order.remove(key)
            order.append(key)
            latest_map[key] = block.strip()

    return [latest_map[key] for key in order if key in latest_map]


def is_major_treatment_line(line: str) -> bool:
    lower = (line or "").strip().lower()
    major_keywords = [
        "status post tumor excision",
        "status post excision",
        "status post resection",
        "status post operation",
        "status post surgery",
        "status post debridement",
        "status post stsg",
        "status post flap",
        "status post graft",
        "status post amputation",
        "status post radiotherapy",
        "status post rt",
        "negative pressure wound therapy",
        "npwt",
        "chemotherapy",
        "immunotherapy",
    ]
    recurrence_keywords = ["with recurrence", "local recurrence", "recurrent"]
    return any(k in lower for k in major_keywords) or any(k in lower for k in recurrence_keywords)


def condense_diagnosis_block(block: str, keep_major_sub_lines: bool) -> str:
    block = (block or "").strip()
    if not block:
        return ""

    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return ""

    first_line = lines[0]
    if not keep_major_sub_lines:
        return first_line

    kept_sub_lines: list[str] = []
    seen: set[str] = set()
    for line in lines[1:]:
        if is_major_treatment_line(line):
            norm = normalize_dx_line(line)
            if norm not in seen:
                kept_sub_lines.append(line)
                seen.add(norm)

    if len(kept_sub_lines) > 3:
        kept_sub_lines = kept_sub_lines[-3:]

    return "\n".join([first_line] + kept_sub_lines) if kept_sub_lines else first_line


def extract_context_terms(text: str) -> set[str]:
    lowered = (text or "").lower()
    english_terms = set(re.findall(r"[a-z][a-z0-9_-]{2,}", lowered))
    english_terms = {t for t in english_terms if t not in {"with", "status", "post", "history", "chief", "purpose", "notes", "record", "current", "timeline", "additional", "data"}}
    return english_terms


def block_activity_score(block: str, context_text: str) -> int:
    lower_block = (block or "").lower()
    score = 0

    if "recurrence" in lower_block or "recurrent" in lower_block:
        score += 3
    if any(k in lower_block for k in ["tumor", "sarcoma", "infection", "wound", "ulcer", "abscess", "fracture", "bleeding"]):
        score += 1
    if any(is_major_treatment_line(line) for line in (block or "").splitlines()[1:]):
        score += 2

    block_terms = extract_context_terms(block)
    context_terms = extract_context_terms(context_text)
    score += min(3, len(block_terms & context_terms))
    return score


def remove_duplicate_diagnosis_blocks(active_blocks: list[str], underlying_blocks: list[str]) -> tuple[list[str], list[str]]:
    active_keys = {get_block_key(block) for block in active_blocks if block.strip()}
    filtered_underlying = []
    seen_underlying: set[str] = set()

    for block in underlying_blocks:
        key = get_block_key(block)
        if not key or key in active_keys or key in seen_underlying:
            continue
        filtered_underlying.append(block)
        seen_underlying.add(key)

    filtered_active = []
    seen_active: set[str] = set()
    for block in active_blocks:
        key = get_block_key(block)
        if not key or key in seen_active:
            continue
        filtered_active.append(block)
        seen_active.add(key)

    return filtered_active, filtered_underlying


def extract_plan_title_from_dx_line(line: str) -> str:
    stripped = (line or "").strip()
    if not stripped:
        return ""

    prefix = "#."
    body = stripped[2:].strip() if stripped.startswith(prefix) else stripped
    for sep in [", status post", ", s/p", " status post", " s/p", ", with recurrence", " with recurrence"]:
        idx = body.lower().find(sep)
        if idx != -1:
            body = body[:idx].strip(" ,;")
            break
    return f"#. {body}" if body else stripped


def build_plan_active_titles(active_text: str) -> str:
    titles: list[str] = []
    seen: set[str] = set()
    for line in (active_text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("#."):
            continue
        plan_title = extract_plan_title_from_dx_line(stripped)
        norm = normalize_dx_line(plan_title)
        if plan_title and norm not in seen:
            titles.append(plan_title)
            seen.add(norm)
    return "\n".join(titles) if titles else "UNKNOWN"


def extract_relevant_admission_date(*texts: str) -> str:
    patterns = [
        r"\b(20\d{2}/\d{1,2}/\d{1,2})\b",
        r"\b(20\d{2}-\d{1,2}-\d{1,2})\b",
        r"\b(\d{1,2}/\d{1,2})\b",
    ]
    matches: list[str] = []
    for text in texts:
        for pattern in patterns:
            matches.extend(re.findall(pattern, text or ""))
    return matches[-1] if matches else "UNKNOWN DATE"


def build_forced_diagnosis_sections(chief_complaint: str, admission_purpose: str, history_text: str, outpatient_notes: str, emergency_notes: str, timeline_text: str, additional_history: str, extra_data: str) -> tuple[str, str, str, str]:
    blocks = collect_latest_diagnosis_blocks(history_text, outpatient_notes, emergency_notes, additional_history, extra_data)
    if not blocks:
        return "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN DATE"

    context_text = "\n".join([
        chief_complaint or "",
        admission_purpose or "",
        timeline_text or "",
        additional_history or "",
        extra_data or "",
        outpatient_notes or "",
        emergency_notes or "",
    ])

    scored = [(block, block_activity_score(block, context_text)) for block in blocks]
    max_score = max(score for _, score in scored)

    active_blocks: list[str] = []
    underlying_blocks: list[str] = []

    if len(scored) == 1:
        active_blocks = [condense_diagnosis_block(scored[0][0], True)]
    else:
        for block, score in scored:
            if score > 0 and score == max_score:
                active_blocks.append(condense_diagnosis_block(block, True))
            else:
                underlying_blocks.append(condense_diagnosis_block(block, False))

        if not active_blocks:
            active_blocks = [condense_diagnosis_block(scored[-1][0], True)]
            underlying_blocks = [condense_diagnosis_block(block, False) for block, _ in scored[:-1]]

    active_blocks = [b for b in active_blocks if b.strip()]
    underlying_blocks = [b for b in underlying_blocks if b.strip()]

    if not underlying_blocks and len(active_blocks) > 1:
        underlying_blocks = [condense_diagnosis_block(block, False) for block in blocks[:-1]]
        active_blocks = [active_blocks[-1]]

    active_blocks, underlying_blocks = remove_duplicate_diagnosis_blocks(active_blocks, underlying_blocks)

    active_text = "\n".join(active_blocks) if active_blocks else "UNKNOWN"
    underlying_text = "\n".join(underlying_blocks) if underlying_blocks else "UNKNOWN"
    plan_active_titles = build_plan_active_titles(active_text)
    admission_date_text = extract_relevant_admission_date(timeline_text, additional_history, extra_data, emergency_notes, outpatient_notes, history_text)
    return active_text, underlying_text, plan_active_titles, admission_date_text


def case_label(case: dict) -> str:
    status_icon = "🟢" if case.get("status") == "active" else "📦"
    return f"{status_icon} {case.get('mrn', '')} | {case.get('name', '')} | {case.get('age', '')} | {case.get('sex', '')}"


def get_filtered_cases(username: str, status_filter: str = "Active", keyword: str = ""):
    username = require_user(username)
    db = load_db(username)
    status_map = {
        "Active": "active",
        "Discharged": "discharged",
        "All": None,
    }
    wanted = status_map.get(status_filter, "active")
    keyword = (keyword or "").strip().lower()

    filtered = []
    for case in db.get("cases", []):
        if wanted and case.get("status") != wanted:
            continue
        haystack = " ".join([
            str(case.get("mrn", "")),
            str(case.get("name", "")),
            str(case.get("age", "")),
            str(case.get("sex", "")),
        ]).lower()
        if keyword and keyword not in haystack:
            continue
        filtered.append(case)

    filtered.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return filtered


def sidebar_case_choices(username: str, status_filter: str = "Active", keyword: str = ""):
    return [(case_label(c), c["id"]) for c in get_filtered_cases(username, status_filter, keyword)]


def sidebar_case_list_md(username: str, status_filter: str = "Active", keyword: str = "") -> str:
    if not username:
        return "### Cases\n\n_請先登入._"
    cases = get_filtered_cases(username, status_filter, keyword)
    title = f"### {status_filter} Cases"
    if keyword:
        title += f" · search: `{keyword}`"
    if not cases:
        return title + "\n\n_No matching cases._"

    lines = [title, ""]
    for case in cases[:30]:
        status = "Active" if case.get("status") == "active" else "Discharged"
        lines.append(
            f"- **{case.get('mrn','')}** | {case.get('name','')} | {case.get('age','')} | {case.get('sex','')}  \\n"
            f"  {status} · updated {case.get('updated_at','')}"
        )
    if len(cases) > 30:
        lines.append(f"\n_And {len(cases)-30} more cases..._")
    return "\n".join(lines)


def patient_summary_md(case: dict | None) -> str:
    if not case:
        return "### No case selected\n請先從左側病人列表選擇一個 case。"
    status_text = "Active" if case.get("status") == "active" else f"Discharged ({case.get('discharged_at', 'UNKNOWN')})"
    return (
        f"### Current Case\n"
        f"- **病歷號 MRN:** {case.get('mrn', '')}\n"
        f"- **姓名 Name:** {case.get('name', '')}\n"
        f"- **年齡 Age:** {case.get('age', '')}\n"
        f"- **性別 Sex:** {case.get('sex', '')}\n"
        f"- **Status:** {status_text}\n"
        f"- **Created:** {case.get('created_at', '')}\n"
        f"- **Updated:** {case.get('updated_at', '')}"
    )



def require_user(username: str) -> str:
    username = (username or "").strip()
    if not username:
        raise gr.Error("請先登入。")
    return username


def is_admin_user(username: str) -> bool:
    admin_username = (os.environ.get("CLINICAL_ADMIN_USERNAME", "admin") or "").strip()
    return bool(username) and username == admin_username


def authenticate_admin(username: str, password: str) -> bool:
    admin_username = (os.environ.get("CLINICAL_ADMIN_USERNAME", "admin") or "").strip()
    admin_password = os.environ.get("CLINICAL_ADMIN_PASSWORD", "admin123")
    return bool(username) and username == admin_username and (password or "").strip() == admin_password


def current_user_md(username: str) -> str:
    username = (username or "").strip()
    if not username:
        return "### 尚未登入"
    role = "Admin" if is_admin_user(username) else "User"
    return f"### 使用者：{username} · {role}"


def require_case(username: str, case_id: str):
    username = require_user(username)
    if not case_id:
        raise gr.Error("請先從左側選擇一個 case。")
    case = get_case_by_id(username, case_id)
    if not case:
        raise gr.Error("找不到這個 case。")
    return case


def sync_sidebar(username: str, status_filter: str, keyword: str, selected_case_id: str | None = None):
    if not username:
        return (gr.update(choices=[], value=None), "### Cases\n\n_請先登入._", None)
    choices = sidebar_case_choices(username, status_filter, keyword)
    valid_ids = [value for _, value in choices]
    value = selected_case_id if selected_case_id in valid_ids else (valid_ids[0] if valid_ids else None)
    return (
        gr.update(choices=choices, value=value),
        sidebar_case_list_md(username, status_filter, keyword),
        value,
    )


# =========================================================
# Prompts
# =========================================================
STAGE1_PROMPT = """
You are a clinical admission assistant.

Your task is NOT to write the admission note yet.

You must generate TWO outputs:
1. A concise current history timeline
2. A practical pre-admission checklist in MIXED Chinese-English style

GOALS:
1. Organize currently available history into a clean timeline
2. Identify missing critical information
3. Generate practical questions to ask the patient
4. Suggest focused physical examination items

STRICT RULES:
- Do NOT hallucinate
- Only use clinically relevant items
- Be concise, practical, and checklist-oriented
- Use mixed Chinese-English wording
- Output should be easy for a PGY to use during bedside history taking
- Do NOT write long paragraphs
- Timeline must be based ONLY on provided information
- Emergency department notes should be incorporated when provided
- If exact date is unavailable, use approximate wording such as "previously", "recently", "on admission"
- For the checklist, you MUST actively check whether the provided information already includes the following domains:
  past history, family history, surgical history, medication history, allergy history, smoking/alcohol/betel nut history, Chinese herbs or health supplements, TOCC if fever or suspected infection is mentioned
- If any of the above domains are not clearly provided, you MUST remind the user to ask about them in the checklist
- TOCC reminder should appear only when fever, chills, infection, URI symptoms, diarrhea, or similar infectious clues are mentioned
- Do not claim a domain is missing if it was clearly provided in the input

OUTPUT FORMAT EXACTLY:

[Current History Timeline 目前病史時間軸整理]
- item 1
- item 2
- item 3

[Admission Checklist 入院前檢查清單]

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
- Timeline section must use bullet points beginning with -
- Checklist items must each begin with [ ]
- Keep each item short
- Use mixed Chinese-English
- Prioritize high-yield admission questions
- Checklist must explicitly remind the user to補問 missing core history domains when absent
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
- Outpatient notes and timeline are supporting materials and should be integrated when relevant
- The diagnosis wording at the beginning of Present illness is the SINGLE source of truth for the later Tentative Diagnosis and Assessment sections
- Any diagnosis copied later MUST keep EXACT same wording, spelling, punctuation, order, and disease naming
- You may only separate diagnoses into Actives and Underlyings; do NOT rename, merge, split, expand, or shorten them

CRITICAL DIAGNOSIS TEMPLATE RULES
- You will be given TWO blocks named [FORCED ACTIVE DIAGNOSES] and [FORCED UNDERLYING DIAGNOSES].
- You MUST use these blocks exactly as the diagnosis lines shown at the start of Present illness.
- Do NOT show the literal text [Diagnosis Seed] anywhere in the output.
- If the same diagnosis appears in both active and underlying groups, keep it ONLY in Actives and do NOT repeat it in Underlyings.
- For ACTIVE diagnoses, only keep the diagnosis line and MAJOR treatments/procedures if provided.
- Do NOT add minor details, trivial procedures, dressing details, or low-yield historical information into the diagnosis lines.
- For UNDERLYING diagnoses, keep diagnosis lines only unless a major treatment is absolutely essential to identify the disease.
- Do NOT rewrite the diagnosis into prose.
- Later, in Tentative Diagnosis and Assessment, you MUST copy the same diagnosis lines exactly and only separate them into Actives and Underlyings.

CRITICAL PRESENT ILLNESS WRITING RULES
- Present illness should read like a coherent clinical story, not a bullet-point timeline.
- Integrate all available timeline information into one flowing narrative paragraph or short paragraphs.
- More recent events should be described in greater detail.
- More remote events can be summarized briefly.
- Keep the chronology clear and natural.
- Do not overload the diagnosis lines with timeline details; put those details into the narrative.
- In the FINAL paragraph of Present illness, summarize the current admission reason and the MOST RECENT IMPORTANT objective findings for this admission:
  • important labs
  • important imaging
  • important abnormal physical examination findings, if provided
- Only include objective findings that are actually given in the input.
- Prioritize the most recent and clinically important findings; do not list every minor value.
- The final paragraph should therefore explain why this admission is happening now, what the recent key lab/image/PE results show, and then close with the required due-to sentence.
- The ending sentence of Present illness MUST close with this structure:
  "Due to the reason of ... , the patient was admitted on X/X."
- The reason in that sentence must be mainly based on Admission purpose and supported by the provided history.
- Use the provided [FORCED ADMISSION DATE] for X/X when available; if unavailable, write UNKNOWN DATE.

========================
FORMAT REQUIREMENTS
========================

一. 主訴(Chief Complaint)
→ Use given chief complaint

二. 病史(Brief History)

【Present illness】
- MUST start with:
"This is a XX-year-old man/woman with the following underlying diseases:"
- Immediately after the above sentence, list the diagnoses in TWO groups using the exact format below:

[Actives]
# diagnosis 1
# diagnosis 2

[Underlyings]
# diagnosis 1
# diagnosis 2

- The diagnosis lines above are the source of truth
- The same diagnosis lines MUST be copied verbatim later into Tentative Diagnosis and Assessment
- The diagnosis lines MUST be based on [FORCED ACTIVE DIAGNOSES] and [FORCED UNDERLYING DIAGNOSES]
- After the diagnosis lines, write Present illness as a chronological clinical story
- Recent events should be more detailed; remote history can be brief
- Treatment history in the narrative should use natural clinical prose, and may mention status post major procedures when relevant
- In the LAST paragraph of Present illness, summarize the current admission purpose/reason and the most recent important lab, imaging, and abnormal PE findings if available
- This last paragraph should focus on why the patient is admitted this time and what the recent workup shows
- End the Present illness with: "Due to the reason of ... , the patient was admitted on X/X."
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

- MUST copy the diagnosis lines from the BEGINNING of Present illness exactly
- NO wording changes
- NO new diagnosis
- NO deleted diagnosis
- ONLY separate into the same two groups below

【Actives】
# diagnosis 1
# diagnosis 2

【Underlyings】
# diagnosis 1
# diagnosis 2

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
- MUST be IDENTICAL to Tentative Diagnosis
- Copy the same diagnosis lines exactly with the same grouping
- NO wording changes

【Plan】
→ ONLY for Actives
→ Use ONLY the diagnosis names from [PLAN ACTIVE DIAGNOSES] as plan headers
→ DO NOT include status post details, procedure history, recurrence sub-lines, or treatment suffixes in the plan headers
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


HANDOFF_PROMPT = """
You are a clinical assistant generating a concise handoff summary for an inpatient case.

STRICT RULES:
- Do NOT hallucinate
- Only use provided information
- Be clinically practical and concise
- Use formal clinical English
- Prioritize current inpatient issues
- If procedure-related information is absent, write "None noted" for procedure summary
- Do not invent vitals, labs, imaging, or events

TASK:
You will be given THREE input sections:
1. Problem
2. Assessment
3. Plan

Use them together to generate the following sections exactly:

[One-liner]
- One sentence summary including age/sex when available, major active disease(s), current status, and admission context if available
- Mainly synthesize the most important Problem + Assessment

[Active Issues]
- Bullet points of current active problems needing management
- Mainly based on Problem and supported by Assessment

[Overnight Concerns]
- Bullet points of what the covering team should watch overnight
- Mainly infer from Assessment
- Include possible complications, monitoring focus, red flags, pain/wound/bleeding/fever/respiratory/hemodynamic concerns if relevant

[To-do List]
- Bullet points of actionable pending tasks
- Mainly based on Plan
- Include follow-up labs, image review, consult follow-up, wound care, medication adjustments, discharge planning, etc. when supported

[Post-op Note / Procedure Summary]
- Short bullets summarizing important recent operation(s), bedside procedures, drains/wounds, major postoperative status, if procedure-related information is present in Problem / Assessment / Plan
- Otherwise write "None noted"

STYLE:
- Keep each bullet short and practical
- Focus on cross-cover usefulness
- Output in clean structured format
"""

OR_PROMPT = """
You are a surgical assistant helping a PGY prepare for upcoming operations.

GOAL:
Generate a practical OR briefing and PGY knowledge bank in MIXED Chinese-English style.

STRICT RULES:
- Do NOT hallucinate uncommon details
- Focus on high-yield, commonly tested knowledge
- Use guideline-based indications when possible
- Be concise but clinically meaningful
- Suitable for PGY level
- Prefer practical points that are commonly asked during OR, ward rounds, and post-op discussions
- If exact anatomy or complication details are not clearly procedure-specific, provide the most relevant commonly tested essentials only

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

9. Common Pimp Questions（常見被問題庫）
- 5–10 short high-yield pimp questions
- Cover indication, anatomy, steps, complication, postoperative care
- Format:
    Q: ...
    A: ...

10. Anatomy Quick Review（解剖快整理）
- Brief bullet review of the most relevant anatomy for this surgery
- Include:
    • key layers / spaces
    • important vessels
    • important nerves
    • structures at risk
- Keep it short and clinically testable

11. Complication Quick Review（併發症快整理）
- Short bullets of common and dangerous complications
- Include:
    • intra-op complications
    • early post-op complications
    • late complications if high-yield
    • what to monitor or how to recognize them

STYLE:
- Mixed Chinese + English
- Bullet points
- Clear and structured
- High-yield only
"""


# =========================================================
# Case management
# =========================================================
def create_case(username, mrn, name, age, sex, status_filter, keyword):
    mrn = (mrn or "").strip()
    name = (name or "").strip()
    age = (age or "").strip()
    sex = (sex or "").strip()

    if not mrn or not name or not age or not sex:
        raise gr.Error("請完整填寫 病歷號、姓名、年齡、性別。")

    username = require_user(username)
    db = load_db(username)
    for case in db.get("cases", []):
        if case.get("mrn") == mrn and case.get("status") == "active":
            raise gr.Error("已有相同病歷號的 active case。")

    case = default_case_record(mrn, name, age, sex)
    db["cases"].append(case)
    save_db(username, db)

    sidebar_dropdown, sidebar_md, selected_case_id = sync_sidebar(username, status_filter, keyword, case["id"])
    return (
        sidebar_dropdown,
        sidebar_md,
        selected_case_id,
        patient_summary_md(case),
        *empty_case_form(),
        f"已建立新 case：{mrn} / {name}",
    )


def load_selected_case(username, case_id):
    case = require_case(username, case_id)
    s1 = case["admission"]["stage1"]
    s2 = case["admission"]["stage2"]
    weekly = case["weekly"]
    discharge = case["discharge"]
    orb = case["or_briefing"]
    handoff = case.get("handoff", {"problem": "", "assessment": "", "plan": "", "output": ""})

    auto_discharge_weekly = discharge["weekly"] or weekly["output"]
    auto_or_history = orb["history"] or s2["history_text"] or s1["history_text"]

    return (
        case["id"],
        patient_summary_md(case),
        s1["chief_complaint"], s1["admission_purpose"], s1["history_text"], s1.get("outpatient_notes", ""), s1.get("emergency_notes", ""), s1["labs"], s1.get("timeline_output", ""), s1.get("checklist_output", s1.get("output", "")),
        s2["chief_complaint"], s2["admission_purpose"], s2["history_text"], s2.get("outpatient_notes", ""), s2.get("emergency_notes", ""), s2.get("timeline_text", ""), s2["labs"], s2["additional_history"], s2["pe_findings"], s2["extra_data"], s2["output"],
        weekly["events"], weekly["previous_weekly"], weekly["output"],
        auto_discharge_weekly, discharge["final_events"], discharge["output"],
        auto_or_history, orb["meds"], orb["surgery"], orb["extra"], orb["output"],
        handoff.get("problem", ""), handoff.get("assessment", ""), handoff.get("plan", ""), handoff.get("output", ""),
        f"已載入 case：{case['mrn']} / {case['name']}",
    )


def move_case_to_archive(username, case_id, status_filter, keyword):
    case = require_case(username, case_id)
    if case.get("status") == "discharged":
        sidebar_dropdown, sidebar_md, selected_case_id = sync_sidebar(username, status_filter, keyword, case_id)
        return (
            sidebar_dropdown,
            sidebar_md,
            selected_case_id,
            patient_summary_md(case),
            "此 case 已經在已出院區。",
        )

    updated = update_case(username, case_id, lambda c: c.update({"status": "discharged", "discharged_at": now_iso()}))
    sidebar_dropdown, sidebar_md, selected_case_id = sync_sidebar(username, status_filter, keyword, updated["id"])
    return (
        sidebar_dropdown,
        sidebar_md,
        selected_case_id,
        patient_summary_md(updated),
        f"已將 case 移到已出院區：{updated['mrn']} / {updated['name']}",
    )


def restore_case(username, case_id, status_filter, keyword):
    case = require_case(username, case_id)
    updated = update_case(username, case_id, lambda c: c.update({"status": "active", "discharged_at": ""}))
    sidebar_dropdown, sidebar_md, selected_case_id = sync_sidebar(username, status_filter, keyword, updated["id"])
    return (
        sidebar_dropdown,
        sidebar_md,
        selected_case_id,
        patient_summary_md(updated),
        f"已將 case 恢復到 active 區：{updated['mrn']} / {updated['name']}",
    )


def delete_case(username, case_id, status_filter, keyword):
    case = require_case(username, case_id)
    username = require_user(username)
    db = load_db(username)
    db["cases"] = [c for c in db.get("cases", []) if c.get("id") != case_id]
    save_db(username, db)


    sidebar_dropdown, sidebar_md, selected_case_id = sync_sidebar(username, status_filter, keyword, None)
    return (
        sidebar_dropdown,
        sidebar_md,
        selected_case_id,
        "### No case selected\n請先從左側病人列表選擇一個 case。",
        *empty_stage1(),
        *empty_stage2(),
        *empty_weekly(),
        *empty_discharge(),
        *empty_or(),
        *empty_handoff(),
        f"已移除 case：{case['mrn']} / {case['name']}",
    )



def save_workspace(
    username,
    case_id,
    chief1, purpose1, history1, outpatient1, emergency1, labs1, timeline1, checklist1,
    chief2, purpose2, history2, outpatient2, emergency2, timeline2, labs2, add_hist, pe2, extra2, output2,
    weekly_events, weekly_prev, weekly_out,
    discharge_weekly, discharge_events, discharge_out,
    or_history, or_meds, or_surgery, or_extra, or_out,
    handoff_problem, handoff_assessment, handoff_plan, handoff_out,
):
    case = require_case(username, case_id)

    updated = update_case(
        username,
        case_id,
        lambda c: (
            c["admission"]["stage1"].update({
                "chief_complaint": chief1,
                "admission_purpose": purpose1,
                "history_text": history1,
                "outpatient_notes": outpatient1,
                "emergency_notes": emergency1,
                "labs": labs1,
                "timeline_output": timeline1,
                "checklist_output": checklist1,
                "output": (
                    "[Current History Timeline 目前病史時間軸整理]\n" + (timeline1 or "") + "\n\n"
                    "[Admission Checklist 入院前檢查清單]\n" + (checklist1 or "")
                ).strip(),
            }),
            c["admission"]["stage2"].update({
                "chief_complaint": chief2,
                "admission_purpose": purpose2,
                "history_text": history2,
                "outpatient_notes": outpatient2,
                "emergency_notes": emergency2,
                "timeline_text": timeline2,
                "labs": labs2,
                "additional_history": add_hist,
                "pe_findings": pe2,
                "extra_data": extra2,
                "output": output2,
            }),
            c["weekly"].update({
                "events": weekly_events,
                "previous_weekly": weekly_prev,
                "output": weekly_out,
            }),
            c["discharge"].update({
                "weekly": discharge_weekly,
                "final_events": discharge_events,
                "output": discharge_out,
            }),
            c["or_briefing"].update({
                "history": or_history,
                "meds": or_meds,
                "surgery": or_surgery,
                "extra": or_extra,
                "output": or_out,
            }),
            c.setdefault("handoff", {}).update({
                "problem": handoff_problem,
                "assessment": handoff_assessment,
                "plan": handoff_plan,
                "output": handoff_out,
            }),
        ),
    )

    return patient_summary_md(updated), f"Workspace saved: {updated['mrn']} / {updated['name']}"

def refresh_sidebar_ui(username, status_filter, keyword, selected_case_id):
    return sync_sidebar(username, status_filter, keyword, selected_case_id)


# =========================================================
# Core workflow functions
# =========================================================
def admission_stage1(username, case_id, chief_complaint, admission_purpose, history_text, outpatient_notes, emergency_notes, labs):
    case = require_case(username, case_id)
    user_input = f"""
Chief complaint: {chief_complaint}

Admission purpose: {admission_purpose}

Known history:
{history_text}

Outpatient notes / OPD record:
{outpatient_notes}

Emergency department notes / ER record:
{emergency_notes}

Labs / imaging:
{labs}
"""
    result = ask_model(STAGE1_PROMPT, user_input)
    timeline_output, checklist_output = split_stage1_output(result)
    combined_output = (
        "[Current History Timeline 目前病史時間軸整理]\n" + (timeline_output or "") + "\n\n"
        "[Admission Checklist 入院前檢查清單]\n" + (checklist_output or "")
    ).strip()

    updated = update_case(
        username,
        case_id,
        lambda c: c["admission"]["stage1"].update(
            {
                "chief_complaint": chief_complaint,
                "admission_purpose": admission_purpose,
                "history_text": history_text,
                "outpatient_notes": outpatient_notes,
                "emergency_notes": emergency_notes,
                "labs": labs,
                "timeline_output": timeline_output,
                "checklist_output": checklist_output,
                "output": combined_output,
            }
        ),
    )
    return timeline_output, checklist_output, patient_summary_md(updated)


def admission_stage2(username, case_id, chief_complaint, admission_purpose, history_text, outpatient_notes, emergency_notes, timeline_text, labs, additional_history, pe_findings, extra_data):
    case = require_case(username, case_id)

    forced_active_dx, forced_underlying_dx, plan_active_dx, forced_admission_date = build_forced_diagnosis_sections(
        chief_complaint,
        admission_purpose,
        history_text,
        outpatient_notes,
        emergency_notes,
        timeline_text,
        additional_history,
        extra_data,
    )

    user_input = f"""
Chief complaint: {chief_complaint}

Admission purpose: {admission_purpose}

Known history:
{history_text}

Outpatient notes / OPD record:
{outpatient_notes}

Emergency department notes / ER record:
{emergency_notes}

Current history timeline:
{timeline_text}

Labs / imaging:
{labs}

Additional history obtained:
{additional_history}

Physical examination findings:
{pe_findings}

Additional data:
{extra_data}

[FORCED ACTIVE DIAGNOSES]
{forced_active_dx if forced_active_dx else "UNKNOWN"}

[FORCED UNDERLYING DIAGNOSES]
{forced_underlying_dx if forced_underlying_dx else "UNKNOWN"}

[PLAN ACTIVE DIAGNOSES]
{plan_active_dx if plan_active_dx else "UNKNOWN"}

[FORCED ADMISSION DATE]
{forced_admission_date if forced_admission_date else "UNKNOWN DATE"}
"""
    result = ask_model(STAGE2_PROMPT, user_input)

    updated = update_case(
        username,
        case_id,
        lambda c: c["admission"]["stage2"].update(
            {
                "chief_complaint": chief_complaint,
                "admission_purpose": admission_purpose,
                "history_text": history_text,
                "outpatient_notes": outpatient_notes,
                "emergency_notes": emergency_notes,
                "timeline_text": timeline_text,
                "labs": labs,
                "additional_history": additional_history,
                "pe_findings": pe_findings,
                "extra_data": extra_data,
                "output": result,
            }
        ),
    )
    auto_or_history = history_text or updated["admission"]["stage1"].get("history_text", "")
    return result, patient_summary_md(updated), auto_or_history



def handoff_summary(username, case_id, problem, assessment, plan):
    case = require_case(username, case_id)
    user_input = f"""
Problem:
{problem}

Assessment:
{assessment}

Plan:
{plan}
"""
    result = ask_model(HANDOFF_PROMPT, user_input)

    updated = update_case(
        username,
        case_id,
        lambda c: c.setdefault("handoff", {}).update(
            {
                "problem": problem,
                "assessment": assessment,
                "plan": plan,
                "output": result,
            }
        ),
    )
    return result, patient_summary_md(updated)


def weekly_summary(username, case_id, events, previous_weekly):
    case = require_case(username, case_id)
    user_input = f"""
Weekly events and timeline:
{events}

Previous weekly summary (if any):
{previous_weekly}
"""
    result = ask_model(WEEKLY_PROMPT, user_input)

    updated = update_case(
        username,
        case_id,
        lambda c: c["weekly"].update(
            {
                "events": events,
                "previous_weekly": previous_weekly,
                "output": result,
            }
        ),
    )
    return result, patient_summary_md(updated), result


def discharge_note(username, case_id, weekly, final_events):
    case = require_case(username, case_id)
    user_input = f"""
Weekly summaries:
{weekly}

Final week events and timeline:
{final_events}
"""
    result = ask_model(DISCHARGE_PROMPT, user_input)

    updated = update_case(
        username,
        case_id,
        lambda c: c["discharge"].update(
            {
                "weekly": weekly,
                "final_events": final_events,
                "output": result,
            }
        ),
    )
    return result, patient_summary_md(updated)


def or_briefing(username, case_id, history, meds, surgery, extra):
    case = require_case(username, case_id)
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

    updated = update_case(
        username,
        case_id,
        lambda c: c["or_briefing"].update(
            {
                "history": history,
                "meds": meds,
                "surgery": surgery,
                "extra": extra,
                "output": result,
            }
        ),
    )
    return result, patient_summary_md(updated)


# =========================================================
# Cross-fill helpers
# =========================================================
def copy_stage1_to_stage2(chief, purpose, history, outpatient_notes, emergency_notes, timeline_text, labs):
    return chief, purpose, history, outpatient_notes, emergency_notes, timeline_text, labs


def copy_weekly_to_discharge(weekly_output):
    return weekly_output


def copy_history_to_or(history_text):
    return history_text


def autofill_discharge_from_weekly(weekly_output, current_discharge_weekly):
    return current_discharge_weekly or weekly_output


def autofill_or_history(stage2_history_text, stage1_history_text, current_or_history):
    return current_or_history or stage2_history_text or stage1_history_text




# =========================================================
# Session / UI helpers
# =========================================================
def login_user(username: str, password: str):
    username = (username or "").strip()
    password = (password or "").strip()
    ok = authenticate_admin(username, password) or authenticate_user(username, password)
    if not ok:
        return (
            "",
            current_user_md(""),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            "### Cases\n\n_請先登入._",
            None,
            "登入失敗，請確認 username / password。",
            username,
            "",
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            "### User Cases\n\n_尚未選擇使用者。_",
            "",
            "",
            "",
            gr.update(choices=[], value=None),
            "### Case Preview\n\n_尚未選擇 case。_",
        )
    sidebar_dropdown, sidebar_md, selected_case_id = sync_sidebar(username, "Active", "", None)
    admin_visible = is_admin_user(username)
    admin_choices = admin_user_choices() if admin_visible else []
    return (
        username,
        current_user_md(username),
        gr.update(visible=False),
        gr.update(visible=True),
        sidebar_dropdown,
        sidebar_md,
        selected_case_id,
        f"登入成功：{username}",
        username,
        "",
        gr.update(visible=admin_visible),
        gr.update(choices=admin_choices, value=None),
        "### User Cases\n\n_尚未選擇使用者。_",
        "",
        "",
        "",
        gr.update(choices=[], value=None),
        "### Case Preview\n\n_尚未選擇 case。_",
    )


def register_user_ui(username: str, password: str, confirm_password: str):
    try:
        message = register_user(username, password, confirm_password)
        return message, username, "", ""
    except Exception as e:
        return f"註冊失敗：{e}", username, "", ""


def logout_user():
    return (
        "",
        current_user_md(""),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(choices=[], value=None),
        "### Cases\n\n_請先登入._",
        None,
        "已登出。",
        "### No case selected\n請先登入。",
        *empty_stage1(),
        *empty_stage2(),
        *empty_weekly(),
        *empty_discharge(),
        *empty_or(),
        *empty_handoff(),
        gr.update(visible=False),
        gr.update(choices=[], value=None),
        "### User Cases\n\n_尚未選擇使用者。_",
        "",
        "",
        "",
        gr.update(choices=[], value=None),
        "### Case Preview\n\n_尚未選擇 case。_",
    )


with gr.Blocks(title="Clinical AI Workspace", theme=gr.themes.Soft()) as demo:
    user_state = gr.State("")
    selected_case_id = gr.State("")

    current_user_banner = gr.Markdown("### 尚未登入")
    login_status = gr.Markdown("")
    with gr.Column(visible=False) as admin_panel:
        gr.Markdown("## Admin 後台 v3")
        admin_user_search = gr.Textbox(label="搜尋帳號", placeholder="輸入 username")
        admin_user_selector = gr.Dropdown(label="選擇使用者", choices=[], value=None)
        with gr.Row():
            admin_load_btn = gr.Button("載入使用者資料")
            admin_delete_btn = gr.Button("刪除已註冊帳號", variant="stop")
        admin_user_summary = gr.Markdown("### User Cases\n\n_尚未選擇使用者。_")
        with gr.Row():
            admin_case_search = gr.Textbox(label="搜尋 case", placeholder="輸入 MRN / 姓名 / 年齡 / 性別 / status")
            admin_case_refresh_btn = gr.Button("刷新 Case 清單")
        admin_case_selector = gr.Dropdown(label="Case 清單預覽", choices=[], value=None)
        admin_case_preview = gr.Markdown("### Case Preview\n\n_尚未選擇 case。_")
        gr.Markdown("可搜尋帳號、載入指定使用者資料，再搜尋與預覽該使用者的 case。")
        admin_user_json = gr.Code(label="使用者帳號資料", language="json")
        admin_cases_json = gr.Code(label="使用者所有輸入 / 輸出資料", language="json")
        admin_status = gr.Markdown("")


    with gr.Column(visible=True) as login_panel:
        gr.Markdown("# Clinical AI Workspace")
        with gr.Row():
            login_username = gr.Textbox(label="Username")
            login_password = gr.Textbox(label="Password", type="password")
        with gr.Row():
            login_btn = gr.Button("Login", variant="primary")
        with gr.Accordion("註冊 / Register", open=False):
            register_username = gr.Textbox(label="New Username")
            register_password = gr.Textbox(label="New Password", type="password")
            register_password_confirm = gr.Textbox(label="Confirm Password", type="password")
            register_btn = gr.Button("Register")
            register_status = gr.Markdown("")

    with gr.Column(visible=False) as app_panel:
        gr.Markdown("# Clinical AI Workspace")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=340):
                status_filter = gr.Radio(
                    choices=["Active", "Discharged", "All"],
                    value="Active",
                    label="Case Filter",
                )
                case_search = gr.Textbox(label="Search Case", placeholder="輸入病歷號 / 姓名 / 年齡 / 性別")
                refresh_sidebar_btn = gr.Button("Refresh List")
                case_selector = gr.Radio(
                    choices=[],
                    value=None,
                    label="病人列表（病歷號｜姓名｜年齡｜性別）",
                    interactive=True,
                )
                load_case_btn = gr.Button("Load Selected Case", variant="primary")
                logout_btn = gr.Button("Logout")
                case_list_preview = gr.Markdown("### Cases\n\n_請先登入._")

                with gr.Accordion("新增病人 / Create Case", open=False):
                    new_mrn = gr.Textbox(label="病歷號 MRN")
                    new_name = gr.Textbox(label="姓名 Name")
                    new_age = gr.Textbox(label="年齡 Age")
                    new_sex = gr.Dropdown(choices=["M", "F", "Other"], label="性別 Sex")
                    create_case_btn = gr.Button("Create Case")

            with gr.Column(scale=3, min_width=900):
                case_summary = gr.Markdown("### No case selected\n請先從左側病人列表選擇一個 case。")
                with gr.Row():
                    discharge_case_btn = gr.Button("Move to Discharged Archive")
                    restore_case_btn = gr.Button("Restore to Active")
                    delete_case_btn = gr.Button("Delete Case", variant="stop")
                with gr.Row():
                    save_workspace_btn = gr.Button("💾 Save Workspace")
                    reload_workspace_btn = gr.Button("🔄 Reload Workspace")
                case_status_message = gr.Markdown("")

                with gr.Tabs():
                    with gr.Tab("Admission Copilot"):
                        gr.Markdown("## Stage 1: Interview Guide")
                        with gr.Row():
                            chief1 = gr.Textbox(label="Chief Complaint")
                            purpose1 = gr.Textbox(label="Admission Purpose")
                        history1 = gr.Textbox(label="Known History / Previous Notes (中英皆可)", lines=10)
                        outpatient1 = gr.Textbox(label="Outpatient Notes / OPD Record（門診紀錄）", lines=6)
                        emergency1 = gr.Textbox(label="Emergency Notes / ER Record（急診紀錄）", lines=6)
                        labs1 = gr.Textbox(label="Labs / Imaging (brief)", lines=5)

                        with gr.Row():
                            btn1 = gr.Button("Generate Interview Guide")
                            btn1_clear = gr.Button("Clear")

                        with gr.Row():
                            timeline1 = gr.Textbox(label="Current History Timeline", lines=18)
                            output1 = gr.Textbox(label="Admission Checklist", lines=18)

                        gr.Markdown("## Stage 2: Final Admission Note")
                        with gr.Row():
                            chief2 = gr.Textbox(label="Chief Complaint")
                            purpose2 = gr.Textbox(label="Admission Purpose")
                        history2 = gr.Textbox(label="Known History / Previous Notes", lines=10)
                        outpatient2 = gr.Textbox(label="Outpatient Notes / OPD Record（門診紀錄）", lines=6)
                        emergency2 = gr.Textbox(label="Emergency Notes / ER Record（急診紀錄）", lines=6)
                        timeline2 = gr.Textbox(label="Current History Timeline", lines=6)
                        labs2 = gr.Textbox(label="Labs / Imaging", lines=5)
                        add_hist = gr.Textbox(label="Additional History Obtained", lines=5)
                        pe2 = gr.Textbox(label="Physical Examination Findings", lines=5)
                        extra2 = gr.Textbox(label="Additional Data", lines=5)

                        with gr.Row():
                            btn_copy_to_stage2 = gr.Button("Copy Stage 1 Inputs to Stage 2")
                            btn2 = gr.Button("Generate Final Admission Note")
                            btn2_clear = gr.Button("Clear")

                        output2 = gr.Textbox(label="Final Admission Note", lines=30)

                    with gr.Tab("交班摘要"):
                        handoff_problem = gr.Textbox(label="Problem", lines=6)
                        handoff_assessment = gr.Textbox(label="Assessment", lines=6)
                        handoff_plan = gr.Textbox(label="Plan", lines=6)

                        with gr.Row():
                            handoff_btn = gr.Button("Generate Handoff Summary")
                            handoff_clear = gr.Button("Clear")

                        handoff_out = gr.Textbox(label="交班摘要 Handoff Summary", lines=24)

                    with gr.Tab("Weekly Summary"):
                        weekly_events = gr.Textbox(label="This week's events & timeline", lines=15)
                        weekly_prev = gr.Textbox(label="Previous weekly summary (optional)", lines=10)

                        with gr.Row():
                            weekly_btn = gr.Button("Generate Weekly Summary")
                            weekly_clear = gr.Button("Clear")

                        weekly_out = gr.Textbox(label="Weekly Summary", lines=25)

                    with gr.Tab("Discharge Note"):
                        discharge_weekly = gr.Textbox(label="All weekly summaries (optional)", lines=15)
                        discharge_events = gr.Textbox(label="Final week events & timeline", lines=10)

                        with gr.Row():
                            discharge_copy = gr.Button("Copy Weekly Output Here")
                            discharge_autofill = gr.Button("Auto-fill from Weekly")
                            discharge_btn = gr.Button("Generate Course and Treatment")
                            discharge_clear = gr.Button("Clear")

                        discharge_out = gr.Textbox(label="Course and Treatment", lines=20)

                    with gr.Tab("OR Briefing"):
                        or_history = gr.Textbox(label="Brief History（病史）", lines=6)
                        or_meds = gr.Textbox(label="Current Medications（用藥）", lines=5)
                        or_surgery = gr.Textbox(label="Planned Surgery（手術名稱）", lines=3)
                        or_extra = gr.Textbox(label="Additional Info（labs / image 可選）", lines=5)

                        with gr.Row():
                            or_copy_history = gr.Button("Copy Admission History Here")
                            or_autofill = gr.Button("Auto-fill Admission History")
                            or_btn = gr.Button("Generate OR Briefing")
                            or_clear = gr.Button("Clear")

                        or_out = gr.Textbox(label="OR Briefing", lines=30)

    # auth
    register_btn.click(
        register_user_ui,
        inputs=[register_username, register_password, register_password_confirm],
        outputs=[register_status, login_username, register_password, register_password_confirm],
    )
    login_btn.click(
        login_user,
        inputs=[login_username, login_password],
        outputs=[user_state, current_user_banner, login_panel, app_panel, case_selector, case_list_preview, selected_case_id, login_status, login_username, login_password, admin_panel, admin_user_selector, admin_user_summary, admin_user_json, admin_cases_json, admin_status, admin_case_selector, admin_case_preview],
    )
    login_password.submit(
        login_user,
        inputs=[login_username, login_password],
        outputs=[user_state, current_user_banner, login_panel, app_panel, case_selector, case_list_preview, selected_case_id, login_status, login_username, login_password, admin_panel, admin_user_selector, admin_user_summary, admin_user_json, admin_cases_json, admin_status, admin_case_selector, admin_case_preview],
    )
    login_username.submit(
        login_user,
        inputs=[login_username, login_password],
        outputs=[user_state, current_user_banner, login_panel, app_panel, case_selector, case_list_preview, selected_case_id, login_status, login_username, login_password, admin_panel, admin_user_selector, admin_user_summary, admin_user_json, admin_cases_json, admin_status, admin_case_selector, admin_case_preview],
    )
    logout_btn.click(
        logout_user,
        outputs=[
            user_state, current_user_banner, login_panel, app_panel, case_selector, case_list_preview,
            selected_case_id, login_status, case_summary,
            chief1, purpose1, history1, outpatient1, emergency1, labs1, timeline1, output1,
            chief2, purpose2, history2, outpatient2, emergency2, timeline2, labs2, add_hist, pe2, extra2, output2,
            weekly_events, weekly_prev, weekly_out,
            discharge_weekly, discharge_events, discharge_out,
            or_history, or_meds, or_surgery, or_extra, or_out,
            handoff_problem, handoff_assessment, handoff_plan, handoff_out,
            admin_panel, admin_user_selector, admin_user_summary, admin_user_json, admin_cases_json, admin_status, admin_case_selector, admin_case_preview,
        ],
    )

    admin_load_btn.click(
        load_admin_user_data,
        inputs=[user_state, admin_user_selector],
        outputs=[admin_user_summary, admin_user_json, admin_cases_json, admin_status, admin_case_selector, admin_case_preview],
    )
    admin_delete_btn.click(
        delete_registered_user,
        inputs=[user_state, admin_user_selector],
        outputs=[admin_user_selector, admin_user_summary, admin_user_json, admin_cases_json, admin_status, admin_case_selector, admin_case_preview],
    )

    # Sidebar refresh / filter
    status_filter.change(
        refresh_sidebar_ui,
        inputs=[user_state, status_filter, case_search, selected_case_id],
        outputs=[case_selector, case_list_preview, selected_case_id],
    )
    case_search.submit(
        refresh_sidebar_ui,
        inputs=[user_state, status_filter, case_search, selected_case_id],
        outputs=[case_selector, case_list_preview, selected_case_id],
    )
    refresh_sidebar_btn.click(
        refresh_sidebar_ui,
        inputs=[user_state, status_filter, case_search, selected_case_id],
        outputs=[case_selector, case_list_preview, selected_case_id],
    )

    create_case_btn.click(
        create_case,
        inputs=[user_state, new_mrn, new_name, new_age, new_sex, status_filter, case_search],
        outputs=[case_selector, case_list_preview, selected_case_id, case_summary, new_mrn, new_name, new_age, new_sex, case_status_message],
    )

    load_case_btn.click(
        load_selected_case,
        inputs=[user_state, case_selector],
        outputs=[
            selected_case_id, case_summary,
            chief1, purpose1, history1, outpatient1, emergency1, labs1, timeline1, output1,
            chief2, purpose2, history2, outpatient2, emergency2, timeline2, labs2, add_hist, pe2, extra2, output2,
            weekly_events, weekly_prev, weekly_out,
            discharge_weekly, discharge_events, discharge_out,
            or_history, or_meds, or_surgery, or_extra, or_out,
            handoff_problem, handoff_assessment, handoff_plan, handoff_out,
            case_status_message,
        ],
    )
    case_selector.change(
        load_selected_case,
        inputs=[user_state, case_selector],
        outputs=[
            selected_case_id, case_summary,
            chief1, purpose1, history1, outpatient1, emergency1, labs1, timeline1, output1,
            chief2, purpose2, history2, outpatient2, emergency2, timeline2, labs2, add_hist, pe2, extra2, output2,
            weekly_events, weekly_prev, weekly_out,
            discharge_weekly, discharge_events, discharge_out,
            or_history, or_meds, or_surgery, or_extra, or_out,
            handoff_problem, handoff_assessment, handoff_plan, handoff_out,
            case_status_message,
        ],
    )

    save_workspace_btn.click(
        save_workspace,
        inputs=[
            user_state, selected_case_id,
            chief1, purpose1, history1, outpatient1, emergency1, labs1, timeline1, output1,
            chief2, purpose2, history2, outpatient2, emergency2, timeline2, labs2, add_hist, pe2, extra2, output2,
            weekly_events, weekly_prev, weekly_out,
            discharge_weekly, discharge_events, discharge_out,
            or_history, or_meds, or_surgery, or_extra, or_out,
            handoff_problem, handoff_assessment, handoff_plan, handoff_out,
        ],
        outputs=[case_summary, case_status_message],
    )
    reload_workspace_btn.click(
        load_selected_case,
        inputs=[user_state, selected_case_id],
        outputs=[
            selected_case_id, case_summary,
            chief1, purpose1, history1, outpatient1, emergency1, labs1, timeline1, output1,
            chief2, purpose2, history2, outpatient2, emergency2, timeline2, labs2, add_hist, pe2, extra2, output2,
            weekly_events, weekly_prev, weekly_out,
            discharge_weekly, discharge_events, discharge_out,
            or_history, or_meds, or_surgery, or_extra, or_out,
            handoff_problem, handoff_assessment, handoff_plan, handoff_out,
            case_status_message,
        ],
    )

    discharge_case_btn.click(
        move_case_to_archive,
        inputs=[user_state, selected_case_id, status_filter, case_search],
        outputs=[case_selector, case_list_preview, selected_case_id, case_summary, case_status_message],
    )
    restore_case_btn.click(
        restore_case,
        inputs=[user_state, selected_case_id, status_filter, case_search],
        outputs=[case_selector, case_list_preview, selected_case_id, case_summary, case_status_message],
    )
    delete_case_btn.click(
        delete_case,
        inputs=[user_state, selected_case_id, status_filter, case_search],
        outputs=[
            case_selector, case_list_preview, selected_case_id, case_summary,
            chief1, purpose1, history1, outpatient1, emergency1, labs1, timeline1, output1,
            chief2, purpose2, history2, outpatient2, emergency2, timeline2, labs2, add_hist, pe2, extra2, output2,
            weekly_events, weekly_prev, weekly_out,
            discharge_weekly, discharge_events, discharge_out,
            or_history, or_meds, or_surgery, or_extra, or_out,
            handoff_problem, handoff_assessment, handoff_plan, handoff_out,
            case_status_message,
        ],
    )

    btn1.click(
        admission_stage1,
        inputs=[user_state, selected_case_id, chief1, purpose1, history1, outpatient1, emergency1, labs1],
        outputs=[timeline1, output1, case_summary],
    )
    btn1_clear.click(lambda: empty_stage1(), outputs=[chief1, purpose1, history1, outpatient1, emergency1, labs1, timeline1, output1])

    btn_copy_to_stage2.click(
        copy_stage1_to_stage2,
        inputs=[chief1, purpose1, history1, outpatient1, emergency1, timeline1, labs1],
        outputs=[chief2, purpose2, history2, outpatient2, emergency2, timeline2, labs2],
    )
    btn2.click(
        admission_stage2,
        inputs=[user_state, selected_case_id, chief2, purpose2, history2, outpatient2, emergency2, timeline2, labs2, add_hist, pe2, extra2],
        outputs=[output2, case_summary, or_history],
    )
    btn2_clear.click(lambda: empty_stage2(), outputs=[chief2, purpose2, history2, outpatient2, emergency2, timeline2, labs2, add_hist, pe2, extra2, output2])

    weekly_btn.click(
        weekly_summary,
        inputs=[user_state, selected_case_id, weekly_events, weekly_prev],
        outputs=[weekly_out, case_summary, discharge_weekly],
    )
    weekly_clear.click(lambda: empty_weekly(), outputs=[weekly_events, weekly_prev, weekly_out])

    discharge_copy.click(copy_weekly_to_discharge, inputs=[weekly_out], outputs=[discharge_weekly])
    discharge_autofill.click(
        autofill_discharge_from_weekly,
        inputs=[weekly_out, discharge_weekly],
        outputs=[discharge_weekly],
    )
    discharge_btn.click(
        discharge_note,
        inputs=[user_state, selected_case_id, discharge_weekly, discharge_events],
        outputs=[discharge_out, case_summary],
    )
    discharge_clear.click(lambda: empty_discharge(), outputs=[discharge_weekly, discharge_events, discharge_out])

    handoff_btn.click(
        handoff_summary,
        inputs=[user_state, selected_case_id, handoff_problem, handoff_assessment, handoff_plan],
        outputs=[handoff_out, case_summary],
    )
    handoff_clear.click(lambda: empty_handoff(), outputs=[handoff_problem, handoff_assessment, handoff_plan, handoff_out])

    or_copy_history.click(copy_history_to_or, inputs=[history2], outputs=[or_history])
    or_autofill.click(
        autofill_or_history,
        inputs=[history2, history1, or_history],
        outputs=[or_history],
    )
    or_btn.click(
        or_briefing,
        inputs=[user_state, selected_case_id, or_history, or_meds, or_surgery, or_extra],
        outputs=[or_out, case_summary],
    )
    or_clear.click(lambda: empty_or(), outputs=[or_history, or_meds, or_surgery, or_extra, or_out])

    admin_user_search.submit(
        refresh_admin_user_search,
        inputs=[user_state, admin_user_search],
        outputs=[admin_user_selector, admin_user_summary, admin_user_json, admin_cases_json, admin_status, admin_case_selector, admin_case_preview],
    )
    admin_case_refresh_btn.click(
        refresh_admin_case_list,
        inputs=[user_state, admin_user_selector, admin_case_search],
        outputs=[admin_case_selector, admin_case_preview],
    )
    admin_case_search.submit(
        refresh_admin_case_list,
        inputs=[user_state, admin_user_selector, admin_case_search],
        outputs=[admin_case_selector, admin_case_preview],
    )
    admin_case_selector.change(
        load_admin_case_preview,
        inputs=[user_state, admin_user_selector, admin_case_selector],
        outputs=[admin_case_preview],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    print("DEBUG OPENAI exists:", bool(os.environ.get("OPENAI_API_KEY")))
    print("DEBUG APP_DATA_DIR:", APP_DATA_DIR)
    print("DEBUG USERS_DB_PATH:", USERS_DB_PATH)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
