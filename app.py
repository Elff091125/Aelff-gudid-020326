import os
import re
import json
import time
import random
import math
from datetime import datetime, date
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st

# Optional (best-effort) deps
try:
    import yaml  # pyyaml
except Exception:
    yaml = None

try:
    from rapidfuzz import fuzz  # rapidfuzz
except Exception:
    fuzz = None

try:
    import altair as alt
except Exception:
    alt = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

# -----------------------------
# Constants / Global Settings
# -----------------------------
CORAL = "#FF7F50"

OPENAI_ENV_KEYS = ["OPENAI_API_KEY"]
GEMINI_ENV_KEYS = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
ANTHROPIC_ENV_KEYS = ["ANTHROPIC_API_KEY"]
GROK_ENV_KEYS = ["XAI_API_KEY", "GROK_API_KEY"]

DEFAULT_DATASET_PATH = "defaultdataset.json"
DEFAULT_AGENTS_PATH = "agents.yaml"
DEFAULT_SKILL_PATH = "SKILL.md"

SUPPORTED_MODELS = [
    # OpenAI
    "gpt-4o-mini",
    "gpt-4.1-mini",
    # Gemini
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    # Anthropic (keep as options)
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    # Grok (xAI)
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

PAINTER_STYLES = [
    {"id": "monet", "name_en": "Monet", "name_zh": "莫內", "accent": "#7AA6C2"},
    {"id": "vangogh", "name_en": "Van Gogh", "name_zh": "梵谷", "accent": "#F2C14E"},
    {"id": "picasso", "name_en": "Picasso", "name_zh": "畢卡索", "accent": "#2D6A4F"},
    {"id": "dali", "name_en": "Dalí", "name_zh": "達利", "accent": "#9B5DE5"},
    {"id": "hokusai", "name_en": "Hokusai", "name_zh": "北齋", "accent": "#457B9D"},
    {"id": "matisse", "name_en": "Matisse", "name_zh": "馬諦斯", "accent": "#E63946"},
    {"id": "rembrandt", "name_en": "Rembrandt", "name_zh": "林布蘭", "accent": "#6D4C41"},
    {"id": "vermeer", "name_en": "Vermeer", "name_zh": "維梅爾", "accent": "#1D3557"},
    {"id": "pollock", "name_en": "Pollock", "name_zh": "波洛克", "accent": "#F72585"},
    {"id": "warhol", "name_en": "Warhol", "name_zh": "沃荷", "accent": "#4CC9F0"},
    {"id": "klimt", "name_en": "Klimt", "name_zh": "克林姆", "accent": "#C9A227"},
    {"id": "miro", "name_en": "Miró", "name_zh": "米羅", "accent": "#FF6B6B"},
    {"id": "kandinsky", "name_en": "Kandinsky", "name_zh": "康丁斯基", "accent": "#00BBF9"},
    {"id": "cezanne", "name_en": "Cézanne", "name_zh": "塞尚", "accent": "#6A994E"},
    {"id": "gauguin", "name_en": "Gauguin", "name_zh": "高更", "accent": "#F94144"},
    {"id": "turner", "name_en": "Turner", "name_zh": "透納", "accent": "#F8961E"},
    {"id": "caravaggio", "name_en": "Caravaggio", "name_zh": "卡拉瓦喬", "accent": "#2B2D42"},
    {"id": "kahlo", "name_en": "Kahlo", "name_zh": "卡蘿", "accent": "#43AA8B"},
    {"id": "magritte", "name_en": "Magritte", "name_zh": "馬格利特", "accent": "#277DA1"},
    {"id": "basquiat", "name_en": "Basquiat", "name_zh": "巴斯奇亞", "accent": "#F3722C"},
]

I18N = {
    "en": {
        "app_title": "WOW Search Studio",
        "nav": "Navigation",
        "page_search": "WOW Search Studio",
        "page_dashboard": "Dashboard",
        "page_dataset": "Dataset Studio",
        "page_agents": "Agent Studio",
        "page_factory": "Factory",
        "page_notes": "AI Note Keeper",
        "theme": "Theme",
        "light": "Light",
        "dark": "Dark",
        "language": "Language",
        "style": "Painter Style",
        "jackpot": "Jackpot",
        "api_status": "API Status",
        "datasets_status": "Datasets",
        "index_status": "Search Index",
        "agents_status": "Agents",
        "ready": "READY",
        "missing": "MISSING",
        "ok": "OK",
        "query": "Query",
        "search": "Search",
        "search_settings": "Search Settings",
        "exact_match": "Exact match",
        "fuzzy_threshold": "Fuzzy threshold",
        "limit": "Result limit per dataset",
        "field_weighting": "Field weighting",
        "balanced": "Balanced",
        "id_boosted": "ID-boosted",
        "narrative_boosted": "Narrative-boosted",
        "include_datasets": "Include datasets",
        "filters": "Refine / Filters",
        "relationship_explorer": "Relationship Explorer",
        "why_linked": "Why linked?",
        "suggestions": "Suggested next searches",
        "prompt_notebook": "Prompt Notebook (Keep prompt on results)",
        "run_agent": "Run agent",
        "pipeline": "Pipeline",
        "single_agent": "Single agent",
        "model": "Model",
        "max_tokens": "Max tokens",
        "temperature": "Temperature",
        "system_prompt": "System prompt",
        "user_prompt": "User prompt",
        "output": "Output",
        "pin": "Pin to Workspace",
        "workspace": "Workspace",
        "clear": "Clear",
        "notes_input": "Paste notes (text or markdown)",
        "transform": "Transform into organized markdown",
        "ai_magics": "AI Magics",
        "ai_keywords": "AI Keywords",
        "keywords": "Keywords (comma-separated)",
        "keyword_color": "Keyword color",
    },
    "zh-TW": {
        "app_title": "WOW 搜尋工作室",
        "nav": "導覽",
        "page_search": "WOW 搜尋工作室",
        "page_dashboard": "儀表板",
        "page_dataset": "資料集工作室",
        "page_agents": "代理人工作室",
        "page_factory": "工廠",
        "page_notes": "AI 筆記管家",
        "theme": "主題",
        "light": "亮色",
        "dark": "暗色",
        "language": "語言",
        "style": "畫家風格",
        "jackpot": "轉盤",
        "api_status": "API 狀態",
        "datasets_status": "資料集",
        "index_status": "搜尋索引",
        "agents_status": "代理人",
        "ready": "就緒",
        "missing": "缺少",
        "ok": "正常",
        "query": "查詢",
        "search": "搜尋",
        "search_settings": "搜尋設定",
        "exact_match": "完全比對",
        "fuzzy_threshold": "模糊比對門檻",
        "limit": "每資料集結果上限",
        "field_weighting": "欄位權重模式",
        "balanced": "平衡",
        "id_boosted": "ID 強化",
        "narrative_boosted": "敘事強化",
        "include_datasets": "包含資料集",
        "filters": "精煉 / 篩選",
        "relationship_explorer": "關聯探索器",
        "why_linked": "為何關聯？",
        "suggestions": "建議下一步搜尋",
        "prompt_notebook": "提示詞筆記本（保留提示詞在結果上）",
        "run_agent": "執行代理人",
        "pipeline": "流程管線",
        "single_agent": "單一代理人",
        "model": "模型",
        "max_tokens": "最大 tokens",
        "temperature": "溫度",
        "system_prompt": "系統提示詞",
        "user_prompt": "使用者提示詞",
        "output": "輸出",
        "pin": "釘選到工作區",
        "workspace": "工作區",
        "clear": "清除",
        "notes_input": "貼上筆記（文字或 Markdown）",
        "transform": "整理成有結構的 Markdown",
        "ai_magics": "AI 魔法",
        "ai_keywords": "AI 關鍵字",
        "keywords": "關鍵字（逗號分隔）",
        "keyword_color": "關鍵字顏色",
    },
}


# -----------------------------
# Utilities
# -----------------------------
def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return I18N.get(lang, I18N["en"]).get(key, key)


def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def safe_write_text(path: str, content: str) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception:
        return False


def normalize_str(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s_lower = s.lower()
    # Keep meaningful separators for IDs; remove most punctuation for linking
    s_lower = re.sub(r"[^\w\s\-:/]", "", s_lower)
    return s_lower.strip()


def parse_date_any(x: Any) -> Optional[datetime]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() in ["nan", "none", "null", "—", "-"]:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        # pandas fallback
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def get_env_key(keys: List[str]) -> Optional[str]:
    for k in keys:
        v = os.getenv(k)
        if v and v.strip():
            return v.strip()
    return None


def infer_provider_from_model(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("gpt-"):
        return "openai"
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith("grok-"):
        return "grok"
    return "openai"


def require_rapidfuzz() -> bool:
    return fuzz is not None


def stable_hash(s: str) -> str:
    # lightweight stable-ish id for record nodes
    return str(abs(hash(s)))


# -----------------------------
# WOW UI Styling
# -----------------------------
def apply_wow_css(theme: str, accent: str):
    is_dark = theme == "dark"
    bg = "#0E1117" if is_dark else "#FFFFFF"
    fg = "#EAEAEA" if is_dark else "#111111"
    panel = "#111827" if is_dark else "#F7F7FB"
    border = "#2A2F3A" if is_dark else "#E8E8EF"

    css = f"""
    <style>
      :root {{
        --wow-bg: {bg};
        --wow-fg: {fg};
        --wow-panel: {panel};
        --wow-border: {border};
        --wow-accent: {accent};
        --wow-coral: {CORAL};
      }}
      .stApp {{
        background: var(--wow-bg);
        color: var(--wow-fg);
      }}
      .wow-chip {{
        display: inline-flex;
        align-items: center;
        gap: .5rem;
        border: 1px solid var(--wow-border);
        background: var(--wow-panel);
        padding: .25rem .6rem;
        border-radius: 999px;
        font-size: 0.85rem;
        white-space: nowrap;
      }}
      .wow-dot {{
        width: .6rem;
        height: .6rem;
        border-radius: 50%;
        background: var(--wow-accent);
        display: inline-block;
      }}
      .wow-dot.ok {{ background: #22c55e; }}
      .wow-dot.warn {{ background: #f59e0b; }}
      .wow-dot.bad {{ background: #ef4444; }}
      .wow-title {{
        font-weight: 800;
        letter-spacing: .2px;
      }}
      .wow-accent {{
        color: var(--wow-accent);
        font-weight: 700;
      }}
      .wow-coral {{
        color: var(--wow-coral);
        font-weight: 700;
      }}
      .wow-kpi {{
        border: 1px solid var(--wow-border);
        background: var(--wow-panel);
        padding: .8rem .9rem;
        border-radius: 14px;
      }}
      .wow-box {{
        border: 1px solid var(--wow-border);
        background: var(--wow-panel);
        padding: .8rem .9rem;
        border-radius: 14px;
      }}
      .wow-small {{
        font-size: .88rem;
        opacity: .92;
      }}
      .wow-hr {{
        border: none;
        height: 1px;
        background: var(--wow-border);
        margin: .8rem 0;
      }}
      .wow-highlight {{
        background: color-mix(in srgb, var(--wow-coral) 18%, transparent);
        border-bottom: 1px solid color-mix(in srgb, var(--wow-coral) 55%, transparent);
        padding: 0 .15rem;
        border-radius: .25rem;
      }}
      .wow-subtle {{
        opacity: .9;
      }}
      .wow-code {{
        border: 1px solid var(--wow-border);
        background: color-mix(in srgb, var(--wow-panel) 88%, black);
        padding: .6rem .7rem;
        border-radius: 12px;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def painter_label(style: Dict[str, Any], lang: str) -> str:
    return style["name_zh"] if lang == "zh-TW" else style["name_en"]


# -----------------------------
# Dataset Loading
# -----------------------------
def load_default_dataset_json(path: str) -> Dict[str, pd.DataFrame]:
    """
    Supports two formats:
      A) {"datasets": {"510k": [...], "recall": [...], "adr": [...], "gudid": [...]} }
      B) {"510k": [...], "recall": [...], "adr": [...], "gudid": [...]}
    """
    if not os.path.exists(path):
        return {"510k": pd.DataFrame(), "recall": pd.DataFrame(), "adr": pd.DataFrame(), "gudid": pd.DataFrame()}

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    ds = obj.get("datasets", obj)
    out = {}
    for k in ["510k", "recall", "adr", "gudid"]:
        records = ds.get(k, [])
        if isinstance(records, dict):
            records = [records]
        out[k] = pd.DataFrame(records)
    return out


def init_datasets():
    if "datasets" not in st.session_state:
        st.session_state["datasets"] = {"510k": pd.DataFrame(), "recall": pd.DataFrame(), "adr": pd.DataFrame(), "gudid": pd.DataFrame()}
    if "dataset_source" not in st.session_state:
        st.session_state["dataset_source"] = "default"

    # Load defaults on first run
    if st.session_state["dataset_source"] == "default" and st.session_state["datasets"]["510k"].empty and os.path.exists(DEFAULT_DATASET_PATH):
        st.session_state["datasets"] = load_default_dataset_json(DEFAULT_DATASET_PATH)


def dataset_counts() -> Dict[str, int]:
    ds = st.session_state.get("datasets", {})
    return {k: int(len(ds.get(k, pd.DataFrame()))) for k in ["510k", "recall", "adr", "gudid"]}


# -----------------------------
# Agents Loading
# -----------------------------
def load_agents_yaml(path: str) -> List[Dict[str, Any]]:
    if yaml is None:
        return []
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        agents = obj.get("agents", obj.get("Agents", [])) or []
        # Normalize minimal fields
        normalized = []
        for a in agents:
            if not isinstance(a, dict):
                continue
            normalized.append({
                "id": a.get("id") or a.get("name") or f"agent_{len(normalized)+1}",
                "name": a.get("name") or a.get("id") or f"Agent {len(normalized)+1}",
                "description": a.get("description", ""),
                "provider": a.get("provider", ""),
                "model": a.get("model", ""),
                "temperature": a.get("temperature", 0.2),
                "max_tokens": a.get("max_tokens", 12000),
                "system_prompt": a.get("system_prompt", ""),
                "user_prompt": a.get("user_prompt", ""),
            })
        return normalized
    except Exception:
        return []


def init_agents():
    if "agents" not in st.session_state:
        st.session_state["agents"] = load_agents_yaml(DEFAULT_AGENTS_PATH)
    if "skill_md" not in st.session_state:
        st.session_state["skill_md"] = safe_read_text(DEFAULT_SKILL_PATH)


# -----------------------------
# API Key UX (env-first; session fallback)
# -----------------------------
def init_api_keys():
    if "api_keys" not in st.session_state:
        st.session_state["api_keys"] = {
            "openai": {"source": "missing", "value": None},
            "gemini": {"source": "missing", "value": None},
            "anthropic": {"source": "missing", "value": None},
            "grok": {"source": "missing", "value": None},
        }

    # Environment keys
    openai_env = get_env_key(OPENAI_ENV_KEYS)
    gemini_env = get_env_key(GEMINI_ENV_KEYS)
    anthropic_env = get_env_key(ANTHROPIC_ENV_KEYS)
    grok_env = get_env_key(GROK_ENV_KEYS)

    if openai_env:
        st.session_state["api_keys"]["openai"] = {"source": "env", "value": openai_env}
    if gemini_env:
        st.session_state["api_keys"]["gemini"] = {"source": "env", "value": gemini_env}
    if anthropic_env:
        st.session_state["api_keys"]["anthropic"] = {"source": "env", "value": anthropic_env}
    if grok_env:
        st.session_state["api_keys"]["grok"] = {"source": "env", "value": grok_env}


def api_key_status(provider: str) -> Tuple[str, str]:
    info = st.session_state["api_keys"].get(provider, {"source": "missing", "value": None})
    if info["source"] == "env":
        return ("ok", "Authenticated via Environment")
    if info["source"] == "session" and info["value"]:
        return ("ok", "Authenticated via Session")
    return ("bad", "Missing")


def api_key_settings_ui():
    st.sidebar.markdown(f"### {t('api_status')}")
    for provider, label, env_keys in [
        ("openai", "OpenAI", OPENAI_ENV_KEYS),
        ("gemini", "Gemini", GEMINI_ENV_KEYS),
        ("anthropic", "Anthropic", ANTHROPIC_ENV_KEYS),
        ("grok", "Grok/xAI", GROK_ENV_KEYS),
    ]:
        info = st.session_state["api_keys"].get(provider, {"source": "missing", "value": None})
        status, msg = api_key_status(provider)

        st.sidebar.markdown(
            f"<div class='wow-chip'><span class='wow-dot {status}'></span><b>{label}</b>&nbsp;<span class='wow-small'>{msg}</span></div>",
            unsafe_allow_html=True
        )

        if info["source"] != "env":
            # Allow input only if not authenticated via environment
            key = st.sidebar.text_input(
                f"{label} API Key",
                value="",
                type="password",
                help=f"Stored only in Streamlit session_state. Env keys ({', '.join(env_keys)}) override UI input.",
                key=f"api_input_{provider}",
            )
            if key and key.strip():
                st.session_state["api_keys"][provider] = {"source": "session", "value": key.strip()}


# -----------------------------
# LLM Calls (best-effort)
# -----------------------------
def call_openai(model: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
    key = st.session_state["api_keys"]["openai"]["value"]
    if not key:
        raise RuntimeError("OpenAI API key missing.")
    try:
        # New-style OpenAI SDK
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        return resp.choices[0].message.content or ""
    except Exception:
        # Legacy fallback
        import openai
        openai.api_key = key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        return resp["choices"][0]["message"]["content"] or ""


def call_gemini(model: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
    key = st.session_state["api_keys"]["gemini"]["value"]
    if not key:
        raise RuntimeError("Gemini API key missing.")
    import google.generativeai as genai
    genai.configure(api_key=key)
    # Gemini "system instruction" support depends on SDK; keep robust
    prompt = f"System:\n{system}\n\nUser:\n{user}".strip()
    gm = genai.GenerativeModel(model_name=model)
    resp = gm.generate_content(
        prompt,
        generation_config={
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
        },
    )
    return getattr(resp, "text", "") or ""


def call_anthropic(model: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
    key = st.session_state["api_keys"]["anthropic"]["value"]
    if not key:
        raise RuntimeError("Anthropic API key missing.")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            system=system,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            messages=[{"role": "user", "content": user}],
        )
        # content list
        parts = []
        for c in getattr(resp, "content", []) or []:
            if hasattr(c, "text"):
                parts.append(c.text)
            elif isinstance(c, dict) and "text" in c:
                parts.append(c["text"])
        return "\n".join(parts).strip()
    except Exception as e:
        raise RuntimeError(f"Anthropic call failed: {e}")


def call_grok(model: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
    # Many Grok/xAI deployments are OpenAI-compatible; try OpenAI-compatible endpoint if provided
    key = st.session_state["api_keys"]["grok"]["value"]
    if not key:
        raise RuntimeError("Grok/xAI API key missing.")
    base_url = os.getenv("XAI_BASE_URL", "").strip()  # optional
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=base_url or None)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"Grok/xAI call failed: {e}")


def call_llm(model: str, system: str, user: str, max_tokens: int = 12000, temperature: float = 0.2, provider: Optional[str] = None) -> str:
    provider = provider or infer_provider_from_model(model)
    if provider == "openai":
        return call_openai(model, system, user, max_tokens, temperature)
    if provider == "gemini":
        return call_gemini(model, system, user, max_tokens, temperature)
    if provider == "anthropic":
        return call_anthropic(model, system, user, max_tokens, temperature)
    if provider == "grok":
        return call_grok(model, system, user, max_tokens, temperature)
    return call_openai(model, system, user, max_tokens, temperature)


# -----------------------------
# Search / Scoring
# -----------------------------
DATASET_FIELDS = {
    "510k": {
        "id_field": "k_number",
        "date_field": "decision_date",
        "fields_balanced": ["k_number", "device_name", "applicant", "manufacturer_name", "product_code", "panel", "decision", "summary"],
        "fields_id_boosted": ["k_number", "product_code", "manufacturer_name", "device_name", "summary"],
        "fields_narrative_boosted": ["summary", "device_name", "manufacturer_name", "product_code", "k_number"],
    },
    "recall": {
        "id_field": "recall_number",
        "date_field": "event_date",
        "fields_balanced": ["recall_number", "firm_name", "manufacturer_name", "product_description", "product_code", "reason_for_recall", "status", "recall_class"],
        "fields_id_boosted": ["recall_number", "product_code", "manufacturer_name", "firm_name", "product_description", "reason_for_recall"],
        "fields_narrative_boosted": ["reason_for_recall", "product_description", "manufacturer_name", "product_code", "recall_number"],
    },
    "adr": {
        "id_field": "adverse_event_id",
        "date_field": "report_date",
        "fields_balanced": ["adverse_event_id", "manufacturer_name", "brand_name", "product_code", "device_problem", "patient_outcome", "event_type", "narrative", "udi_di", "recall_number_link"],
        "fields_id_boosted": ["adverse_event_id", "udi_di", "recall_number_link", "product_code", "manufacturer_name", "brand_name", "device_problem", "narrative"],
        "fields_narrative_boosted": ["narrative", "device_problem", "manufacturer_name", "brand_name", "product_code", "udi_di", "recall_number_link"],
    },
    "gudid": {
        "id_field": "primary_di",
        "date_field": "publish_date",
        "fields_balanced": ["primary_di", "udi_di", "device_description", "manufacturer_name", "brand_name", "product_code", "gmdn_term", "device_class"],
        "fields_id_boosted": ["primary_di", "udi_di", "product_code", "manufacturer_name", "brand_name", "device_description"],
        "fields_narrative_boosted": ["device_description", "gmdn_term", "manufacturer_name", "brand_name", "product_code", "primary_di", "udi_di"],
    },
}


def score_record(q: str, record: Dict[str, Any], fields: List[str], exact: bool, threshold: int) -> Tuple[float, List[str]]:
    """
    Returns (score 0..100, matched_fields)
    """
    qn = normalize_str(q)
    if not qn:
        return (0.0, [])

    matched = []
    best = 0.0

    for f in fields:
        val = record.get(f, "")
        vn = normalize_str(val)
        if not vn:
            continue

        if exact:
            if qn in vn:
                s = 100.0
            else:
                s = 0.0
        else:
            if fuzz is None:
                # Fallback simple containment
                s = 100.0 if qn in vn else 0.0
            else:
                s = float(fuzz.token_set_ratio(qn, vn))

        if s >= threshold:
            matched.append(f)
        if s > best:
            best = s

    return (best, matched)


def search_dataset(df: pd.DataFrame, dataset_key: str, q: str, exact: bool, threshold: int, limit: int, weighting: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    spec = DATASET_FIELDS[dataset_key]
    if weighting == "id":
        fields = spec["fields_id_boosted"]
    elif weighting == "narrative":
        fields = spec["fields_narrative_boosted"]
    else:
        fields = spec["fields_balanced"]

    records = df.to_dict(orient="records")
    out_rows = []
    for r in records:
        s, matched = score_record(q, r, fields, exact=exact, threshold=threshold)
        if s > 0:
            out_rows.append({**r, "__score": s, "__matched_fields": ", ".join(matched)})

    if not out_rows:
        return pd.DataFrame()

    out = pd.DataFrame(out_rows)
    out = out.sort_values("__score", ascending=False).head(int(limit)).reset_index(drop=True)
    return out


# -----------------------------
# Cross-dataset relationships
# -----------------------------
LINK_KEYS = {
    "udi_di": ("udi_di", "primary_di"),
    "recall_number": ("recall_number", "recall_number_link"),
    "manufacturer": ("manufacturer_name", "firm_name", "applicant"),
    "product_code": ("product_code",),
    "brand_device": ("brand_name", "device_name", "product_description", "device_description"),
}


def extract_entities(df: pd.DataFrame, dataset_key: str) -> Dict[str, List[str]]:
    """
    Returns entity_type -> list of normalized values (non-empty)
    """
    ents = {k: [] for k in ["udi_di", "recall_number", "manufacturer", "product_code", "brand_device", "k_number"]}
    if df is None or df.empty:
        return ents

    for _, row in df.iterrows():
        r = row.to_dict()
        # UDI/DI
        for f in LINK_KEYS["udi_di"]:
            if f in r and r.get(f):
                v = normalize_str(r.get(f))
                if v:
                    ents["udi_di"].append(v)
        # recall number
        for f in LINK_KEYS["recall_number"]:
            if f in r and r.get(f):
                v = normalize_str(r.get(f))
                if v:
                    ents["recall_number"].append(v)
        # manufacturer
        for f in LINK_KEYS["manufacturer"]:
            if f in r and r.get(f):
                v = normalize_str(r.get(f))
                if v:
                    ents["manufacturer"].append(v)
        # product_code
        for f in LINK_KEYS["product_code"]:
            if f in r and r.get(f):
                v = normalize_str(r.get(f))
                if v:
                    ents["product_code"].append(v)
        # brand/device text
        for f in LINK_KEYS["brand_device"]:
            if f in r and r.get(f):
                v = normalize_str(r.get(f))
                if v:
                    ents["brand_device"].append(v)

        if dataset_key == "510k" and r.get("k_number"):
            ents["k_number"].append(normalize_str(r.get("k_number")))

    for k in ents:
        ents[k] = list(sorted(set(ents[k])))
    return ents


def compute_shared_entities(results_by_ds: Dict[str, pd.DataFrame]) -> Dict[str, set]:
    """
    Returns entity_type -> set of values that appear in >=2 datasets
    """
    appearances = {etype: {} for etype in ["udi_di", "recall_number", "manufacturer", "product_code"]}
    for ds_key, df in results_by_ds.items():
        ents = extract_entities(df, ds_key)
        for etype in appearances:
            for v in ents.get(etype, []):
                appearances[etype].setdefault(v, set()).add(ds_key)

    shared = {}
    for etype, m in appearances.items():
        shared[etype] = {v for v, dsset in m.items() if len(dsset) >= 2}
    return shared


def linked_badges_for_row(row: Dict[str, Any], shared: Dict[str, set]) -> List[str]:
    badges = []
    mfg = normalize_str(row.get("manufacturer_name") or row.get("firm_name") or row.get("applicant"))
    pc = normalize_str(row.get("product_code"))
    udi = normalize_str(row.get("udi_di") or row.get("primary_di"))
    rn = normalize_str(row.get("recall_number") or row.get("recall_number_link"))

    if udi and udi in shared.get("udi_di", set()):
        badges.append("UDI")
    if rn and rn in shared.get("recall_number", set()):
        badges.append("RECALL#")
    if mfg and mfg in shared.get("manufacturer", set()):
        badges.append("MFG")
    if pc and pc in shared.get("product_code", set()):
        badges.append("PC")
    return badges


def build_relationship_graph(results_by_ds: Dict[str, pd.DataFrame], shared: Dict[str, set]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str, int]]]:
    """
    Returns (nodes, edges) where nodes are dicts for plotting, edges are (src, dst, etype, weight)
    We model entity nodes + dataset nodes.
    """
    nodes = []
    edges = []
    # Dataset nodes
    for ds_key in ["510k", "recall", "adr", "gudid"]:
        nodes.append({"id": f"ds:{ds_key}", "label": ds_key.upper(), "kind": "dataset"})

    # Entity nodes (shared only)
    for etype, values in shared.items():
        for v in sorted(values):
            label = v
            nodes.append({"id": f"ent:{etype}:{v}", "label": label, "kind": "entity", "etype": etype})

    # Edges dataset<->entity based on occurrences in results
    for ds_key, df in results_by_ds.items():
        if df is None or df.empty:
            continue
        ents = extract_entities(df, ds_key)
        for etype, values in ents.items():
            if etype not in shared:
                continue
            for v in values:
                if v in shared[etype]:
                    edges.append((f"ds:{ds_key}", f"ent:{etype}:{v}", etype, 1))

    # Collapse edge weights
    weights = {}
    for s, d, etype, w in edges:
        weights[(s, d, etype)] = weights.get((s, d, etype), 0) + w
    edges2 = [(s, d, etype, w) for (s, d, etype), w in weights.items()]
    return nodes, edges2


# -----------------------------
# UI Components
# -----------------------------
def status_strip():
    counts = dataset_counts()
    agents_ok = (yaml is not None) and os.path.exists(DEFAULT_AGENTS_PATH)
    skill_ok = os.path.exists(DEFAULT_SKILL_PATH)

    openai_s, openai_msg = api_key_status("openai")
    gemini_s, gemini_msg = api_key_status("gemini")

    datasets_total = sum(counts.values())
    index_ready = "READY" if datasets_total > 0 else "MISSING"

    c1, c2, c3, c4, c5 = st.columns([1.6, 2.2, 1.6, 1.6, 2.0])

    with c1:
        st.markdown(f"<div class='wow-chip'><span class='wow-dot'></span><span class='wow-title'>{t('app_title')}</span></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div class='wow-chip'><span class='wow-dot {openai_s}'></span><b>OpenAI</b>&nbsp;<span class='wow-small'>{openai_msg}</span></div> "
            f"<div style='display:inline-block;width:.4rem'></div>"
            f"<div class='wow-chip'><span class='wow-dot {gemini_s}'></span><b>Gemini</b>&nbsp;<span class='wow-small'>{gemini_msg}</span></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='wow-chip'><span class='wow-dot ok'></span><b>{t('datasets_status')}</b>&nbsp;<span class='wow-small'>510k {counts['510k']} | Recall {counts['recall']} | ADR {counts['adr']} | GUDID {counts['gudid']}</span></div>",
            unsafe_allow_html=True
        )
    with c4:
        dot = "ok" if index_ready == "READY" else "bad"
        st.markdown(
            f"<div class='wow-chip'><span class='wow-dot {dot}'></span><b>{t('index_status')}</b>&nbsp;<span class='wow-small'>{index_ready}</span></div>",
            unsafe_allow_html=True
        )
    with c5:
        dot = "ok" if (agents_ok and skill_ok) else "warn"
        msg = "agents.yaml & SKILL.md OK" if (agents_ok and skill_ok) else "Check agents.yaml / SKILL.md"
        st.markdown(
            f"<div class='wow-chip'><span class='wow-dot {dot}'></span><b>{t('agents_status')}</b>&nbsp;<span class='wow-small'>{msg}</span></div>",
            unsafe_allow_html=True
        )


def kpi_row(results_by_ds: Dict[str, pd.DataFrame], shared: Dict[str, set]):
    counts = {k: (0 if results_by_ds.get(k) is None else int(len(results_by_ds.get(k)))) for k in ["510k", "recall", "adr", "gudid"]}
    total = sum(counts.values())
    top_shared_mfg = next(iter(shared.get("manufacturer", set())), "")
    top_shared_pc = next(iter(shared.get("product_code", set())), "")
    top_shared_udi = next(iter(shared.get("udi_di", set())), "")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"<div class='wow-kpi'><div class='wow-small'>Total hits</div><div style='font-size:1.3rem;font-weight:800'>{total}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='wow-kpi'><div class='wow-small'>510(k)</div><div style='font-size:1.3rem;font-weight:800'>{counts['510k']}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='wow-kpi'><div class='wow-small'>Recall</div><div style='font-size:1.3rem;font-weight:800'>{counts['recall']}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='wow-kpi'><div class='wow-small'>ADR</div><div style='font-size:1.3rem;font-weight:800'>{counts['adr']}</div></div>", unsafe_allow_html=True)
    with c5:
        st.markdown(f"<div class='wow-kpi'><div class='wow-small'>GUDID</div><div style='font-size:1.3rem;font-weight:800'>{counts['gudid']}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-hr'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='wow-box'>"
        f"<div class='wow-small wow-subtle'>Cross-dataset highlights (shared entities)</div>"
        f"<div style='margin-top:.3rem'>"
        f"<span class='wow-coral'>Manufacturer</span>: <span class='wow-small'>{top_shared_mfg or '—'}</span> &nbsp; | &nbsp; "
        f"<span class='wow-coral'>Product Code</span>: <span class='wow-small'>{top_shared_pc or '—'}</span> &nbsp; | &nbsp; "
        f"<span class='wow-coral'>UDI/DI</span>: <span class='wow-small'>{top_shared_udi or '—'}</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )


def highlight_shared_styler(df: pd.DataFrame, shared: Dict[str, set]) -> Optional[pd.io.formats.style.Styler]:
    if df is None or df.empty:
        return None

    # columns where highlighting makes sense
    cols = [c for c in df.columns if c in [
        "manufacturer_name", "firm_name", "applicant",
        "product_code", "udi_di", "primary_di",
        "recall_number", "recall_number_link",
        "k_number"
    ]]
    if not cols:
        return df.style

    def cell_style(val, col):
        v = normalize_str(val)
        if not v:
            return ""
        # Determine type by column name
        if col in ["udi_di", "primary_di"] and v in shared.get("udi_di", set()):
            return f"background-color: {CORAL}22; border-bottom: 1px solid {CORAL}88;"
        if col in ["recall_number", "recall_number_link"] and v in shared.get("recall_number", set()):
            return f"background-color: {CORAL}22; border-bottom: 1px solid {CORAL}88;"
        if col in ["product_code"] and v in shared.get("product_code", set()):
            return f"background-color: {CORAL}22; border-bottom: 1px solid {CORAL}88;"
        if col in ["manufacturer_name", "firm_name", "applicant"] and v in shared.get("manufacturer", set()):
            return f"background-color: {CORAL}22; border-bottom: 1px solid {CORAL}88;"
        return ""

    def apply_styles(s: pd.Series):
        return [cell_style(v, s.name) for v in s]

    styler = df.style
    for c in cols:
        styler = styler.apply(apply_styles, axis=0, subset=[c])
    return styler


def dataset_panel(title: str, ds_key: str, df_hits: pd.DataFrame, shared: Dict[str, set], accent: str):
    st.markdown(f"### {title} <span class='wow-small'>(hits: {0 if df_hits is None else len(df_hits)})</span>", unsafe_allow_html=True)

    if df_hits is None or df_hits.empty:
        st.info("No results.")
        return

    # Add linked badges
    df = df_hits.copy()
    df["linked_to"] = df.apply(lambda r: " ".join(linked_badges_for_row(r.to_dict(), shared)) if shared else "", axis=1)

    # Mini charts (best-effort)
    spec = DATASET_FIELDS[ds_key]
    date_field = spec["date_field"]

    c1, c2 = st.columns([1.3, 1.0])

    with c1:
        if alt is not None and date_field in df.columns:
            dfc = df.copy()
            dfc["_dt"] = dfc[date_field].apply(parse_date_any)
            dfc = dfc[dfc["_dt"].notna()]
            if not dfc.empty:
                chart = alt.Chart(dfc).mark_circle(size=70, opacity=0.75).encode(
                    x=alt.X("_dt:T", title=date_field),
                    y=alt.Y("__score:Q", title="score"),
                    tooltip=[spec["id_field"], "__score", "__matched_fields"],
                    color=alt.value(accent),
                ).properties(height=180)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("Timeline chart: no parseable dates.")
        else:
            st.caption("Timeline chart unavailable (Altair missing or date field missing).")

    with c2:
        # Simple distribution chart: product_code top N (if available)
        if alt is not None and "product_code" in df.columns:
            top = df["product_code"].fillna("").astype(str)
            top = top[top != ""].value_counts().head(10).reset_index()
            top.columns = ["product_code", "count"]
            if not top.empty:
                chart = alt.Chart(top).mark_bar().encode(
                    y=alt.Y("product_code:N", sort="-x"),
                    x=alt.X("count:Q"),
                    color=alt.value(accent),
                    tooltip=["product_code", "count"],
                ).properties(height=180)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("Distribution chart: no product_code values.")
        else:
            st.caption("Distribution chart unavailable.")

    # Table
    # Prioritized columns per dataset
    column_priority = {
        "510k": ["k_number", "device_name", "applicant", "manufacturer_name", "product_code", "decision_date", "decision", "device_class", "panel", "__score", "__matched_fields", "linked_to"],
        "recall": ["recall_number", "recall_class", "status", "firm_name", "manufacturer_name", "product_description", "product_code", "reason_for_recall", "event_date", "__score", "__matched_fields", "linked_to"],
        "adr": ["adverse_event_id", "report_date", "event_type", "patient_outcome", "device_problem", "manufacturer_name", "brand_name", "product_code", "udi_di", "recall_number_link", "__score", "__matched_fields", "linked_to"],
        "gudid": ["primary_di", "udi_di", "device_description", "manufacturer_name", "brand_name", "product_code", "device_class", "mri_safety", "publish_date", "__score", "__matched_fields", "linked_to"],
    }[ds_key]
    cols = [c for c in column_priority if c in df.columns] + [c for c in df.columns if c not in column_priority]
    df_view = df[cols].copy()

    styler = highlight_shared_styler(df_view, shared)
    st.dataframe(
        styler if styler is not None else df_view,
        use_container_width=True,
        height=280,
        hide_index=True,
    )

    # Selection mechanism (stable across Streamlit versions)
    with st.expander("Select a record to open detail drawer"):
        # Provide a dropdown based selector as a reliable alternative
        id_field = spec["id_field"]
        if id_field in df.columns:
            options = df[id_field].fillna("").astype(str).tolist()
            sel = st.selectbox(f"Select {id_field}", options=[""] + options, key=f"sel_{ds_key}")
            if sel:
                rec = df[df[id_field].astype(str) == sel].head(1).to_dict(orient="records")[0]
                st.session_state["selected_record"] = {"dataset": ds_key, "record": rec}
        else:
            st.caption("No identifiable id field to select.")


def detail_drawer(results_by_ds: Dict[str, pd.DataFrame], shared: Dict[str, set], accent: str):
    sel = st.session_state.get("selected_record")
    if not sel:
        st.markdown("<div class='wow-box wow-small'>Click a record in any dataset panel to open the detail drawer.</div>", unsafe_allow_html=True)
        return

    ds_key = sel["dataset"]
    rec = sel["record"]
    spec = DATASET_FIELDS[ds_key]
    id_field = spec["id_field"]
    rid = rec.get(id_field, "—")

    st.markdown(
        f"<div class='wow-box'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:1rem'>"
        f"<div>"
        f"<div class='wow-small'>{ds_key.upper()} record</div>"
        f"<div style='font-size:1.15rem;font-weight:900'><span class='wow-coral'>{rid}</span></div>"
        f"<div class='wow-small'>score: <span class='wow-accent'>{rec.get('__score', '—')}</span> &nbsp; | matched: {rec.get('__matched_fields','')}</div>"
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    tab_overview, tab_linked, tab_refine, tab_agent, tab_raw = st.tabs(["Overview", "Linked Records", "Search Refinement", "Agent Actions", "Raw JSON"])

    with tab_overview:
        # Render key fields with shared highlights
        fields = [c for c in rec.keys() if not c.startswith("__")]
        # prioritize
        pref = DATASET_FIELDS[ds_key]["fields_balanced"]
        fields = [f for f in pref if f in fields] + [f for f in fields if f not in pref]
        for f in fields[:30]:
            v = rec.get(f, "")
            vn = normalize_str(v)
            shared_hit = (
                (f in ["manufacturer_name", "firm_name", "applicant"] and vn in shared.get("manufacturer", set())) or
                (f in ["product_code"] and vn in shared.get("product_code", set())) or
                (f in ["udi_di", "primary_di"] and vn in shared.get("udi_di", set())) or
                (f in ["recall_number", "recall_number_link"] and vn in shared.get("recall_number", set()))
            )
            if shared_hit:
                st.markdown(f"**{f}**: <span class='wow-highlight'>{v}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**{f}**: {v}")

        # Quick action: pin
        if st.button(t("pin"), key="pin_detail"):
            st.session_state.setdefault("workspace", [])
            st.session_state["workspace"].append({"dataset": ds_key, "record": rec, "pinned_at": now_iso()})
            st.success("Pinned to Workspace.")

    with tab_linked:
        st.caption("Cross-dataset links are computed using UDI/DI, recall number, manufacturer, and product_code.")

        mfg = normalize_str(rec.get("manufacturer_name") or rec.get("firm_name") or rec.get("applicant"))
        pc = normalize_str(rec.get("product_code"))
        udi = normalize_str(rec.get("udi_di") or rec.get("primary_di"))
        rn = normalize_str(rec.get("recall_number") or rec.get("recall_number_link"))

        def filter_linked(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            keep = np.zeros(len(df), dtype=bool)
            if udi:
                for col in ["udi_di", "primary_di"]:
                    if col in df.columns:
                        keep |= df[col].astype(str).map(normalize_str) == udi
            if rn:
                for col in ["recall_number", "recall_number_link"]:
                    if col in df.columns:
                        keep |= df[col].astype(str).map(normalize_str) == rn
            if pc and "product_code" in df.columns:
                keep |= df["product_code"].astype(str).map(normalize_str) == pc
            if mfg:
                for col in ["manufacturer_name", "firm_name", "applicant"]:
                    if col in df.columns:
                        keep |= df[col].astype(str).map(normalize_str) == mfg
            return df[keep].head(50)

        for other in ["510k", "recall", "adr", "gudid"]:
            if other == ds_key:
                continue
            df_other = results_by_ds.get(other, pd.DataFrame())
            linked = filter_linked(df_other)
            st.markdown(f"#### Linked in {other.upper()} (max 50)")
            if linked is None or linked.empty:
                st.info("No linked records in current hit sets.")
            else:
                st.dataframe(linked.head(50), use_container_width=True, hide_index=True)

    with tab_refine:
        st.markdown("One-click pivots:")
        chips = []
        if rec.get("product_code"):
            chips.append(("product_code", str(rec.get("product_code"))))
        if rec.get("manufacturer_name") or rec.get("firm_name") or rec.get("applicant"):
            chips.append(("manufacturer", str(rec.get("manufacturer_name") or rec.get("firm_name") or rec.get("applicant"))))
        if rec.get("udi_di") or rec.get("primary_di"):
            chips.append(("udi_di", str(rec.get("udi_di") or rec.get("primary_di"))))
        if rec.get("recall_number") or rec.get("recall_number_link"):
            chips.append(("recall_number", str(rec.get("recall_number") or rec.get("recall_number_link"))))

        cols = st.columns(min(4, max(1, len(chips))))
        for i, (k, v) in enumerate(chips):
            with cols[i % len(cols)]:
                if st.button(f"Search {k}: {v}", key=f"chip_{k}_{i}"):
                    st.session_state["query"] = v
                    st.session_state["auto_search"] = True
                    st.rerun()

    with tab_agent:
        agents = st.session_state.get("agents", [])
        if not agents:
            st.warning("No agents loaded (agents.yaml missing or PyYAML not installed).")
        else:
            agent_names = [f"{a['name']} ({a['id']})" for a in agents]
            idx = st.selectbox("Select agent", options=list(range(len(agents))), format_func=lambda i: agent_names[i], key="detail_agent_sel")

            a = agents[idx]
            default_model = a.get("model") or "gpt-4o-mini"
            model = st.selectbox(t("model"), options=SUPPORTED_MODELS, index=max(0, SUPPORTED_MODELS.index(default_model) if default_model in SUPPORTED_MODELS else 0), key="detail_agent_model")
            max_tokens = st.number_input(t("max_tokens"), min_value=256, max_value=200000, value=int(a.get("max_tokens", 12000)), step=256, key="detail_agent_max_tokens")
            temperature = st.slider(t("temperature"), 0.0, 1.0, float(a.get("temperature", 0.2)), 0.05, key="detail_agent_temp")

            system_prompt = st.text_area(t("system_prompt"), value=(st.session_state.get("skill_md","") + "\n\n" + (a.get("system_prompt") or "")).strip(), height=160, key="detail_agent_system")
            user_prompt = st.text_area(t("user_prompt"), value=(a.get("user_prompt") or "Analyze the selected record. Quote evidence. Mark gaps explicitly.").strip(), height=120, key="detail_agent_user")
            payload = json.dumps(rec, ensure_ascii=False, indent=2)

            if st.button(t("run_agent"), key="detail_agent_run"):
                try:
                    with st.spinner("Running agent..."):
                        out = call_llm(
                            model=model,
                            provider=infer_provider_from_model(model),
                            system=system_prompt,
                            user=f"{user_prompt}\n\nINPUT_RECORD_JSON:\n{payload}",
                            max_tokens=int(max_tokens),
                            temperature=float(temperature),
                        )
                    st.session_state.setdefault("agent_runs", [])
                    st.session_state["agent_runs"].append({
                        "ts": now_iso(),
                        "mode": "single_record",
                        "dataset": ds_key,
                        "record_id": rid,
                        "agent_id": a["id"],
                        "model": model,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "output": out,
                    })
                    st.markdown("##### Output")
                    st.markdown(out)
                except Exception as e:
                    st.error(str(e))

    with tab_raw:
        st.code(json.dumps({k: v for k, v in rec.items() if not k.startswith("__")}, ensure_ascii=False, indent=2), language="json")


def relationship_explorer(results_by_ds: Dict[str, pd.DataFrame], shared: Dict[str, set], accent: str):
    st.markdown(f"### {t('relationship_explorer')}")
    if go is None:
        st.caption("Relationship graph unavailable (Plotly missing).")
        return

    nodes, edges = build_relationship_graph(results_by_ds, shared)
    if len(edges) == 0:
        st.info("No cross-dataset relationships detected in current results.")
        return

    # Create a simple radial layout: dataset nodes on a circle, entities outside
    ds_nodes = [n for n in nodes if n["kind"] == "dataset"]
    ent_nodes = [n for n in nodes if n["kind"] == "entity"]

    # Coordinates
    pos = {}
    R1 = 1.0
    for i, n in enumerate(ds_nodes):
        ang = 2 * math.pi * (i / max(1, len(ds_nodes)))
        pos[n["id"]] = (R1 * math.cos(ang), R1 * math.sin(ang))
    R2 = 2.0
    for j, n in enumerate(ent_nodes):
        ang = 2 * math.pi * (j / max(1, len(ent_nodes)))
        pos[n["id"]] = (R2 * math.cos(ang), R2 * math.sin(ang))

    # Edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    edge_w = []
    for s, d, etype, w in edges:
        x0, y0 = pos.get(s, (0, 0))
        x1, y1 = pos.get(d, (0, 0))
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(f"{etype} ({w})")
        edge_w.append(w)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color=CORAL),
        hoverinfo="none",
        mode="lines",
        opacity=0.75,
    )

    # Node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    for n in nodes:
        x, y = pos.get(n["id"], (0, 0))
        node_x.append(x)
        node_y.append(y)
        if n["kind"] == "dataset":
            node_text.append(n["label"])
            node_color.append(accent)
            node_size.append(22)
        else:
            # Entity
            label = n["label"]
            etype = n.get("etype", "")
            node_text.append(f"{etype}: {label}")
            node_color.append(CORAL)
            node_size.append(12)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n["label"] for n in nodes],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=1.2,
            line_color="#00000022",
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(t("why_linked")):
        st.markdown(
            f"- Coral edges/nodes represent entities appearing in **2+ datasets**.\n"
            f"- High-confidence links: exact matches on **UDI/DI** and **Recall Number**.\n"
            f"- Medium-confidence links: exact matches on **Manufacturer** and **Product Code**.\n"
            f"- Links are computed only within the current hit sets (not the full corpus)."
        )


def suggestion_chips(results_by_ds: Dict[str, pd.DataFrame]):
    # Build suggestion candidates from top entities in results
    ent_counts = {"manufacturer": {}, "product_code": {}, "udi_di": {}, "recall_number": {}}

    for ds_key, df in results_by_ds.items():
        if df is None or df.empty:
            continue
        for _, row in df.head(50).iterrows():
            r = row.to_dict()
            mfg = normalize_str(r.get("manufacturer_name") or r.get("firm_name") or r.get("applicant"))
            pc = normalize_str(r.get("product_code"))
            udi = normalize_str(r.get("udi_di") or r.get("primary_di"))
            rn = normalize_str(r.get("recall_number") or r.get("recall_number_link"))
            if mfg:
                ent_counts["manufacturer"][mfg] = ent_counts["manufacturer"].get(mfg, 0) + 1
            if pc:
                ent_counts["product_code"][pc] = ent_counts["product_code"].get(pc, 0) + 1
            if udi:
                ent_counts["udi_di"][udi] = ent_counts["udi_di"].get(udi, 0) + 1
            if rn:
                ent_counts["recall_number"][rn] = ent_counts["recall_number"].get(rn, 0) + 1

    def topk(d, k=2):
        return [x for x, _ in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]]

    suggestions = []
    for v in topk(ent_counts["manufacturer"], 2):
        suggestions.append(("Manufacturer", v))
    for v in topk(ent_counts["product_code"], 2):
        suggestions.append(("Product Code", v))
    for v in topk(ent_counts["udi_di"], 1):
        suggestions.append(("UDI/DI", v))
    for v in topk(ent_counts["recall_number"], 2):
        suggestions.append(("Recall #", v))

    st.markdown(f"### {t('suggestions')}")
    if not suggestions:
        st.caption("No suggestions yet. Run a search.")
        return

    cols = st.columns(min(5, len(suggestions)))
    for i, (k, v) in enumerate(suggestions[:10]):
        with cols[i % len(cols)]:
            if st.button(f"{k}: {v}", key=f"sugg_{k}_{i}"):
                st.session_state["query"] = v
                st.session_state["auto_search"] = True
                st.rerun()


# -----------------------------
# Prompt Notebook + Agent pipeline (editable outputs)
# -----------------------------
def prompt_notebook(results_by_ds: Dict[str, pd.DataFrame]):
    st.markdown(f"### {t('prompt_notebook')}")

    st.session_state.setdefault("workspace", [])
    pinned = st.session_state["workspace"]

    c1, c2 = st.columns([1.0, 1.2])
    with c1:
        st.markdown(f"<div class='wow-box'><div class='wow-small'>Pinned records</div><div style='font-size:1.4rem;font-weight:900'>{len(pinned)}</div></div>", unsafe_allow_html=True)
        if pinned:
            if st.button(t("clear"), key="clear_workspace"):
                st.session_state["workspace"] = []
                st.rerun()

    with c2:
        st.markdown("<div class='wow-box'><div class='wow-small'>Run agents on selected subset</div></div>", unsafe_allow_html=True)

    agents = st.session_state.get("agents", [])
    if not agents:
        st.warning("No agents available. Add agents.yaml or install PyYAML.")
        return

    run_mode = st.radio("Mode", options=[t("single_agent"), t("pipeline")], horizontal=True, key="nb_mode")

    # Inputs
    model = st.selectbox(t("model"), options=SUPPORTED_MODELS, index=max(0, SUPPORTED_MODELS.index("gpt-4o-mini")), key="nb_model")
    max_tokens = st.number_input(t("max_tokens"), min_value=256, max_value=200000, value=12000, step=256, key="nb_max_tokens")
    temperature = st.slider(t("temperature"), 0.0, 1.0, 0.2, 0.05, key="nb_temp")

    base_system = st.text_area(
        t("system_prompt"),
        value=(st.session_state.get("skill_md","") or "").strip(),
        height=120,
        key="nb_system",
        help="This is prepended to each agent's system prompt.",
    )
    user_prompt = st.text_area(
        t("user_prompt"),
        value="Analyze the current pinned/filtered records. Quote evidence from records. Mark missing info as Gap. Avoid fabrication.",
        height=120,
        key="nb_user_prompt",
    )

    # Build payload from pinned records if any, else from current filtered hits (top N)
    payload_records = []
    if pinned:
        payload_records = [p for p in pinned]
    else:
        # Use current hit sets
        for ds_key, df in results_by_ds.items():
            if df is None or df.empty:
                continue
            top = df.head(30).to_dict(orient="records")
            payload_records.append({"dataset": ds_key, "records": top})
    payload = json.dumps(payload_records, ensure_ascii=False, indent=2)

    if run_mode == t("single_agent"):
        agent_names = [f"{a['name']} ({a['id']})" for a in agents]
        idx = st.selectbox("Select agent", options=list(range(len(agents))), format_func=lambda i: agent_names[i], key="nb_agent_sel")
        a = agents[idx]
        sys = (base_system + "\n\n" + (a.get("system_prompt") or "")).strip()
        usr = (a.get("user_prompt") or "").strip()
        final_user = (usr + "\n\n" + user_prompt).strip()

        if st.button(t("run_agent"), key="nb_run_single"):
            try:
                with st.spinner("Running agent..."):
                    out = call_llm(
                        model=model,
                        provider=infer_provider_from_model(model),
                        system=sys,
                        user=f"{final_user}\n\nINPUT_SNAPSHOT_JSON:\n{payload}",
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                    )
                st.session_state.setdefault("agent_runs", [])
                st.session_state["agent_runs"].append({
                    "ts": now_iso(),
                    "mode": "prompt_notebook_single",
                    "agent_id": a["id"],
                    "model": model,
                    "system_prompt": sys,
                    "user_prompt": final_user,
                    "input_snapshot": payload_records,
                    "output": out,
                })
                st.markdown("#### Output")
                st.markdown(out)
            except Exception as e:
                st.error(str(e))

    else:
        # Pipeline: select multiple agents in order
        agent_ids = [a["id"] for a in agents]
        selected_ids = st.multiselect("Select agents (run in order)", options=agent_ids, default=agent_ids[:2], key="nb_pipeline_ids")
        pipeline_agents = [a for a in agents if a["id"] in selected_ids]
        if not pipeline_agents:
            st.info("Select at least one agent.")
            return

        st.session_state.setdefault("pipeline_state", {"steps": []})
        if st.button("Initialize pipeline", key="nb_pipeline_init"):
            st.session_state["pipeline_state"] = {
                "steps": [{"agent_id": a["id"], "agent_name": a["name"], "output": ""} for a in pipeline_agents],
                "input": payload,
            }

        ps = st.session_state.get("pipeline_state", {"steps": []})
        if ps.get("steps"):
            st.markdown("#### Pipeline Runner (editable outputs)")
            current_input = ps.get("input", payload)

            for i, step in enumerate(ps["steps"]):
                a = next((x for x in agents if x["id"] == step["agent_id"]), None)
                if not a:
                    continue

                st.markdown(f"##### Step {i+1}: {step['agent_name']}  ·  <span class='wow-small'>({step['agent_id']})</span>", unsafe_allow_html=True)
                sys = (base_system + "\n\n" + (a.get("system_prompt") or "")).strip()
                usr = ((a.get("user_prompt") or "") + "\n\n" + user_prompt).strip()

                c_run, c_edit = st.columns([0.22, 0.78])
                with c_run:
                    if st.button(f"Run step {i+1}", key=f"pipe_run_{i}"):
                        try:
                            with st.spinner(f"Running step {i+1}..."):
                                out = call_llm(
                                    model=model,
                                    provider=infer_provider_from_model(model),
                                    system=sys,
                                    user=f"{usr}\n\nINPUT:\n{current_input}",
                                    max_tokens=int(max_tokens),
                                    temperature=float(temperature),
                                )
                            ps["steps"][i]["output"] = out
                            st.session_state["pipeline_state"] = ps
                            st.success("Step completed.")
                        except Exception as e:
                            st.error(str(e))

                with c_edit:
                    out = st.text_area(
                        f"{t('output')} (edit allowed; becomes input to next step)",
                        value=ps["steps"][i].get("output", ""),
                        height=160,
                        key=f"pipe_out_{i}",
                    )
                    ps["steps"][i]["output"] = out
                    st.session_state["pipeline_state"] = ps

                # Next input becomes edited output if present, else remains current_input
                if ps["steps"][i]["output"].strip():
                    current_input = ps["steps"][i]["output"].strip()

            if st.button("Save pipeline run to audit log", key="pipe_save"):
                st.session_state.setdefault("agent_runs", [])
                st.session_state["agent_runs"].append({
                    "ts": now_iso(),
                    "mode": "pipeline",
                    "model": model,
                    "system_prompt_base": base_system,
                    "user_prompt": user_prompt,
                    "steps": ps["steps"],
                    "initial_input": ps.get("input", ""),
                    "final_output": current_input,
                })
                st.success("Saved.")


# -----------------------------
# Pages
# -----------------------------
def page_search_studio(accent: str):
    ds = st.session_state["datasets"]

    # Global command bar
    st.markdown("## " + t("page_search"))
    c1, c2, c3 = st.columns([2.2, 1.4, 0.8])

    with c1:
        q = st.text_input(t("query"), value=st.session_state.get("query", ""), key="query")
    with c2:
        with st.popover(t("search_settings")) if hasattr(st, "popover") else st.expander(t("search_settings"), expanded=True):
            exact = st.toggle(t("exact_match"), value=st.session_state.get("exact", False), key="exact")
            threshold = st.slider(t("fuzzy_threshold"), 0, 100, int(st.session_state.get("threshold", 60)), 1, key="threshold")
            limit = st.number_input(t("limit"), min_value=10, max_value=5000, value=int(st.session_state.get("limit", 200)), step=10, key="limit")
            weighting_label = st.selectbox(
                t("field_weighting"),
                options=[t("balanced"), t("id_boosted"), t("narrative_boosted")],
                index=0,
                key="weighting_label"
            )
            weighting = "balanced"
            if weighting_label == t("id_boosted"):
                weighting = "id"
            elif weighting_label == t("narrative_boosted"):
                weighting = "narrative"

            st.session_state["weighting"] = weighting

            st.markdown("**" + t("include_datasets") + "**")
            inc_510k = st.checkbox("510(k)", value=st.session_state.get("inc_510k", True), key="inc_510k")
            inc_recall = st.checkbox("Recall", value=st.session_state.get("inc_recall", True), key="inc_recall")
            inc_adr = st.checkbox("ADR", value=st.session_state.get("inc_adr", True), key="inc_adr")
            inc_gudid = st.checkbox("GUDID", value=st.session_state.get("inc_gudid", True), key="inc_gudid")

    with c3:
        run = st.button(t("search"), type="primary", use_container_width=True)

    # Auto-search (used by suggestion chips & pivot buttons)
    if st.session_state.get("auto_search"):
        run = True
        st.session_state["auto_search"] = False

    if not require_rapidfuzz():
        st.warning("rapidfuzz not installed. Fuzzy scoring will fall back to basic substring matching.")

    results_by_ds = {"510k": pd.DataFrame(), "recall": pd.DataFrame(), "adr": pd.DataFrame(), "gudid": pd.DataFrame()}

    if run and q.strip():
        with st.spinner("Searching across datasets..."):
            exact = bool(st.session_state.get("exact", False))
            threshold = int(st.session_state.get("threshold", 60))
            limit = int(st.session_state.get("limit", 200))
            weighting = st.session_state.get("weighting", "balanced")

            if st.session_state.get("inc_510k", True):
                results_by_ds["510k"] = search_dataset(ds["510k"], "510k", q, exact, threshold, limit, weighting)
            if st.session_state.get("inc_recall", True):
                results_by_ds["recall"] = search_dataset(ds["recall"], "recall", q, exact, threshold, limit, weighting)
            if st.session_state.get("inc_adr", True):
                results_by_ds["adr"] = search_dataset(ds["adr"], "adr", q, exact, threshold, limit, weighting)
            if st.session_state.get("inc_gudid", True):
                results_by_ds["gudid"] = search_dataset(ds["gudid"], "gudid", q, exact, threshold, limit, weighting)

        st.session_state["last_results"] = results_by_ds
        st.session_state["last_query"] = q

    # Render if exists
    results_by_ds = st.session_state.get("last_results", results_by_ds)
    shared = compute_shared_entities(results_by_ds)

    # KPI row
    kpi_row(results_by_ds, shared)

    # Four dataset panels + detail drawer
    left, right = st.columns([1.55, 1.0], gap="large")
    with left:
        p1, p2 = st.columns(2)
        with p1:
            dataset_panel("510(k)", "510k", results_by_ds.get("510k"), shared, accent)
        with p2:
            dataset_panel("Recall", "recall", results_by_ds.get("recall"), shared, accent)

        p3, p4 = st.columns(2)
        with p3:
            dataset_panel("ADR", "adr", results_by_ds.get("adr"), shared, accent)
        with p4:
            dataset_panel("GUDID", "gudid", results_by_ds.get("gudid"), shared, accent)

        suggestion_chips(results_by_ds)
        relationship_explorer(results_by_ds, shared, accent)

    with right:
        st.markdown("### Detail Drawer")
        detail_drawer(results_by_ds, shared, accent)
        st.markdown("<div class='wow-hr'></div>", unsafe_allow_html=True)
        prompt_notebook(results_by_ds)


def page_dataset_studio():
    st.markdown("## " + t("page_dataset"))

    st.markdown("<div class='wow-box wow-small'>Load the default datasets at startup, or upload new datasets (JSON) to replace them for this session.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.0, 1.2], gap="large")

    with c1:
        st.markdown("### Current dataset source")
        st.write("Source:", st.session_state.get("dataset_source", "default"))

        if st.button("Load defaultdataset.json", type="primary"):
            st.session_state["datasets"] = load_default_dataset_json(DEFAULT_DATASET_PATH)
            st.session_state["dataset_source"] = "default"
            st.success("Loaded defaults.")
            st.rerun()

        st.markdown("### Upload dataset package JSON")
        up = st.file_uploader("Upload JSON with keys {datasets:{510k,recall,adr,gudid}} or top-level keys", type=["json"])
        if up is not None:
            try:
                obj = json.loads(up.read().decode("utf-8"))
                ds = obj.get("datasets", obj)
                new_datasets = {}
                for k in ["510k", "recall", "adr", "gudid"]:
                    records = ds.get(k, [])
                    if isinstance(records, dict):
                        records = [records]
                    new_datasets[k] = pd.DataFrame(records)
                st.session_state["datasets"] = new_datasets
                st.session_state["dataset_source"] = "upload"
                st.success("Uploaded dataset package loaded into session.")
                st.rerun()
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    with c2:
        st.markdown("### Dataset preview")
        counts = dataset_counts()
        st.write(counts)
        ds = st.session_state["datasets"]
        for k in ["510k", "recall", "adr", "gudid"]:
            with st.expander(f"Preview: {k} ({counts[k]})"):
                df = ds.get(k, pd.DataFrame())
                if df is None or df.empty:
                    st.info("Empty.")
                else:
                    st.dataframe(df.head(50), use_container_width=True, hide_index=True)


def page_agent_studio():
    st.markdown("## " + t("page_agents"))

    if yaml is None:
        st.warning("PyYAML not installed. Upload/edit agents.yaml will be limited.")
    agents = st.session_state.get("agents", [])

    c1, c2 = st.columns([1.0, 1.0], gap="large")
    with c1:
        st.markdown("### agents.yaml")
        raw = safe_read_text(DEFAULT_AGENTS_PATH) if os.path.exists(DEFAULT_AGENTS_PATH) else ""
        content = st.text_area("Edit agents.yaml", value=raw, height=320)
        if st.button("Save agents.yaml"):
            if safe_write_text(DEFAULT_AGENTS_PATH, content):
                st.session_state["agents"] = load_agents_yaml(DEFAULT_AGENTS_PATH)
                st.success("Saved and reloaded agents.")
                st.rerun()
            else:
                st.error("Failed to save agents.yaml (read-only environment?).")

        up = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"])
        if up is not None:
            try:
                txt = up.read().decode("utf-8")
                if safe_write_text(DEFAULT_AGENTS_PATH, txt):
                    st.session_state["agents"] = load_agents_yaml(DEFAULT_AGENTS_PATH)
                    st.success("Uploaded and reloaded agents.")
                    st.rerun()
                else:
                    st.error("Failed to write agents.yaml.")
            except Exception as e:
                st.error(str(e))

    with c2:
        st.markdown("### SKILL.md")
        skill = st.text_area("Edit SKILL.md", value=st.session_state.get("skill_md", ""), height=320)
        if st.button("Save SKILL.md"):
            if safe_write_text(DEFAULT_SKILL_PATH, skill):
                st.session_state["skill_md"] = skill
                st.success("Saved SKILL.md.")
            else:
                st.error("Failed to save SKILL.md.")

        st.markdown("### Loaded agents")
        if not agents:
            st.info("No agents loaded.")
        else:
            df = pd.DataFrame(agents)
            show_cols = [c for c in ["id", "name", "description", "provider", "model", "temperature", "max_tokens"] if c in df.columns]
            st.dataframe(df[show_cols], use_container_width=True, hide_index=True)


def coral_highlight_keywords(md: str, keywords: List[str], color: str = CORAL) -> str:
    # naive highlight for display (HTML), keep original markdown visible
    # We'll only apply to plain text rendering; not altering underlying markdown in storage.
    if not md:
        return ""
    out = md
    for kw in sorted(set([k.strip() for k in keywords if k.strip()]), key=len, reverse=True):
        # word-boundary-ish
        out = re.sub(
            re.escape(kw),
            lambda m: f"<span style='color:{color};font-weight:800'>{m.group(0)}</span>",
            out,
            flags=re.IGNORECASE
        )
    return out


def page_ai_note_keeper():
    st.markdown("## " + t("page_notes"))

    st.session_state.setdefault("notes", {"raw": "", "organized": "", "prompt": "", "model": "gpt-4o-mini"})
    notes = st.session_state["notes"]

    c1, c2 = st.columns([1.05, 0.95], gap="large")

    with c1:
        st.markdown("### " + t("notes_input"))
        raw = st.text_area("Input", value=notes.get("raw", ""), height=260, key="notes_raw")
        notes["raw"] = raw

        st.markdown("### Controls")
        model = st.selectbox(t("model"), options=SUPPORTED_MODELS, index=max(0, SUPPORTED_MODELS.index(notes.get("model", "gpt-4o-mini"))), key="notes_model")
        notes["model"] = model
        max_tokens = st.number_input(t("max_tokens"), min_value=256, max_value=200000, value=int(st.session_state.get("notes_max_tokens", 12000)), step=256, key="notes_max_tokens")
        temperature = st.slider(t("temperature"), 0.0, 1.0, float(st.session_state.get("notes_temp", 0.2)), 0.05, key="notes_temp")

        prompt = st.text_area(
            "Transform prompt (editable)",
            value=notes.get("prompt", "Transform the notes into well-organized markdown with headings, bullets, and a 'Keywords' section. Highlight important keywords using coral color in the output by wrapping them in <span class='wow-coral'>keyword</span>. Quote exact phrases when possible. Mark missing info as Gap."),
            height=140,
            key="notes_prompt"
        )
        notes["prompt"] = prompt

        if st.button(t("transform"), type="primary"):
            try:
                system = (st.session_state.get("skill_md","") or "").strip()
                user = f"{prompt}\n\nNOTES_INPUT:\n{raw}"
                with st.spinner("Transforming notes..."):
                    out = call_llm(
                        model=model,
                        provider=infer_provider_from_model(model),
                        system=system,
                        user=user,
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                    )
                notes["organized"] = out
                st.success("Done.")
            except Exception as e:
                st.error(str(e))

        st.markdown("### " + t("ai_magics"))
        magic = st.selectbox("Choose a magic", options=[
            "AI Keywords (custom color highlighting)",
            "AI Summarize (structured summary)",
            "AI Action Items (tasks + owners + dates)",
            "AI Meeting Minutes (agenda/decisions/risks)",
            "AI Q&A (generate questions with evidence answers)",
        ], key="magic_sel")

        if magic.startswith("AI Keywords"):
            kw = st.text_input(t("keywords"), value="", key="magic_kw")
            color = st.color_picker(t("keyword_color"), value=CORAL, key="magic_color")
            if st.button("Apply keyword highlighting (local, no API)"):
                kws = [k.strip() for k in kw.split(",") if k.strip()]
                base = notes.get("organized") or notes.get("raw") or ""
                highlighted = coral_highlight_keywords(base, kws, color=color)
                st.session_state["notes_preview_html"] = highlighted
                st.success("Applied highlighting in preview panel.")
        else:
            if st.button("Run selected magic (uses model)"):
                try:
                    system = (st.session_state.get("skill_md","") or "").strip()
                    base = notes.get("organized") or notes.get("raw") or ""
                    instruction = {
                        "AI Summarize (structured summary)": "Create a structured summary with: Overview, Key points, Risks, Open questions, Evidence quotes.",
                        "AI Action Items (tasks + owners + dates)": "Extract action items. Output a markdown table with columns: Action, Owner (if unknown write Gap), Due date (if unknown write Gap), Evidence quote.",
                        "AI Meeting Minutes (agenda/decisions/risks)": "Convert into meeting minutes: Attendees (Gap if unknown), Agenda, Decisions, Action items, Risks, Next steps.",
                        "AI Q&A (generate questions with evidence answers)": "Generate 12 questions a reviewer should ask, and provide evidence-based answers strictly from the input; if missing, mark Gap.",
                    }.get(magic, "Analyze the input.")
                    user = f"{instruction}\n\nINPUT_TEXT_OR_MARKDOWN:\n{base}"
                    with st.spinner("Running magic..."):
                        out = call_llm(
                            model=model,
                            provider=infer_provider_from_model(model),
                            system=system,
                            user=user,
                            max_tokens=int(max_tokens),
                            temperature=float(temperature),
                        )
                    notes["organized"] = out
                    st.success("Magic completed and saved to Organized panel.")
                except Exception as e:
                    st.error(str(e))

    with c2:
        st.markdown("### Organized Markdown")
        organized = st.text_area("Organized (editable)", value=notes.get("organized", ""), height=360, key="notes_organized")
        notes["organized"] = organized

        st.markdown("### Preview (HTML highlight-friendly)")
        preview_html = st.session_state.get("notes_preview_html", "")
        if preview_html:
            st.markdown(f"<div class='wow-box'>{preview_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='wow-box wow-small'>Tip: Use AI Keywords magic to highlight keywords with a chosen color in this preview.</div>", unsafe_allow_html=True)

        st.markdown("### Keep prompt on note")
        st.caption("The current transform prompt is stored with this note in session_state. You can reuse it later or change models per run.")


def page_dashboard(accent: str):
    st.markdown("## " + t("page_dashboard"))
    st.markdown("<div class='wow-box'>Awesome interactive dashboard: use the Search Studio for deep exploration; this page summarizes your session.</div>", unsafe_allow_html=True)

    counts = dataset_counts()
    st.markdown("### Session status")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='wow-kpi'><div class='wow-small'>Datasets loaded</div><div style='font-size:1.3rem;font-weight:900'>{sum(counts.values())}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='wow-kpi'><div class='wow-small'>Pinned records</div><div style='font-size:1.3rem;font-weight:900'>{len(st.session_state.get('workspace', []))}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='wow-kpi'><div class='wow-small'>Agent runs (audit)</div><div style='font-size:1.3rem;font-weight:900'>{len(st.session_state.get('agent_runs', []))}</div></div>", unsafe_allow_html=True)
    with c4:
        last_q = st.session_state.get("last_query", "—")
        st.markdown(f"<div class='wow-kpi'><div class='wow-small'>Last query</div><div style='font-size:1.05rem;font-weight:900'>{last_q}</div></div>", unsafe_allow_html=True)

    st.markdown("### Recent agent runs (last 5)")
    runs = st.session_state.get("agent_runs", [])[-5:]
    if not runs:
        st.info("No runs yet.")
    else:
        for r in reversed(runs):
            st.markdown(
                f"<div class='wow-box wow-small'>"
                f"<b>{r.get('ts','')}</b> · mode: <span class='wow-accent'>{r.get('mode','')}</span> · model: {r.get('model','')} · agent: {r.get('agent_id','')}"
                f"</div>",
                unsafe_allow_html=True
            )


def page_factory():
    st.markdown("## " + t("page_factory"))
    st.markdown("<div class='wow-box wow-small'>Factory placeholder: keep existing features; extend here for batch workflows, document OCR, and export pipelines.</div>", unsafe_allow_html=True)


# -----------------------------
# Main App
# -----------------------------
def main():
    st.set_page_config(page_title="WOW Search Studio", layout="wide")

    # Session defaults
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("lang", "en")
    st.session_state.setdefault("style_id", PAINTER_STYLES[0]["id"])

    init_api_keys()
    init_datasets()
    init_agents()

    # Sidebar - WOW global controls
    st.sidebar.markdown(f"## {t('nav')}")
    page = st.sidebar.radio(
        " ",
        options=[t("page_dashboard"), t("page_search"), t("page_dataset"), t("page_agents"), t("page_notes"), t("page_factory")],
        index=1,
    )

    # Theme / language / style
    st.sidebar.markdown("### WOW UI")
    theme = st.sidebar.selectbox(t("theme"), options=[t("dark"), t("light")], index=0 if st.session_state["theme"] == "dark" else 1)
    st.session_state["theme"] = "dark" if theme == t("dark") else "light"

    lang = st.sidebar.selectbox(t("language"), options=["English", "繁體中文"], index=0 if st.session_state["lang"] == "en" else 1)
    st.session_state["lang"] = "en" if lang == "English" else "zh-TW"

    style_map = {s["id"]: s for s in PAINTER_STYLES}
    style_labels = [painter_label(s, st.session_state["lang"]) for s in PAINTER_STYLES]
    style_ids = [s["id"] for s in PAINTER_STYLES]
    style_idx = style_ids.index(st.session_state["style_id"]) if st.session_state["style_id"] in style_ids else 0

    chosen = st.sidebar.selectbox(t("style"), options=list(range(len(PAINTER_STYLES))), index=style_idx, format_func=lambda i: style_labels[i])
    st.session_state["style_id"] = PAINTER_STYLES[chosen]["id"]

    if st.sidebar.button(t("jackpot")):
        st.session_state["style_id"] = random.choice(PAINTER_STYLES)["id"]
        st.rerun()

    style = style_map.get(st.session_state["style_id"], PAINTER_STYLES[0])
    accent = style["accent"]

    # Apply WOW CSS
    apply_wow_css(st.session_state["theme"], accent)

    # API key UI (env-first rules)
    api_key_settings_ui()

    # Global status strip
    status_strip()
    st.markdown("<div class='wow-hr'></div>", unsafe_allow_html=True)

    # Route pages
    if page == t("page_search"):
        page_search_studio(accent)
    elif page == t("page_dataset"):
        page_dataset_studio()
    elif page == t("page_agents"):
        page_agent_studio()
    elif page == t("page_notes"):
        page_ai_note_keeper()
    elif page == t("page_factory"):
        page_factory()
    else:
        page_dashboard(accent)


if __name__ == "__main__":
    main()
```

---

## `defaultdataset.json`

```json
{
  "meta": {
    "name": "Default Mock FDA-like Datasets (510k/Recall/ADR/GUDID)",
    "version": "1.0",
    "generated_at": "2026-02-04T00:00:00Z",
    "notes": "Merged from the technical specification examples. Designed to validate cross-dataset linking via UDI/DI, recall number, manufacturer, and product_code."
  },
  "datasets": {
    "510k": [
      {
        "k_number": "K240123",
        "decision_date": "2024-06-18",
        "decision": "SESE",
        "device_name": "NovaPulse Infusion Pump",
        "applicant": "NovaMed Systems, Inc.",
        "manufacturer_name": "NovaMed Systems, Inc.",
        "product_code": "FRN",
        "regulation_number": "880.5725",
        "device_class": "II",
        "panel": "Anesthesiology",
        "review_advisory_committee": "—",
        "predicate_k_numbers": [
          "K221111",
          "K210987"
        ],
        "summary": "Battery-powered infusion pump with wireless connectivity and dose error reduction software."
      },
      {
        "k_number": "K221111",
        "decision_date": "2022-04-10",
        "decision": "SESE",
        "device_name": "NovaPulse Infusion Pump (Gen1)",
        "applicant": "NovaMed Systems, Inc.",
        "manufacturer_name": "NovaMed Systems, Inc.",
        "product_code": "FRN",
        "regulation_number": "880.5725",
        "device_class": "II",
        "panel": "Anesthesiology",
        "review_advisory_committee": "—",
        "predicate_k_numbers": [
          "K190222"
        ],
        "summary": "Earlier generation infusion pump; predicate chain used for substantial equivalence."
      },
      {
        "k_number": "K230456",
        "decision_date": "2023-11-02",
        "decision": "SESE",
        "device_name": "CardioSense ECG Patch",
        "applicant": "HeartArc Medical",
        "manufacturer_name": "HeartArc Medical",
        "product_code": "DXH",
        "regulation_number": "870.2800",
        "device_class": "II",
        "panel": "Cardiovascular",
        "review_advisory_committee": "—",
        "predicate_k_numbers": [
          "K201010"
        ],
        "summary": "Single-use ECG monitoring patch. Includes mobile app data review."
      },
      {
        "k_number": "K231777",
        "decision_date": "2023-12-14",
        "decision": "SESE",
        "device_name": "OrthoAlign Surgical Navigation",
        "applicant": "OrthoWorks Ltd.",
        "manufacturer_name": "OrthoWorks Ltd.",
        "product_code": "HDD",
        "regulation_number": "888.1100",
        "device_class": "II",
        "panel": "Orthopedic",
        "review_advisory_committee": "—",
        "predicate_k_numbers": [
          "K210333"
        ],
        "summary": "Navigation software used for orthopedic alignment; cybersecurity and software V&V referenced."
      },
      {
        "k_number": "K240123",
        "decision_date": "2024-02-14",
        "decision": "SESE",
        "device_name": "FlowPilot FP-2 Infusion Pump",
        "applicant": "NorthRiver Devices LLC",
        "manufacturer_name": "NorthRiver Devices LLC",
        "product_code": "FRN",
        "regulation_number": "880.5725",
        "device_class": "II",
        "panel": "Anesthesiology",
        "review_advisory_committee": "General Hospital",
        "predicate_k_numbers": [
          "K201111",
          "K210455"
        ],
        "summary": "Substantial equivalence based on intended use and technological characteristics; added battery monitoring enhancement."
      },
      {
        "k_number": "K240305",
        "decision_date": "2024-03-27",
        "decision": "SESE",
        "device_name": "StapleWave SW-45 Surgical Stapler",
        "applicant": "BlueWave Surgical Co.",
        "manufacturer_name": "BlueWave Surgical Co.",
        "product_code": "GAG",
        "regulation_number": "878.4750",
        "device_class": "II",
        "panel": "General & Plastic Surgery",
        "review_advisory_committee": "Surgery",
        "predicate_k_numbers": [
          "K193210"
        ],
        "summary": "SE determination with design changes in handle ergonomics; staple formation equivalent under bench testing."
      },
      {
        "k_number": "K240402",
        "decision_date": "2024-04-30",
        "decision": "SESE",
        "device_name": "StapleWave Cartridge SWC-45",
        "applicant": "BlueWave Surgical Co.",
        "manufacturer_name": "BlueWave Surgical Co.",
        "product_code": "GAB",
        "regulation_number": "878.4750",
        "device_class": "II",
        "panel": "General & Plastic Surgery",
        "review_advisory_committee": "Surgery",
        "predicate_k_numbers": [
          "K182909"
        ],
        "summary": "SE based on equivalent staple line performance; labeling updated for compatible stapler models."
      },
      {
        "k_number": "K240588",
        "decision_date": "2024-06-21",
        "decision": "SESE",
        "device_name": "RespiraScan Panel",
        "applicant": "Sunrise Diagnostics Ltd.",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "product_code": "OUI",
        "regulation_number": "866.3980",
        "device_class": "II",
        "panel": "Microbiology",
        "review_advisory_committee": "Microbiology",
        "predicate_k_numbers": [
          "K220900"
        ],
        "summary": "SE for multiplex respiratory panel; performance evaluated against comparator methods and reference materials."
      },
      {
        "k_number": "K230990",
        "decision_date": "2023-12-08",
        "decision": "SESE",
        "device_name": "RespiraScan Analyzer RSA-200",
        "applicant": "Sunrise Diagnostics Ltd.",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "product_code": "OHT",
        "regulation_number": "862.2570",
        "device_class": "II",
        "panel": "Clinical Chemistry",
        "review_advisory_committee": "Chemistry",
        "predicate_k_numbers": [
          "K210120"
        ],
        "summary": "SE based on equivalent amplification/detection workflow; software features documented in cybersecurity file."
      },
      {
        "k_number": "K241010",
        "decision_date": "2024-10-02",
        "decision": "SESE",
        "device_name": "OrchiFill Dermal Filler",
        "applicant": "Orchid Aesthetics Corp.",
        "manufacturer_name": "Orchid Aesthetics Corp.",
        "product_code": "LMH",
        "regulation_number": "878.3500",
        "device_class": "III",
        "panel": "General & Plastic Surgery",
        "review_advisory_committee": "Surgery",
        "predicate_k_numbers": [
          "K221777"
        ],
        "summary": "SE with emphasis on biocompatibility and clinical performance evidence provided by applicant."
      },
      {
        "k_number": "K240777",
        "decision_date": "2024-08-19",
        "decision": "SESE",
        "device_name": "HarborDrive HD-8 Powered Wheelchair",
        "applicant": "Harbor Mobility Systems",
        "manufacturer_name": "Harbor Mobility Systems",
        "product_code": "ITI",
        "regulation_number": "890.3860",
        "device_class": "II",
        "panel": "Physical Medicine",
        "review_advisory_committee": "Rehabilitation",
        "predicate_k_numbers": [
          "K200333"
        ],
        "summary": "SE based on equivalent mobility performance; updated control firmware and battery configuration."
      }
    ],
    "recall": [
      {
        "recall_number": "Z-1234-2024",
        "recall_class": "II",
        "event_date": "2024-08-05",
        "termination_date": "",
        "status": "Ongoing",
        "firm_name": "NovaMed Systems, Inc.",
        "manufacturer_name": "NovaMed Systems, Inc.",
        "product_description": "NovaPulse Infusion Pump",
        "product_code": "FRN",
        "code_info": "Lots: NP-2406A to NP-2407F",
        "reason_for_recall": "Potential battery failure leading to interruption of infusion.",
        "distribution_pattern": "US nationwide",
        "quantity_in_commerce": 3100,
        "country": "US",
        "state": "CA"
      },
      {
        "recall_number": "Z-0456-2023",
        "recall_class": "III",
        "event_date": "2023-03-20",
        "termination_date": "2023-10-01",
        "status": "Terminated",
        "firm_name": "HeartArc Medical",
        "manufacturer_name": "HeartArc Medical",
        "product_description": "CardioSense ECG Patch",
        "product_code": "DXH",
        "code_info": "Serials: CS-ECG-0001 to 2100",
        "reason_for_recall": "Labeling issue: contraindication not adequately displayed in IFU.",
        "distribution_pattern": "US limited distribution",
        "quantity_in_commerce": 850,
        "country": "US",
        "state": "MA"
      },
      {
        "recall_number": "Z-0999-2024",
        "recall_class": "I",
        "event_date": "2024-01-12",
        "termination_date": "",
        "status": "Ongoing",
        "firm_name": "OrthoWorks Ltd.",
        "manufacturer_name": "OrthoWorks Ltd.",
        "product_description": "OrthoAlign Surgical Navigation",
        "product_code": "HDD",
        "code_info": "Versions: 3.2.0 to 3.2.3",
        "reason_for_recall": "Software malfunction may display incorrect alignment guidance under rare conditions.",
        "distribution_pattern": "US and EU",
        "quantity_in_commerce": 120,
        "country": "US",
        "state": "NY"
      },
      {
        "recall_number": "Z-0421-2024",
        "recall_class": "I",
        "event_date": "2024-04-16",
        "termination_date": null,
        "status": "Ongoing",
        "firm_name": "BlueWave Surgical Co.",
        "manufacturer_name": "BlueWave Surgical Co.",
        "product_description": "Surgical stapler StapleWave SW-45 may misfire causing incomplete staple formation.",
        "product_code": "GAG",
        "code_info": "Lots 24A01-24A45",
        "reason_for_recall": "Potential for misfire leading to bleeding and prolonged surgery.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 18500,
        "country": "US",
        "state": "IL"
      },
      {
        "recall_number": "Z-0510-2024",
        "recall_class": "II",
        "event_date": "2024-05-28",
        "termination_date": "2024-11-02",
        "status": "Terminated",
        "firm_name": "NorthRiver Devices LLC",
        "manufacturer_name": "NorthRiver Devices LLC",
        "product_description": "Infusion set FlowPilot Set may leak at connector under high backpressure.",
        "product_code": "FPA",
        "code_info": "Lots 23F10-24F07",
        "reason_for_recall": "Leakage may result in under-infusion.",
        "distribution_pattern": "US and Canada",
        "quantity_in_commerce": 42000,
        "country": "US",
        "state": "MA"
      },
      {
        "recall_number": "Z-0777-2024",
        "recall_class": "I",
        "event_date": "2024-09-03",
        "termination_date": null,
        "status": "Ongoing",
        "firm_name": "Acme MedTech, Inc.",
        "manufacturer_name": "Acme MedTech, Inc.",
        "product_description": "PulseSure PS-3000 implantable pulse generator may experience premature battery depletion.",
        "product_code": "DXY",
        "code_info": "Serial range PS3K-2401XXXX to PS3K-2406XXXX",
        "reason_for_recall": "Battery depletion could lead to loss of therapy.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 3200,
        "country": "US",
        "state": "MD"
      },
      {
        "recall_number": "Z-0103-2025",
        "recall_class": "II",
        "event_date": "2025-01-15",
        "termination_date": null,
        "status": "Ongoing",
        "firm_name": "Sunrise Diagnostics Ltd.",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "product_description": "RespiraScan Panel may yield false negative results under specific reagent storage conditions.",
        "product_code": "OUI",
        "code_info": "Kits expiring 2025-03 to 2025-06",
        "reason_for_recall": "False negatives may delay appropriate treatment.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 12000,
        "country": "US",
        "state": "WA"
      },
      {
        "recall_number": "Z-0218-2024",
        "recall_class": "III",
        "event_date": "2024-02-22",
        "termination_date": "2024-04-12",
        "status": "Terminated",
        "firm_name": "Harbor Mobility Systems",
        "manufacturer_name": "Harbor Mobility Systems",
        "product_description": "HarborDrive Battery HB-24 label may list incorrect charging current specification.",
        "product_code": "KJP",
        "code_info": "Label revision L-02",
        "reason_for_recall": "Labeling correction; no reported injuries.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 7800,
        "country": "US",
        "state": "OR"
      },
      {
        "recall_number": "Z-0934-2024",
        "recall_class": "II",
        "event_date": "2024-12-06",
        "termination_date": null,
        "status": "Ongoing",
        "firm_name": "Orchid Aesthetics Corp.",
        "manufacturer_name": "Orchid Aesthetics Corp.",
        "product_description": "OrchiFill Kit syringes may contain natural rubber latex in plunger seal not stated on IFU.",
        "product_code": "FMF",
        "code_info": "Lots OK10-24K01 to OK10-24K22",
        "reason_for_recall": "Potential allergic reactions in latex-sensitive patients.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 9500,
        "country": "US",
        "state": "CA"
      },
      {
        "recall_number": "Z-0602-2023",
        "recall_class": "II",
        "event_date": "2023-10-11",
        "termination_date": "2024-01-08",
        "status": "Terminated",
        "firm_name": "Sunrise Diagnostics Ltd.",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "product_description": "RespiraScan Analyzer RSA-200 may display intermittent error codes due to firmware issue.",
        "product_code": "OHT",
        "code_info": "Firmware v1.0.0 to v1.0.2",
        "reason_for_recall": "Instrument interruptions could delay results reporting.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 640,
        "country": "US",
        "state": "WA"
      },
      {
        "recall_number": "Z-0333-2024",
        "recall_class": "I",
        "event_date": "2024-03-05",
        "termination_date": null,
        "status": "Ongoing",
        "firm_name": "Acme MedTech, Inc.",
        "manufacturer_name": "Acme MedTech, Inc.",
        "product_description": "PulseSure Lead PSL-20 may fracture under repeated flexion.",
        "product_code": "DTB",
        "code_info": "Lots PSL20-23H01 to PSL20-24A12",
        "reason_for_recall": "Lead fracture may result in ineffective pacing.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 8700,
        "country": "US",
        "state": "MD"
      },
      {
        "recall_number": "Z-0189-2024",
        "recall_class": "II",
        "event_date": "2024-01-30",
        "termination_date": "2024-07-19",
        "status": "Terminated",
        "firm_name": "NorthRiver Devices LLC",
        "manufacturer_name": "NorthRiver Devices LLC",
        "product_description": "FlowPilot FP-2 infusion pump battery may discharge faster than expected.",
        "product_code": "FRN",
        "code_info": "Batches FP2-BAT-23Q4",
        "reason_for_recall": "Unexpected shutdown could interrupt infusion.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 5100,
        "country": "US",
        "state": "MA"
      },
      {
        "recall_number": "Z-0144-2025",
        "recall_class": "III",
        "event_date": "2025-01-06",
        "termination_date": "2025-01-28",
        "status": "Terminated",
        "firm_name": "Harbor Mobility Systems",
        "manufacturer_name": "Harbor Mobility Systems",
        "product_description": "HarborDrive HD-8 user manual missing a troubleshooting section in certain print runs.",
        "product_code": "ITI",
        "code_info": "Manual rev 1.1",
        "reason_for_recall": "Documentation correction.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 2100,
        "country": "US",
        "state": "OR"
      }
    ],
    "adr": [
      {
        "adverse_event_id": "ADR-2024-000778",
        "report_date": "2024-08-22",
        "event_type": "Malfunction",
        "patient_outcome": "No injury",
        "device_problem": "Power/Battery Problem",
        "manufacturer_name": "NovaMed Systems, Inc.",
        "brand_name": "NovaPulse",
        "product_code": "FRN",
        "device_class": "II",
        "udi_di": "00810000012345",
        "recall_number_link": "Z-1234-2024",
        "narrative": "Unit shut down unexpectedly during infusion. Battery indicator showed adequate charge prior to event."
      },
      {
        "adverse_event_id": "ADR-2024-000812",
        "report_date": "2024-09-02",
        "event_type": "Serious Injury",
        "patient_outcome": "Hospitalization",
        "device_problem": "Dose Delivery Issue",
        "manufacturer_name": "NovaMed Systems, Inc.",
        "brand_name": "NovaPulse",
        "product_code": "FRN",
        "device_class": "II",
        "udi_di": "00810000012345",
        "recall_number_link": "Z-1234-2024",
        "narrative": "Reported interruption of therapy; investigation ongoing; possible power failure coincident with alarm."
      },
      {
        "adverse_event_id": "ADR-2023-000155",
        "report_date": "2023-04-09",
        "event_type": "Malfunction",
        "patient_outcome": "No injury",
        "device_problem": "Labeling/IFU Problem",
        "manufacturer_name": "HeartArc Medical",
        "brand_name": "CardioSense",
        "product_code": "DXH",
        "device_class": "II",
        "udi_di": "00820000077777",
        "recall_number_link": "Z-0456-2023",
        "narrative": "User reported missing contraindication statement in printed IFU within shipped kit."
      },
      {
        "adverse_event_id": "ADR-2024-000301",
        "report_date": "2024-02-02",
        "event_type": "Malfunction",
        "patient_outcome": "No injury",
        "device_problem": "Software Problem",
        "manufacturer_name": "OrthoWorks Ltd.",
        "brand_name": "OrthoAlign",
        "product_code": "HDD",
        "device_class": "II",
        "udi_di": "00990000111111",
        "recall_number_link": "Z-0999-2024",
        "narrative": "Navigation display lag observed; device restarted; procedure continued with alternative guidance."
      },
      {
        "adverse_event_id": "MDR-2024-000001",
        "report_date": "2024-04-22",
        "event_type": "Malfunction",
        "patient_outcome": "Serious Injury",
        "device_problem": "Misfire / Failure to staple",
        "manufacturer_name": "BlueWave Surgical Co.",
        "brand_name": "StapleWave",
        "product_code": "GAG",
        "device_class": "II",
        "udi_di": "00666099000034",
        "recall_number_link": "Z-0421-2024",
        "narrative": "During surgery, stapler misfired; surgeon used alternative device. Patient experienced bleeding requiring additional intervention."
      },
      {
        "adverse_event_id": "MDR-2024-000002",
        "report_date": "2024-03-18",
        "event_type": "Malfunction",
        "patient_outcome": "Hospitalization",
        "device_problem": "Lead fracture",
        "manufacturer_name": "Acme MedTech, Inc.",
        "brand_name": "PulseSure Lead",
        "product_code": "DTB",
        "device_class": "III",
        "udi_di": "00812345000029",
        "recall_number_link": "Z-0333-2024",
        "narrative": "Loss of capture suspected; imaging indicated possible lead integrity issue. Patient hospitalized for revision planning."
      },
      {
        "adverse_event_id": "MDR-2024-000003",
        "report_date": "2024-09-19",
        "event_type": "Malfunction",
        "patient_outcome": "Death",
        "device_problem": "Premature battery depletion",
        "manufacturer_name": "Acme MedTech, Inc.",
        "brand_name": "PulseSure",
        "product_code": "DXY",
        "device_class": "III",
        "udi_di": "00812345000012",
        "recall_number_link": "Z-0777-2024",
        "narrative": "Device reached ERI unexpectedly; therapy interruption suspected. Patient later died; causality not confirmed in report."
      },
      {
        "adverse_event_id": "MDR-2024-000004",
        "report_date": "2024-02-05",
        "event_type": "Malfunction",
        "patient_outcome": "No Injury",
        "device_problem": "Unexpected shutdown",
        "manufacturer_name": "NorthRiver Devices LLC",
        "brand_name": "FlowPilot",
        "product_code": "FRN",
        "device_class": "II",
        "udi_di": "00777001000018",
        "recall_number_link": "Z-0189-2024",
        "narrative": "Pump shut down during routine use; alarm sounded. No reported injury; infusion was restarted after battery replacement."
      },
      {
        "adverse_event_id": "MDR-2024-000005",
        "report_date": "2024-06-04",
        "event_type": "Malfunction",
        "patient_outcome": "No Injury",
        "device_problem": "Leakage",
        "manufacturer_name": "NorthRiver Devices LLC",
        "brand_name": "FlowPilot Set",
        "product_code": "FPA",
        "device_class": "II",
        "udi_di": "00777001000025",
        "recall_number_link": "Z-0510-2024",
        "narrative": "Connector leakage observed; user replaced infusion set. No injury reported."
      },
      {
        "adverse_event_id": "MDR-2025-000006",
        "report_date": "2025-01-22",
        "event_type": "Injury",
        "patient_outcome": "Serious Injury",
        "device_problem": "False negative test result",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "brand_name": "RespiraScan Panel",
        "product_code": "OUI",
        "device_class": "II",
        "udi_di": "00999111000057",
        "recall_number_link": "Z-0103-2025",
        "narrative": "Patient initially tested negative; later confirmed positive. Delay in treatment alleged. Investigation pending."
      },
      {
        "adverse_event_id": "MDR-2023-000007",
        "report_date": "2023-11-03",
        "event_type": "Malfunction",
        "patient_outcome": "No Injury",
        "device_problem": "Software error code",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "brand_name": "RespiraScan Analyzer",
        "product_code": "OHT",
        "device_class": "II",
        "udi_di": "00999111000064",
        "recall_number_link": "Z-0602-2023",
        "narrative": "Analyzer displayed intermittent error; test run restarted. No patient impact reported."
      },
      {
        "adverse_event_id": "MDR-2024-000008",
        "report_date": "2024-12-20",
        "event_type": "Injury",
        "patient_outcome": "Other Serious",
        "device_problem": "Allergic reaction",
        "manufacturer_name": "Orchid Aesthetics Corp.",
        "brand_name": "OrchiFill Kit",
        "product_code": "FMF",
        "device_class": "II",
        "udi_di": "00444988000101",
        "recall_number_link": "Z-0934-2024",
        "narrative": "Patient reported allergic reaction consistent with latex sensitivity; IFU did not indicate latex presence per reporter."
      },
      {
        "adverse_event_id": "MDR-2024-000009",
        "report_date": "2024-08-26",
        "event_type": "Malfunction",
        "patient_outcome": "No Injury",
        "device_problem": "Battery overheating",
        "manufacturer_name": "Harbor Mobility Systems",
        "brand_name": "HarborDrive Battery",
        "product_code": "KJP",
        "device_class": "II",
        "udi_di": "00555123000088",
        "recall_number_link": null,
        "narrative": "Battery pack became warm during charging; user discontinued charging. No injury; device inspected by service center."
      },
      {
        "adverse_event_id": "MDR-2025-000010",
        "report_date": "2025-01-09",
        "event_type": "Malfunction",
        "patient_outcome": "No Injury",
        "device_problem": "User manual missing troubleshooting section",
        "manufacturer_name": "Harbor Mobility Systems",
        "brand_name": "HarborDrive",
        "product_code": "ITI",
        "device_class": "II",
        "udi_di": "00555123000071",
        "recall_number_link": "Z-0144-2025",
        "narrative": "User reported inability to resolve warning indicator due to missing troubleshooting content in manual; no injury."
      }
    ],
    "gudid": [
      {
        "primary_di": "00810000012345",
        "udi_di": "00810000012345",
        "device_description": "NovaPulse Infusion Pump, wireless-enabled, battery-powered.",
        "device_class": "II",
        "manufacturer_name": "NovaMed Systems, Inc.",
        "brand_name": "NovaPulse",
        "product_code": "FRN",
        "gmdn_term": "Infusion pump",
        "mri_safety": "MR Unsafe",
        "sterile": false,
        "single_use": false,
        "implantable": false,
        "contains_nrl": false,
        "version_or_model_number": "NP-2",
        "catalog_number": "NP2-BASE",
        "record_status": "Published",
        "publish_date": "2024-05-30",
        "company_contact_email": "support@novamed.example",
        "company_contact_phone": "+1-555-0100",
        "company_state": "CA",
        "company_country": "US"
      },
      {
        "primary_di": "00820000077777",
        "udi_di": "00820000077777",
        "device_description": "CardioSense ECG Patch, single-use wearable sensor.",
        "device_class": "II",
        "manufacturer_name": "HeartArc Medical",
        "brand_name": "CardioSense",
        "product_code": "DXH",
        "gmdn_term": "Electrocardiograph monitor",
        "mri_safety": "MR Conditional",
        "sterile": true,
        "single_use": true,
        "implantable": false,
        "contains_nrl": false,
        "version_or_model_number": "CS-1",
        "catalog_number": "CS-ECG-PATCH",
        "record_status": "Published",
        "publish_date": "2023-09-01",
        "company_contact_email": "qa@heartarc.example",
        "company_contact_phone": "+1-555-0200",
        "company_state": "MA",
        "company_country": "US"
      },
      {
        "primary_di": "00990000111111",
        "udi_di": "00990000111111",
        "device_description": "OrthoAlign Surgical Navigation System (software and workstation).",
        "device_class": "II",
        "manufacturer_name": "OrthoWorks Ltd.",
        "brand_name": "OrthoAlign",
        "product_code": "HDD",
        "gmdn_term": "Surgical navigation system",
        "mri_safety": "Not evaluated",
        "sterile": false,
        "single_use": false,
        "implantable": false,
        "contains_nrl": false,
        "version_or_model_number": "OA-3",
        "catalog_number": "OA3-WS",
        "record_status": "Published",
        "publish_date": "2024-01-05",
        "company_contact_email": "security@orthoworks.example",
        "company_contact_phone": "+44-20-0000-0000",
        "company_state": "",
        "company_country": "UK"
      },
      {
        "primary_di": "00812345000012",
        "udi_di": "00812345000012",
        "device_description": "Implantable cardiac pulse generator",
        "device_class": "III",
        "manufacturer_name": "Acme MedTech, Inc.",
        "brand_name": "PulseSure",
        "product_code": "DXY",
        "gmdn_term": "Cardiac pulse generator, implantable",
        "mri_safety": "MR Conditional",
        "sterile": true,
        "single_use": false,
        "implantable": true,
        "contains_nrl": false,
        "version_or_model_number": "PS-3000",
        "catalog_number": "AC-PS3000",
        "record_status": "Published",
        "publish_date": "2024-03-14",
        "company_contact_email": "regulatory@acmemedtech.example",
        "company_contact_phone": "+1-301-555-0101",
        "company_state": "MD",
        "company_country": "US"
      },
      {
        "primary_di": "00812345000029",
        "udi_di": "00812345000029",
        "device_description": "Cardiac lead, pacing, silicone insulated",
        "device_class": "III",
        "manufacturer_name": "Acme MedTech, Inc.",
        "brand_name": "PulseSure Lead",
        "product_code": "DTB",
        "gmdn_term": "Cardiac pacing lead",
        "mri_safety": "MR Conditional",
        "sterile": true,
        "single_use": true,
        "implantable": true,
        "contains_nrl": false,
        "version_or_model_number": "PSL-20",
        "catalog_number": "AC-PSL20",
        "record_status": "Published",
        "publish_date": "2024-05-01",
        "company_contact_email": "regulatory@acmemedtech.example",
        "company_contact_phone": "+1-301-555-0101",
        "company_state": "MD",
        "company_country": "US"
      },
      {
        "primary_di": "00777001000018",
        "udi_di": "00777001000018",
        "device_description": "Infusion pump, programmable ambulatory",
        "device_class": "II",
        "manufacturer_name": "NorthRiver Devices LLC",
        "brand_name": "FlowPilot",
        "product_code": "FRN",
        "gmdn_term": "Infusion pump, ambulatory",
        "mri_safety": "Not Evaluated",
        "sterile": false,
        "single_use": false,
        "implantable": false,
        "contains_nrl": false,
        "version_or_model_number": "FP-2",
        "catalog_number": "NR-FP2",
        "record_status": "Published",
        "publish_date": "2023-11-20",
        "company_contact_email": "qa@northriver.example",
        "company_contact_phone": "+1-617-555-0123",
        "company_state": "MA",
        "company_country": "US"
      },
      {
        "primary_di": "00777001000025",
        "udi_di": "00777001000025",
        "device_description": "Infusion set, sterile, single-use",
        "device_class": "II",
        "manufacturer_name": "NorthRiver Devices LLC",
        "brand_name": "FlowPilot Set",
        "product_code": "FPA",
        "gmdn_term": "Infusion set",
        "mri_safety": "Not Evaluated",
        "sterile": true,
        "single_use": true,
        "implantable": false,
        "contains_nrl": true,
        "version_or_model_number": "FS-1",
        "catalog_number": "NR-FS1",
        "record_status": "Published",
        "publish_date": "2024-01-18",
        "company_contact_email": "qa@northriver.example",
        "company_contact_phone": "+1-617-555-0123",
        "company_state": "MA",
        "company_country": "US"
      },
      {
        "primary_di": "00666099000034",
        "udi_di": "00666099000034",
        "device_description": "Surgical stapler, single-use",
        "device_class": "II",
        "manufacturer_name": "BlueWave Surgical Co.",
        "brand_name": "StapleWave",
        "product_code": "GAG",
        "gmdn_term": "Surgical stapler, disposable",
        "mri_safety": "Not Evaluated",
        "sterile": true,
        "single_use": true,
        "implantable": false,
        "contains_nrl": false,
        "version_or_model_number": "SW-45",
        "catalog_number": "BW-SW45",
        "record_status": "Published",
        "publish_date": "2024-06-09",
        "company_contact_email": "complaints@bluewavesurg.example",
        "company_contact_phone": "+1-312-555-0190",
        "company_state": "IL",
        "company_country": "US"
      },
      {
        "primary_di": "00666099000041",
        "udi_di": "00666099000041",
        "device_description": "Staple cartridge, sterile, single-use",
        "device_class": "II",
        "manufacturer_name": "BlueWave Surgical Co.",
        "brand_name": "StapleWave Cartridge",
        "product_code": "GAB",
        "gmdn_term": "Staple cartridge",
        "mri_safety": "Not Evaluated",
        "sterile": true,
        "single_use": true,
        "implantable": false,
        "contains_nrl": false,
        "version_or_model_number": "SWC-45",
        "catalog_number": "BW-SWC45",
        "record_status": "Published",
        "publish_date": "2024-06-09",
        "company_contact_email": "complaints@bluewavesurg.example",
        "company_contact_phone": "+1-312-555-0190",
        "company_state": "IL",
        "company_country": "US"
      },
      {
        "primary_di": "00999111000057",
        "udi_di": "00999111000057",
        "device_description": "In vitro diagnostic test, respiratory panel",
        "device_class": "II",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "brand_name": "RespiraScan Panel",
        "product_code": "OUI",
        "gmdn_term": "Respiratory pathogen assay",
        "mri_safety": "Not Applicable",
        "sterile": false,
        "single_use": true,
        "implantable": false,
        "contains_nrl": false,
        "version_or_model_number": "RSP-1",
        "catalog_number": "SD-RSP1",
        "record_status": "Published",
        "publish_date": "2024-02-07",
        "company_contact_email": "support@sunrisedx.example",
        "company_contact_phone": "+1-206-555-0142",
        "company_state": "WA",
        "company_country": "US"
      },
      {
        "primary_di": "00999111000064",
        "udi_di": "00999111000064",
        "device_description": "In vitro diagnostic instrument, PCR analyzer",
        "device_class": "II",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "brand_name": "RespiraScan Analyzer",
        "product_code": "OHT",
        "gmdn_term": "PCR analyzer",
        "mri_safety": "Not Applicable",
        "sterile": false,
        "single_use": false,
        "implantable": false,
        "contains_nrl": false,
        "version_or_model_number": "RSA-200",
        "catalog_number": "SD-RSA200",
        "record_status": "Published",
        "publish_date": "2023-09-28",
        "company_contact_email": "support@sunrisedx.example",
        "company_contact_phone": "+1-206-555-0142",
        "company_state": "WA",
        "company_country": "US"
      },
      {
        "primary_di": "00555123000071",
        "udi_di": "00555123000071",
        "device_description": "Wheelchair, powered",
        "device_class": "II",
        "manufacturer_name": "Harbor Mobility Systems",
        "brand_name": "HarborDrive",
        "product_code": "ITI",
        "gmdn_term": "Wheelchair, powered",
        "mri_safety": "Not Evaluated",
        "sterile": false,
        "single_use": false,
        "implantable": false,
        "contains_nrl": false,
        "version_or_model_number": "HD-8",
        "catalog_number": "HM-HD8",
        "record_status": "Published",
        "publish_date": "2023-06-03",
        "company_contact_email": "service@harbormobility.example",
        "company_contact_phone": "+1-503-555-0166",
        "company_state": "OR",
        "company_country": "US"
      },
      {
        "primary_di": "00555123000088",
        "udi_di": "00555123000088",
        "device_description": "Wheelchair battery pack",
        "device_class": "II",
        "manufacturer_name": "Harbor Mobility Systems",
        "brand_name": "HarborDrive Battery",
        "product_code": "KJP",
        "gmdn_term": "Battery, rechargeable",
        "mri_safety": "Not Evaluated",
        "sterile": false,
        "single_use": false,
        "implantable": false,
        "contains_nrl": false,
        "version_or_model_number": "HB-24",
        "catalog_number": "HM-HB24",
        "record_status": "Published",
        "publish_date": "2024-08-12",
        "company_contact_email": "service@harbormobility.example",
        "company_contact_phone": "+1-503-555-0166",
        "company_state": "OR",
        "company_country": "US"
      },
      {
        "primary_di": "00444988000095",
        "udi_di": "00444988000095",
        "device_description": "Dermal filler, hyaluronic acid",
        "device_class": "III",
        "manufacturer_name": "Orchid Aesthetics Corp.",
        "brand_name": "OrchiFill",
        "product_code": "LMH",
        "gmdn_term": "Dermal filler",
        "mri_safety": "Not Applicable",
        "sterile": true,
        "single_use": true,
        "implantable": true,
        "contains_nrl": false,
        "version_or_model_number": "OF-HA2",
        "catalog_number": "OA-OFHA2",
        "record_status": "Published",
        "publish_date": "2024-10-05",
        "company_contact_email": "safety@orchidaesthetics.example",
        "company_contact_phone": "+1-213-555-0188",
        "company_state": "CA",
        "company_country": "US"
      },
      {
        "primary_di": "00444988000101",
        "udi_di": "00444988000101",
        "device_description": "Syringe kit for dermal filler, sterile",
        "device_class": "II",
        "manufacturer_name": "Orchid Aesthetics Corp.",
        "brand_name": "OrchiFill Kit",
        "product_code": "FMF",
        "gmdn_term": "Syringe, single-use",
        "mri_safety": "Not Applicable",
        "sterile": true,
        "single_use": true,
        "implantable": false,
        "contains_nrl": true,
        "version_or_model_number": "OK-10",
        "catalog_number": "OA-OK10",
        "record_status": "Published",
        "publish_date": "2024-10-05",
        "company_contact_email": "safety@orchidaesthetics.example",
        "company_contact_phone": "+1-213-555-0188",
        "company_state": "CA",
        "company_country": "US"
      }
    ]
  }
}
