import streamlit as st
import requests
# --- UTF-8 safety + smart-quote cleanup ---
import os, sys, unicodedata

# ensure process/stdout use UTF-8 (helps on Windows)
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SMART_MAP = {
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00A0": " ",
}

def to_utf8_clean(s: str) -> str:
    if not s:
        return ""
    # normalize composed chars, replace smart punctuation, ensure utf-8 roundtrip
    s = unicodedata.normalize("NFC", str(s))
    for k, v in SMART_MAP.items():
        s = s.replace(k, v)
    return s.encode("utf-8", "ignore").decode("utf-8")


st.set_page_config(page_title="SEO Optimizer", layout="centered")
st.title("SEO Optimization Panel")

# your n8n webhook URL
N8N_WEBHOOK_URL = "https://your-n8n-url/webhook/seo-optimize"  # <-- change this

category = st.selectbox("Category", ["NEET", "JEE", "GOV"])
task = st.text_input("Task", value="SEO Optimization")

if st.button("Run"):
    try:
        payload = {
            "category": category,
            "task": task
        }
        resp = requests.post(N8N_WEBHOOK_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

        output_text = data.get("output", "No output from n8n.")
        score = data.get("listing_score", 0)
    except Exception as e:
        output_text = f"Error: {e}"
        score = 0
else:
    output_text = ""
    score = 0

st.text_area("Output", value=output_text, height=200)
st.number_input("Listing Score", value=score, step=1)
