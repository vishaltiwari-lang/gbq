# filename: app.py
# Shakti 1.0 — PW Amazon SEO Optimizer (L1 OpenAI + optional L2: OpenAI/Gemini/Claude)
# HTML report only, taller table view, styled credentials

import io, re, json, zipfile
from datetime import datetime
from xml.etree import ElementTree as ET

import streamlit as st
import pandas as pd

# ---- Primary (OpenAI) ----
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---- Secondary engines ----
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import anthropic
except Exception:
    anthropic = None

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Shakti 1.0 — PW SEO Optimizer", layout="centered")

# Inline keys optional; you can still enter via UI
OPENAI_API_KEY_INLINE = ""
GEMINI_API_KEY_INLINE = ""
ANTHROPIC_API_KEY_INLINE = ""

OPENAI_MODELS = {
    "GPT-4.1 Mini (fast/cost-effective)": "gpt-4.1-mini",
    "GPT-4.1": "gpt-4.1",
    "GPT-4o": "gpt-4o",
    "GPT-5 (if enabled)": "gpt-5",
}
GEMINI_MODELS = {
    "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 1.5 Flash": "gemini-1.5-flash",
}
CLAUDE_MODELS = {
    "Claude 3.5 Sonnet (2024-10-22)": "claude-3.5-sonnet-20241022",
    "Claude 3.5 Haiku (2024-10-22)": "claude-3.5-haiku-20241022",
    "Claude 3 Opus (2024-02-29)": "claude-3-opus-20240229",
}

# -------------- HELPERS --------------
def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            xml = z.read("word/document.xml")
        root = ET.fromstring(xml)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        texts = [t.text or "" for t in root.iterfind(".//w:t", ns)]
        return " ".join("".join(texts).split()).strip()
    except Exception as e:
        return f"[DOCX READ ERROR] {e}"

def coerce_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return None

def ensure_listing_shape(obj: dict):
    return {
        "new_title": (obj.get("new_title") or "").strip(),
        "new_description": (obj.get("new_description") or "").strip(),
        "keywords_short": obj.get("keywords_short") or obj.get("short_tail_keywords") or [],
        "keywords_mid": obj.get("keywords_mid") or obj.get("mid_tail_keywords") or [],
        "keywords_long": obj.get("keywords_long") or obj.get("long_tail_keywords") or [],
    }

def html_report_bytes(app_meta: dict, inputs: dict, draft: dict, final_: dict):
    style = """
    <style>
      body { font-family: Arial, sans-serif; line-height: 1.45; padding: 24px; color: #1f2937; }
      .title { font-size: 24px; font-weight: 800; color: #4f46e5; margin: 0 0 4px; }
      .sub { color:#6b7280; margin: 0 0 16px; }
      .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin: 16px 0; }
      h3 { margin: 16px 0 8px; color:#111827; }
      table { border-collapse: collapse; width: 100%; }
      td, th { border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; text-align: left; }
      .mono { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background:#f9fafb; padding:8px; border-radius:8px;}
      .badge { display:inline-block; padding:4px 10px; border-radius:999px; background:#eef2ff; border:1px solid #c7d2fe; color:#3730a3; font-weight:600; margin-right:6px;}
    </style>
    """
    creds = f"""
    <div class="card">
      <span class="badge">Shakti 1.0</span>
      <span class="badge">PW In-house</span>
      <span class="badge">Author: Vishal Tiwari (pw17633)</span>
      <span class="badge">Project Head: Kumar Sanskar</span>
    </div>
    """
    def kv(label, value):
        return f"<tr><td><b>{label}</b></td><td>{value}</td></tr>"
    def jo(lst): return ", ".join(lst or [])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Shakti 1.0 Report</title>{style}</head>
<body>
  <div class="title">Shakti 1.0 — PW SEO Optimizer</div>
  <div class="sub">PW in-house SEO optimization app made by Vishal Tiwari (pw17633), Project Head: Kumar Sanskar</div>
  {creds}

  <div class="card">
    <h3>Inputs</h3>
    <table>
      {kv("Previous Title", inputs.get('prev_title') or '—')}
      {kv("Previous Description", f"<div class='mono'>{(inputs.get('prev_desc') or '—')}</div>")}
      {kv("Product Link", inputs.get('product_link') or '—')}
    </table>
  </div>

  <div class="card">
    <h3>Draft (L1 — OpenAI)</h3>
    <table>
      {kv("Title", draft.get('new_title') or '—')}
      {kv("Description (HTML)", f"<div class='mono'>{(draft.get('new_description') or '—').replace('<','&lt;').replace('>','&gt;')}</div>")}
      {kv("Short-tail", jo(draft.get('keywords_short')) or '—')}
      {kv("Mid-tail", jo(draft.get('keywords_mid')) or '—')}
      {kv("Long-tail", jo(draft.get('keywords_long')) or '—')}
    </table>
  </div>

  <div class="card">
    <h3>Final (L2 — Refined)</h3>
    <table>
      {kv("Title", final_.get('new_title') or '—')}
      {kv("Description (HTML)", f"<div class='mono'>{(final_.get('new_description') or '—').replace('<','&lt;').replace('>','&gt;')}</div>")}
      {kv("Short-tail", jo(final_.get('keywords_short')) or '—')}
      {kv("Mid-tail", jo(final_.get('keywords_mid')) or '—')}
      {kv("Long-tail", jo(final_.get('keywords_long')) or '—')}
    </table>
  </div>
</body></html>"""
    return html.encode("utf-8"), "text/html", "shakti_report.html"

# ----------------- UI -----------------
st.markdown("""
<style>
  .block-container {padding-top: 1.2rem; max-width: 980px;}
  .cred {display:flex; gap:.5rem; flex-wrap:wrap; margin:.6rem 0 1rem;}
  .cred .pill {background:#eef2ff; border:1px solid #c7d2fe; color:#3730a3; padding:.35rem .65rem; border-radius:999px; font-weight:600; font-size:.85rem;}
  .subtitle {color:#6b7280; margin-top:.15rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("## Shakti 1.0 — PW SEO Optimizer")
st.markdown("<div class='subtitle'>PW in-house SEO optimization app made by <b>Vishal Tiwari (pw17633)</b>, Project Head: <b>Kumar Sanskar</b></div>", unsafe_allow_html=True)
st.markdown("<div class='cred'><span class='pill'>Shakti 1.0</span><span class='pill'>PW In-house</span><span class='pill'>Author: Vishal Tiwari</span><span class='pill'>Project Head: Kumar Sanskar</span></div>", unsafe_allow_html=True)

tabs = st.tabs(["① Inputs", "② AI Engines", "③ Results"])

# ---- TAB 1: INPUTS ----
with tabs[0]:
    st.subheader("Listing Inputs")
    prev_title = st.text_input("Previous Title", "")
    prev_desc  = st.text_area("Previous Description", height=160, value="")
    product_link = st.text_input("Amazon Product Link (optional)", "")

    st.markdown("---")
    st.subheader("System Prompts")
    st.caption("Level 1 (OpenAI)")
    use_docx_l1 = st.toggle("Load L1 from .docx", key="docx_l1")
    system_prompt_l1 = ""
    if use_docx_l1:
        docx_file_l1 = st.file_uploader("Upload L1 prompt (.docx)", type=["docx"], key="docx_file_l1")
        if docx_file_l1:
            system_prompt_l1 = extract_text_from_docx(docx_file_l1.read())
            if system_prompt_l1.startswith("[DOCX READ ERROR]"):
                st.warning(system_prompt_l1)
            else:
                with st.expander("Preview L1 prompt"):
                    st.write(system_prompt_l1)
    else:
        system_prompt_l1 = st.text_area("Paste L1 system prompt", height=180, key="l1_prompt_text")

    st.caption("Level 2 (Refinement Engine)")
    use_docx_l2 = st.toggle("Load L2 from .docx", key="docx_l2")
    system_prompt_l2 = ""
    if use_docx_l2:
        docx_file_l2 = st.file_uploader("Upload L2 prompt (.docx)", type=["docx"], key="docx_file_l2")
        if docx_file_l2:
            system_prompt_l2 = extract_text_from_docx(docx_file_l2.read())
            if system_prompt_l2.startswith("[DOCX READ ERROR]"):
                st.warning(system_prompt_l2)
            else:
                with st.expander("Preview L2 prompt"):
                    st.write(system_prompt_l2)
    else:
        system_prompt_l2 = st.text_area("Paste L2 system prompt", height=160, key="l2_prompt_text")

# ---- TAB 2: ENGINES ----
with tabs[1]:
    st.subheader("Primary Engine — OpenAI (required)")
    openai_key_mode = st.radio("OpenAI Key Source", ["Inline", "Enter now"], horizontal=True, key="okm")
    openai_key = (OPENAI_API_KEY_INLINE or "").strip() if openai_key_mode == "Inline" else st.text_input("OpenAI API Key", type="password", key="okey")
    openai_choice = st.selectbox("OpenAI Model", list(OPENAI_MODELS.keys()), index=0)
    openai_model = OPENAI_MODELS[openai_choice]

    st.markdown("---")
    st.subheader("Secondary Engine — choose one")
    second_engine = st.selectbox("Engine", ["None", "OpenAI (second pass)", "Gemini (Google)", "Claude (Anthropic)"], index=1)

    # OpenAI second pass
    openai2_key = openai_key
    openai2_model = openai_model
    if second_engine == "OpenAI (second pass)":
        ok2_mode = st.radio("OpenAI-2 Key Source", ["Reuse primary key", "Enter another key"], horizontal=True, key="ok2m")
        if ok2_mode == "Enter another key":
            openai2_key = st.text_input("OpenAI API Key (second pass)", type="password", key="okey2")
        openai2_choice = st.selectbox("OpenAI Model (second pass)", list(OPENAI_MODELS.keys()), index=0, key="openai2_choice")
        openai2_model = OPENAI_MODELS[openai2_choice]

    # Gemini
    gemini_key = ""
    gemini_model = ""
    if second_engine == "Gemini (Google)":
        gem_key_mode = st.radio("Gemini Key Source", ["Inline", "Enter now"], horizontal=True, key="gkm")
        gemini_key = (GEMINI_API_KEY_INLINE or "").strip() if gem_key_mode == "Inline" else st.text_input("Gemini API Key", type="password", key="gkey")
        gem_choice = st.selectbox("Gemini Model", list(GEMINI_MODELS.keys()), index=0)
        gemini_model = GEMINI_MODELS[gem_choice]

    # Claude
    anthropic_key = ""
    claude_model = ""
    if second_engine == "Claude (Anthropic)":
        a_key_mode = st.radio("Claude Key Source", ["Inline", "Enter now"], horizontal=True, key="akm")
        anthropic_key = (ANTHROPIC_API_KEY_INLINE or "").strip() if a_key_mode == "Inline" else st.text_input("Anthropic API Key", type="password", key="akey")
        claude_choice = st.selectbox("Claude Model", list(CLAUDE_MODELS.keys()), index=0)
        claude_model = CLAUDE_MODELS[claude_choice]

# ---- TAB 3: RESULTS ----
with tabs[2]:
    st.subheader("Run & Results")

    json_contract = """
Return ONLY a JSON object with keys:
- new_title (string, <=200 chars)
- new_description (string, HTML, <=2000 chars)
- keywords_short (array of strings)
- keywords_mid (array of strings)
- keywords_long (array of strings)
No extra text or markdown.
"""
    user_instruction_l1 = f"""
You are given an existing Amazon listing fragment.

Inputs:
- Previous Title: {prev_title or '(empty)'}
- Previous Description: {prev_desc or '(empty)'}
- Product Link: {product_link or '(none)'}

TASK: Using the Level-1 system prompt’s framework, produce an improved listing.
{json_contract}
""".strip()

    user_instruction_l2 = f"""
Refine a JSON listing produced by another model using the Level-2 system prompt rules.
- Keep the exact JSON structure (new_title, new_description, keywords_short, keywords_mid, keywords_long).
- Title <= 200 chars; Description is HTML <= 2000 chars; avoid keyword stuffing; ensure clarity & compliance.
Return ONLY the JSON.

Context:
Previous Title: {prev_title or '(empty)'}
Previous Description: {prev_desc or '(empty)'}
Product Link: {product_link or '(none)'}
""".strip()

    run_btn = st.button("Run L1 → L2 Pipeline 🚀", type="primary")

    draft_result, final_result = {}, {}
    if run_btn:
        # validations
        if not (system_prompt_l1 or "").strip():
            st.error("Please provide Level-1 system prompt (paste or .docx).")
            st.stop()
        if second_engine != "None" and not (system_prompt_l2 or "").strip():
            st.error("You selected a secondary engine. Provide the Level-2 system prompt.")
            st.stop()
        if not openai_key:
            st.error("OpenAI API key is required for Level-1.")
            st.stop()
        if second_engine == "OpenAI (second pass)" and not openai2_key:
            st.error("Provide OpenAI key for the second pass (or reuse primary).")
            st.stop()
        if second_engine == "Gemini (Google)" and not gemini_key:
            st.error("Provide Gemini API key for the second pass.")
            st.stop()
        if second_engine == "Claude (Anthropic)" and not anthropic_key:
            st.error("Provide Anthropic API key for the second pass.")
            st.stop()

        # ---- L1 OpenAI
        try:
            if OpenAI is None:
                raise RuntimeError("openai SDK not installed. pip install openai")
            client = OpenAI(api_key=openai_key)
            completion = client.chat.completions.create(
                model=openai_model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt_l1},
                    {"role": "user", "content": user_instruction_l1},
                ],
            )
            raw = (completion.choices[0].message.content or "").strip()
            parsed = coerce_json(raw)
            if parsed is None:
                st.warning("L1 (OpenAI) did not return valid JSON. Raw output shown.")
                st.code(raw, language="json")
            else:
                draft_result = ensure_listing_shape(parsed)
                st.success("L1 draft generated (OpenAI).")
        except Exception as e:
            st.error(f"L1 (OpenAI) call failed: {e}")

        # ---- L2 refine
        if draft_result:
            if second_engine == "None":
                final_result = draft_result
                st.info("No secondary engine selected. Using L1 draft as final.")
            elif second_engine == "OpenAI (second pass)":
                try:
                    client2 = OpenAI(api_key=openai2_key)
                    completion2 = client2.chat.completions.create(
                        model=openai2_model,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": system_prompt_l2},
                            {"role": "user", "content": user_instruction_l2 + "\n\nDraft JSON:\n" + json.dumps(draft_result, ensure_ascii=False)},
                        ],
                    )
                    raw2 = (completion2.choices[0].message.content or "").strip()
                    parsed2 = coerce_json(raw2)
                    final_result = ensure_listing_shape(parsed2) if parsed2 else draft_result
                    if parsed2: st.success("Final result generated (OpenAI L2).")
                    else: st.warning("L2 (OpenAI) returned non-JSON. Using L1 draft.")
                except Exception as e:
                    st.warning(f"L2 (OpenAI) failed: {e}. Using L1 draft.")
                    final_result = draft_result

            elif second_engine == "Gemini (Google)":
                try:
                    if genai is None:
                        raise RuntimeError("google-generativeai not installed. pip install google-generativeai")
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel(gemini_model)
                    prompt = system_prompt_l2 + "\n\n" + user_instruction_l2 + "\n\nDraft JSON:\n" + json.dumps(draft_result, ensure_ascii=False)
                    resp = model.generate_content(prompt)
                    text = (resp.text or "").strip()
                    parsed2 = coerce_json(text)
                    final_result = ensure_listing_shape(parsed2) if parsed2 else draft_result
                    if parsed2: st.success("Final result generated (Gemini L2).")
                    else: st.warning("L2 (Gemini) returned non-JSON. Using L1 draft.")
                except Exception as e:
                    st.warning(f"L2 (Gemini) failed: {e}. Using L1 draft.")
                    final_result = draft_result

            elif second_engine == "Claude (Anthropic)":
                try:
                    if anthropic is None:
                        raise RuntimeError("anthropic SDK not installed. pip install anthropic")
                    aclient = anthropic.Anthropic(api_key=anthropic_key)
                    msg = aclient.messages.create(
                        model=claude_model,
                        max_tokens=2000,
                        temperature=0.2,
                        system=system_prompt_l2,
                        messages=[{
                            "role": "user",
                            "content": user_instruction_l2 + "\n\nDraft JSON:\n" + json.dumps(draft_result, ensure_ascii=False)
                        }],
                    )
                    text = "".join([b.text for b in msg.content if getattr(b, "type", "") == "text"])
                    parsed2 = coerce_json(text)
                    final_result = ensure_listing_shape(parsed2) if parsed2 else draft_result
                    if parsed2: st.success("Final result generated (Claude L2).")
                    else: st.warning("L2 (Claude) returned non-JSON. Using L1 draft.")
                except Exception as e:
                    st.warning(f"L2 (Claude) failed: {e}. Using L1 draft.")
                    final_result = draft_result

        # ---- Render
        if final_result:
            st.markdown("### ✅ Final Output")
            st.markdown("**New Title**")
            st.write(final_result["new_title"] or "—")

            st.markdown("**New Description (HTML)**")
            st.code(final_result["new_description"] or "—", language="html")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Short-tail**")
                st.write("\n".join(f"• {k}" for k in (final_result["keywords_short"] or [])) or "—")
            with c2:
                st.markdown("**Mid-tail**")
                st.write("\n".join(f"• {k}" for k in (final_result["keywords_mid"] or [])) or "—")
            with c3:
                st.markdown("**Long-tail**")
                st.write("\n".join(f"• {k}" for k in (final_result["keywords_long"] or [])) or "—")

            # Taller table view
            df = pd.DataFrame([
                {
                    "Stage": "L1 Draft",
                    "New Title": draft_result.get("new_title") or "",
                    "New Description (HTML)": draft_result.get("new_description") or "",
                    "Short-tail": ", ".join(draft_result.get("keywords_short") or []),
                    "Mid-tail": ", ".join(draft_result.get("keywords_mid") or []),
                    "Long-tail": ", ".join(draft_result.get("keywords_long") or []),
                },
                {
                    "Stage": "L2 Final",
                    "New Title": final_result.get("new_title") or "",
                    "New Description (HTML)": final_result.get("new_description") or "",
                    "Short-tail": ", ".join(final_result.get("keywords_short") or []),
                    "Mid-tail": ", ".join(final_result.get("keywords_mid") or []),
                    "Long-tail": ", ".join(final_result.get("keywords_long") or []),
                },
            ])
            st.markdown("#### Table View")
            st.dataframe(df, use_container_width=True, height=520)  # taller height

            # Downloads
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Table (CSV)", data=csv_bytes, file_name="shakti_table.csv", mime="text/csv")

            # HTML report only
            data_bytes, mime, fname = html_report_bytes(
                app_meta={"name": "Shakti 1.0"},
                inputs={"prev_title": prev_title, "prev_desc": prev_desc, "product_link": product_link},
                draft=draft_result,
                final_=final_result
            )
            st.download_button("⬇️ Download Report (HTML)", data=data_bytes, file_name=fname, mime=mime)
