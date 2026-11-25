"""Streamlit UI: speak a question, read (and optionally hear) a cited answer.

Run:  streamlit run app.py      (allow microphone access in your browser)

Audio is recorded in the browser and transcribed by Groq Whisper. The transcript
is shown for editing because recognition can still mishear names.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from openai import APIError, OpenAI

from voicerag.bootstrap import App, build_app
from voicerag.config import PROJECT_ROOT, load_settings
from voicerag.models import FinalAnswer

BUDGET_SECONDS = 5.0     # from the brief: answer within ~5 s of the learner finishing
BUDGET_INR = 0.50        # from the brief: cost per question

st.set_page_config(page_title="Voice Study Assistant", page_icon="🎙️", layout="wide")


# ----------------------------------------------------------------------------
# resources (loaded once per server process, not on every click)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and indexing the PDF (first run only)...")
def get_app(pdf_path: str) -> App:
    """Build the pipeline. The index itself is cached on disk by file hash."""
    return build_app(load_settings(), Path(pdf_path))


@st.cache_resource
def get_transcription_client(api_key: str) -> OpenAI:
    """Create the Groq-compatible client used for Whisper transcription."""
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=60.0,
    )


def save_upload(uploaded) -> Path:
    """Store an uploaded PDF under data/uploads/ and return its path."""
    folder = PROJECT_ROOT / "data" / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / Path(uploaded.name).name
    target.write_bytes(uploaded.getvalue())
    return target


# ----------------------------------------------------------------------------
# rendering helpers
# ----------------------------------------------------------------------------
def speak(text: str) -> None:
    """Read `text` aloud with the browser's speech synthesis (best effort)."""
    components.html(
        f"<script>const u = new SpeechSynthesisUtterance({json.dumps(text)});"
        "window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);</script>",
        height=0,
    )


def render_metrics(result: FinalAnswer) -> None:
    """Per-question cost, tokens and per-stage timings."""
    m = result.metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total time", f"{m.total_seconds:.2f} s", delta=f"budget {BUDGET_SECONDS:.0f} s", delta_color="off")
    c2.metric("Cost (notional)", f"₹{m.cost_inr:.4f}", delta=f"budget ₹{BUDGET_INR:.2f}", delta_color="off")
    c3.metric("LLM calls / tokens", f"{len(m.calls)} / {m.prompt_tokens + m.completion_tokens}")

    rows = [{"stage": k, "seconds": round(v, 3)} for k, v in m.stage_seconds.items()]
    rows.append({"stage": "local compute (total - LLM wait)", "seconds": round(m.local_seconds, 3)})
    st.table(rows)
    st.caption("Per LLM call")
    st.table(
        [
            {
                "stage": c.stage, "model": c.model, "prompt tok": c.prompt_tokens,
                "completion tok": c.completion_tokens, "seconds": round(c.seconds, 2),
                "₹": round(c.cost_inr, 5), "tokens estimated": c.estimated,
            }
            for c in m.calls
        ]
    )


def render_result(result: FinalAnswer) -> None:
    """Answer, citations, evidence and metrics for one question."""
    if result.found:
        st.success(result.answer)
        st.markdown("**Sources:** " + ", ".join(f"page {p}" for p in result.pages))
    else:
        st.warning(result.answer)

    for warning in result.warnings:
        st.caption(f"⚠️ {warning}")

    with st.expander("How this answer was produced"):
        for step in result.path:
            st.markdown(f"- {step}")
    with st.expander("Evidence shown to the model"):
        for sc in result.sources:
            st.markdown(f"**{sc.chunk.prompt_label}**  · rerank score `{sc.score:.3f}` · {sc.origin}")
            st.text(sc.chunk.text)
    with st.expander("Cost and timing", expanded=True):
        render_metrics(result)


def ask(app: App, question: str) -> None:
    """Run the pipeline and remember the result for display."""
    with st.spinner("Thinking..."):
        try:
            st.session_state.last = app.pipeline.ask(question, st.session_state.session)
        except Exception as exc:    # show provider/config errors instead of a stack trace
            st.session_state.last = None
            st.error(f"Could not answer: {exc}")
            return
    st.session_state.last_question = question


# ----------------------------------------------------------------------------
# page
# ----------------------------------------------------------------------------
st.title("🎙️ Voice Study Assistant")

with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Use a different PDF", type="pdf")
    pdf_path = save_upload(uploaded) if uploaded else load_settings().pdf_path
    speak_answers = st.checkbox("Speak the answer aloud", value=False)

try:
    app = get_app(str(pdf_path))
except Exception as exc:
    st.error(f"Startup failed: {exc}")
    st.stop()

if "session" not in st.session_state or st.session_state.get("pdf") != str(pdf_path):
    st.session_state.session = app.pipeline.new_session()
    st.session_state.update(
        pdf=str(pdf_path), last=None, last_question="", typed_question=""
    )
elif "typed_question" not in st.session_state:
    st.session_state.typed_question = st.session_state.last_question

with st.sidebar:
    idx = app.index
    st.caption(
        f"{len(idx.chunks)} chunks · index {'loaded from cache' if idx.from_cache else 'built'} "
        f"in {idx.seconds:.1f} s · key `{idx.key}`"
    )
    st.caption(f"LLM: `{app.settings.llm_model}` · pricing: {app.settings.pricing_note}")
    ledger = st.session_state.session.ledger
    st.header("This session")
    st.metric("Questions asked", ledger.count)
    st.metric("Total cost (notional)", f"₹{ledger.total_cost_inr:.4f}")
    st.metric("Average per question", f"₹{ledger.average_cost_inr:.4f}")
    st.metric("Average time", f"{ledger.average_seconds:.2f} s")
    if st.button("New conversation"):
        st.session_state.session = app.pipeline.new_session()
        st.session_state.last, st.session_state.last_question = None, ""
        st.session_state.typed_question = ""
        st.rerun()

# ---- input: Groq Whisper with browser speech recognition fallback ----
from streamlit_mic_recorder import mic_recorder, speech_to_text  # noqa: E402


st.subheader("Ask")
audio = mic_recorder(
    start_prompt="🎙️ Start speaking", stop_prompt="⏹️ Stop",
    just_once=True, format="wav", key="stt"
)
settings = load_settings()
if not settings.groq_api_key:
    st.session_state.show_browser_fallback = True
    st.caption("Groq is not configured. Use the browser speech option below or type a question.")

if audio and audio.get("bytes"):
    audio_digest = hashlib.sha256(audio["bytes"]).hexdigest()
    if audio_digest != st.session_state.get("last_audio_digest"):
        st.session_state.last_audio_digest = audio_digest
        if not settings.groq_api_key:
            st.info("No Groq key is configured; use the browser speech option below.")
        else:
            try:
                with st.spinner("Transcribing with Whisper..."):
                    response = get_transcription_client(
                        settings.groq_api_key
                    ).audio.transcriptions.create(
                        model=settings.whisper_model,
                        file=("question.wav", audio["bytes"], "audio/wav"),
                        language="en",
                        prompt="A spoken question about a PDF. Preserve proper nouns, place names, dates, and numbers.",
                    )
                spoken = response.text.strip()
                if spoken:
                    st.session_state.typed_question = spoken
                else:
                    st.warning("Whisper returned an empty transcript. Please try again or type your question.")
                    st.session_state.show_browser_fallback = True
            except APIError as exc:
                st.error(f"Groq Whisper transcription failed: {exc}")
                st.session_state.show_browser_fallback = True

if st.session_state.get("show_browser_fallback"):
    st.caption(
        "Fallback: use your browser's speech recognition (Chrome recommended). "
        "It is not billed through Groq, but availability and limits depend on the browser service."
    )
    browser_spoken = speech_to_text(
        language="en",
        start_prompt="🎙️ Try browser speech",
        stop_prompt="⏹️ Stop",
        just_once=True,
        key="browser_stt_fallback",
    )
    if browser_spoken:
        st.session_state.typed_question = browser_spoken

with st.form("typed", clear_on_submit=False):
    typed = st.text_input(
        "Transcript (edit it if Whisper misheard, then press Ask)",
        key="typed_question",
    )
    if st.form_submit_button("Ask") and typed.strip():
        ask(app, typed.strip())
        if speak_answers and st.session_state.last:
            speak(st.session_state.last.answer)

# ---- output ----
if st.session_state.last:
    st.markdown(f"**You asked:** {st.session_state.last_question}")
    render_result(st.session_state.last)
