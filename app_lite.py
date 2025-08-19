from __future__ import annotations

import os
import sys
import time
from typing import List, Dict

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv  # type: ignore
from groq import Groq

from utils.loader import load_profile
from utils.logging_utils import get_logger
from rag.build_index_lite import build_index
from rag.retriever_lite import retrieve


logger = get_logger("codex.app")

ROOT_DIR = os.path.dirname(__file__)
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")


def get_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Get free API key from https://console.groq.com/")
    
    # Simple initialization without extra parameters
    try:
        return Groq(api_key=key)
    except Exception as e:
        # Fallback for older versions
        return Groq(api_key=key)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_mode_file(mode: str) -> str:
    mapping = {
        "Interview": "mode_interview.txt",
        "Storytelling": "mode_storytelling.txt",
        "Fast Facts": "mode_fastfacts.txt",
        "Humble Brag": "mode_humblebrag.txt",
        "Reflective": "mode_reflective.txt",
    }
    return os.path.join(PROMPTS_DIR, mapping[mode])


def compose_messages(question: str, mode: str, reflective: bool) -> tuple[List[Dict[str, str]], Dict]:
    prof = load_profile()
    system = load_text(os.path.join(PROMPTS_DIR, "system_base.txt"))
    mode_text = load_text(get_mode_file(mode))

    # Retrieve context
    r = retrieve(question, top_k=4, prioritize_reflection=reflective)

    # Guardrail: if weak, steer the assistant to refuse gracefully
    min_required = 0.20  # cosine similarity threshold heuristic
    weak = r["count"] == 0 or r["max_score"] < min_required or r["avg_score"] < 0.18

    context = "\n\n".join([
        f"[Source: {os.path.basename(x['source'])}#{x['heading']}]\n{x['text']}" for x in r["results"]
    ])

    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": f"PROFILE\n{prof.to_prompt_block()}"},
        {"role": "system", "content": f"MODE\n{mode_text}"},
        {"role": "system", "content": f"CONTEXT\n{context}"},
    ]

    if weak:
        missing_hint = (
            "Context appears thin. If you cannot answer strictly from CONTEXT and PROFILE, say what's missing "
            "and suggest adding a relevant doc (e.g., project README with metrics, performance review, or case study)."
        )
        messages.append({"role": "system", "content": missing_hint})

    messages.append({"role": "user", "content": question})

    return messages, r


def render_sources(r):
    with st.expander("Why this (retrieved context and scores)", expanded=False):
        for item in r["results"]:
            st.markdown(f"- {os.path.basename(item['source'])}#{item['heading']} — score={item['score']:.3f}")
            st.code(item["text"][:1200] + ("..." if len(item["text"]) > 1200 else ""))


def maybe_rebuild_index():
    with st.spinner("Rebuilding index..."):
        meta = build_index()
    st.success(f"Index rebuilt with {meta['count'] if 'count' in meta else len(meta['chunks'])} chunks.")


def main():
    st.set_page_config(page_title="Personal Codex Agent", page_icon="🧭", layout="wide")

    load_dotenv()

    st.title("Personal Codex Agent 🧭")
    st.caption("Answers about the candidate, grounded in local documents only. Uses lightweight local embeddings + Groq API.")

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        mode = st.radio(
            "Mode",
            options=["Interview", "Storytelling", "Fast Facts", "Humble Brag", "Reflective"],
            index=0,
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)
        reflective = st.toggle("Reflective mode (prioritize reflection)", value=(mode == "Reflective"))
        show_sources = st.toggle("Show sources", value=True)
        rebuild = st.button("Rebuild index")
        if rebuild:
            try:
                maybe_rebuild_index()
            except Exception as e:
                st.error(str(e))

        # Show model info
        st.info("💡 Using:\n- Embeddings: all-MiniLM-L6-v2 (local)\n- Chat: Groq API (free tier)")

    # Sample quick-start questions
    sample_qs = [
        "What's your strongest signal for this role?",
        "Tell me a short story about shipping meaningful impact.",
        "What are 5 fast facts I should know about you?",
        "Where have you driven measurable outcomes?",
        "What did you learn recently and how did you improve?",
    ]

    cols = st.columns(len(sample_qs))
    for i, q in enumerate(sample_qs):
        if cols[i].button(q):
            st.session_state["last_question"] = q

    # Chat container
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources") and show_sources:
                st.markdown("\n\n" + m["sources"])  # attribution footer
                render_sources(m["retrieval"])  # detailed panel

    # Input
    q = st.chat_input("Ask about the candidate...")
    if st.session_state.get("last_question") and not q:
        q = st.session_state.pop("last_question")

    if q:
        st.session_state["messages"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        try:
            messages, r = compose_messages(q, mode, reflective)
            client = get_client()
            model = os.environ.get("GROQ_MODEL", "llama3-8b-8192")

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=1024,
                    )
                    answer = resp.choices[0].message.content or ""

                    sources_footer = " ".join(
                        [f"[Doc: {os.path.basename(x['source'])}#{x['heading']}]" for x in r["results"]]
                    )

                    st.markdown(answer)
                    if show_sources:
                        st.markdown("\n\n" + sources_footer)
                        render_sources(r)

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_footer if show_sources else None,
                    "retrieval": r,
                }
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if "GROQ_API_KEY" in str(e):
                st.info("Get a free Groq API key at: https://console.groq.com/")


if __name__ == "__main__":
    main()
