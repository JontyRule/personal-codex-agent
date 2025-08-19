from __future__ import annotations

import os
import sys
import time
import base64
import datetime
import random
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


def get_background_image_base64(image_path: str) -> str:
    """Convert image to base64 string for CSS background"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None


def get_dynamic_greeting() -> str:
    """Return a greeting based on time of day"""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        greetings = ["Good morning! Ready for some questions?", "Morning! What would you like to know?", "Rise and shine! Let's chat!"]
    elif 12 <= hour < 17:
        greetings = ["Good afternoon! How can I help?", "Afternoon! What's on your mind?", "Hey there! Ready to explore?"]
    elif 17 <= hour < 22:
        greetings = ["Good evening! What brings you here?", "Evening! Let's dive into some questions!", "Hey! Perfect time for a chat!"]
    else:
        greetings = ["Working late? Let's chat!", "Night owl? I'm here to help!", "Burning the midnight oil? Ask away!"]
    
    return random.choice(greetings)





def generate_related_questions(query: str, retrieval_data: Dict) -> List[str]:
    """Generate related questions based on the query and retrieved content"""
    question_templates = [
        "Can you tell me more about {}?",
        "What's your experience with {}?",
        "How did you develop your skills in {}?",
        "What challenges did you face with {}?",
        "What's your biggest accomplishment in {}?"
    ]
    
    # Extract key topics from the query and retrieved content
    topics = []
    query_words = query.lower().split()
    
    # Common professional topics that might be relevant
    professional_topics = [
        "leadership", "teamwork", "projects", "challenges", "goals", 
        "achievements", "skills", "experience", "growth", "learning"
    ]
    
    # Find topics mentioned in query
    for topic in professional_topics:
        if topic in query.lower():
            topics.append(topic)
    
    # If no specific topics found, use general ones
    if not topics:
        topics = random.sample(["your background", "your projects", "your skills", "your goals"], 2)
    
    # Generate related questions
    related = []
    for topic in topics[:2]:  # Limit to 2 topics
        template = random.choice(question_templates)
        related.append(template.format(topic))
    
    return related


def typing_animation(text: str, container):
    """Simple typing animation effect"""
    placeholder = container.empty()
    displayed_text = ""
    
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text + "▌")
        time.sleep(0.02)  # Adjust speed here
    
    placeholder.markdown(text)  # Final text without cursor


def get_mode_file(mode: str) -> str:
    mapping = {
        "Interview": "mode_interview.txt",
        "Storytelling": "mode_storytelling.txt",
        "Fast Facts": "mode_fastfacts.txt",
        "Humble Brag": "mode_humblebrag.txt",
        "Reflective": "mode_reflective.txt",
        "Humorous": "mode_humorous.txt"
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
    with st.expander("Source Details & Retrieval Scores", expanded=False):
        st.markdown("**Retrieved chunks with similarity scores:**")
        for item in r["results"]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{os.path.basename(item['source'])}#{item['heading']}**")
            with col2:
                score_color = "Good" if item['score'] > 0.7 else "OK" if item['score'] > 0.5 else "Low"
                st.markdown(f"{score_color} {item['score']:.3f}")
            
            # Show truncated text in a nice code block
            text_preview = item["text"][:800] + ("..." if len(item["text"]) > 800 else "")
            st.code(text_preview, language="markdown")


def maybe_rebuild_index():
    with st.spinner("Rebuilding index..."):
        meta = build_index()
    st.success(f"Index rebuilt with {meta['count'] if 'count' in meta else len(meta['chunks'])} chunks.")


def main():
    st.set_page_config(
        page_title="AskJonty", 
        page_icon="", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_dotenv()

    # Get background image as base64
    background_path = os.path.join(ROOT_DIR, "assets", "background.png")
    background_b64 = get_background_image_base64(background_path)
    
    # Add custom CSS for background image
    if background_b64:
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url('data:image/png;base64,{background_b64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        .stApp::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url('data:image/png;base64,{background_b64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            opacity: 0.15;
            z-index: -1;
            pointer-events: none;
        }}
        
        /* Ensure main content has white background */
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 2rem;
            margin-top: 1rem;
        }}
        
        /* Ensure sidebar remains readable */
        .css-1d391kg {{
            background-color: rgba(255, 255, 255, 0.85) !important;
        }}
        
        /* Chat messages with background */
        .stChatMessage {{
            background-color: rgba(255, 255, 255, 0.9) !important;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.info("Background image not found. Please add 'background.png' to the assets folder.")

    # Sidebar controls
    with st.sidebar:
        st.markdown("## Settings")
        
        mode = st.radio(
            "**Response Mode**",
            options=["Interview", "Storytelling", "Fast Facts", "Humble Brag", "Reflective", "Humorous"],
            index=0,
        )
        
        st.markdown("---")
        
        temperature = st.slider("**Temperature**", 0.0, 1.0, 0.4, 0.05, 
                               help="Lower = more consistent, Higher = more creative")
        
        reflective = st.toggle("**Reflective mode**", value=(mode == "Reflective"),
                              help="Prioritize self-reflection documents")
        
        show_sources = st.toggle("**Show sources**", value=True,
                                help="Display source attribution")
        
        st.markdown("---")
        
        rebuild = st.button("Rebuild Index", type="secondary", use_container_width=True)
        if rebuild:
            try:
                maybe_rebuild_index()
            except Exception as e:
                st.error(str(e))
        
        st.markdown("---")
        
        # Model info with dark styling
        st.markdown("""
        **Current Setup:**
        - **Chat**: Groq API (LLaMA 3)
        - **Embeddings**: Local (all-MiniLM-L6-v2)
        - **Cost**: Free!
        """)

    # Header with dynamic greeting
    st.markdown("# AskJonty")
    st.markdown("*AI-powered JontyBot grounded in your Jontys documents, note hallucination is possible*")

    # Dynamic greeting
    if "greeting_shown" not in st.session_state:
        st.session_state.greeting_shown = True
        st.info(get_dynamic_greeting())

    # Sample quick-start questions
    st.markdown("### Quick Start Questions")
    sample_qs = [
        "Tell me about yourself",
        "Describe what you like to do in the evenings?",
        "What are your key accomplishments?", 
        "How do you approach problem-solving?",
        "Where do you see yourself in 10 years?",
    ]

    # Create responsive button layout
    cols = st.columns(len(sample_qs))
    for i, q in enumerate(sample_qs):
        with cols[i]:
            if st.button(q, key=f"q_{i}", use_container_width=True):
                st.session_state["last_question"] = q

    st.markdown("---")  # Separator line

    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for m in st.session_state["messages"]:
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        elif m["role"] == "assistant":
            with st.chat_message("assistant", avatar="assets/avatar-ai.png"):
                st.markdown(m["content"])
                
                # Show sources if enabled and available
                if show_sources and m.get("sources"):
                    st.markdown(f"\n\n**Sources:** {m['sources']}")
                
                # Show related questions if available
                if m.get("related_questions"):
                    with st.expander("💡 Related Questions", expanded=False):
                        for rq in m["related_questions"]:
                            if st.button(rq, key=f"related_{hash(rq)}_{len(st.session_state['messages'])}"):
                                st.session_state["last_question"] = rq

    # Input
    q = st.chat_input("Ask about Jonty...")
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

            with st.chat_message("assistant", avatar="assets/avatar-ai.png"):
                # Show typing animation
                typing_container = st.empty()
                with st.spinner("Thinking..."):
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=1024,
                    )
                    answer = resp.choices[0].message.content or ""

                    sources_footer = " ".join(
                        [f"[{os.path.basename(x['source'])}]" for x in r["results"]]
                    )

                    # Generate related questions
                    related_questions = generate_related_questions(q, r)

                    # Display with typing animation
                    typing_animation(answer, typing_container)
                    
                    if show_sources:
                        st.markdown("\n\n" + sources_footer)
                    
                    # Show related questions
                    if related_questions:
                        with st.expander("💡 You might also ask:", expanded=False):
                            for rq in related_questions:
                                if st.button(rq, key=f"related_new_{hash(rq)}"):
                                    st.session_state["last_question"] = rq

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_footer if show_sources else None,
                    "retrieval": r,
                    "related_questions": related_questions,
                }
            )
        except Exception as e:
            st.error(f"**Error**: {str(e)}")
            if "GROQ_API_KEY" in str(e):
                st.info("**Get a free Groq API key**: https://console.groq.com/")
    

if __name__ == "__main__":
    main()
