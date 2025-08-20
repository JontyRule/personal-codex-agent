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
# Updated import to use deployment-ready retriever
from rag.retriever_deployment import retrieve_with_prebuilt as retrieve

logger = get_logger("codex.app")

ROOT_DIR = os.path.dirname(__file__)
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

def get_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Get free API key from https://console.groq.com/")
    
    try:
        return Groq(api_key=key)
    except Exception as e:
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

    # Retrieve context using pre-built embeddings
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

def check_prebuilt_index():
    """Check if pre-built index exists, show instructions if not"""
    index_path = os.path.join(ROOT_DIR, "rag", "cache", "index.faiss")
    meta_path = os.path.join(ROOT_DIR, "rag", "cache", "meta.json")
    
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        st.error("**Pre-built index not found!**")
        st.info("""
        **To fix this:**
        1. Run locally: `python scripts/build_embeddings_local.py`
        2. Commit the generated files in `rag/cache/`
        3. Redeploy the app
        
        This builds the embeddings once locally and includes them in deployment.
        """)
        return False
    return True

# Render deployment fix
import os
if "PORT" in os.environ:
    # Running on Render - configure for external access
    os.environ.setdefault("STREAMLIT_SERVER_PORT", os.environ["PORT"])
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")

def main():
    st.set_page_config(
        page_title="AskJonty", 
        page_icon="😎", 
        layout="wide",
        initial_sidebar_state="collapsed"  # Start collapsed for mobile
    )

    load_dotenv()

    # Check if pre-built index exists
    if not check_prebuilt_index():
        st.stop()

    # Add mobile-friendly CSS
    st.markdown("""
    <style>
    /* Mobile-first responsive design */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Main container responsive padding */
    .main .block-container {
        padding: 1rem;
        max-width: 100%;
    }
    
    /* Mobile-optimized header */
    h1 {
        font-size: 1.8rem !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
    }
    
    /* Mobile-friendly buttons */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        border: 2px solid #1e3a8a;
        background: linear-gradient(45deg, #f8f9fa, white);
        color: #1e3a8a;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #1e3a8a;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(30, 58, 138, 0.2);
    }
    
    /* Chat input mobile optimization */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background: white;
        border-top: 1px solid #e9ecef;
        z-index: 1000;
    }
    
    /* Chat messages mobile-friendly */
    .stChatMessage {
        margin: 0.5rem 0;
        padding: 1rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Sidebar mobile optimization */
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Quick start questions responsive grid */
    .element-container .stColumns {
        gap: 0.5rem;
    }
    
    /* Mobile typography */
    .stMarkdown p {
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Expander mobile-friendly */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        padding: 0.75rem;
    }
    
    /* Sources display mobile */
    .stMarkdown strong {
        color: #1e3a8a;
        font-size: 0.85rem;
    }
    
    /* Mobile viewport adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem;
        }
        
        h1 {
            font-size: 1.5rem !important;
        }
        
        .stButton > button {
            padding: 0.6rem 0.8rem;
            font-size: 0.85rem;
        }
        
        .stChatMessage {
            padding: 0.75rem;
            font-size: 0.9rem;
        }
        
        /* Hide sidebar by default on mobile */
        .css-1d391kg {
            transform: translateX(-100%);
        }
    }
    
    /* Very small screens */
    @media (max-width: 480px) {
        .main .block-container {
            padding: 0.25rem;
        }
        
        h1 {
            font-size: 1.3rem !important;
        }
        
        .stButton > button {
            padding: 0.5rem;
            font-size: 0.8rem;
        }
    }
    
    /* Landscape mobile optimization */
    @media (max-height: 500px) and (orientation: landscape) {
        .main .block-container {
            padding: 0.5rem;
        }
        
        h1 {
            font-size: 1.2rem !important;
            margin-bottom: 0.25rem !important;
        }
    }
    
    /* Touch-friendly interactive elements */
    .stRadio > div {
        gap: 0.75rem;
    }
    
    .stRadio label {
        padding: 0.5rem;
        font-size: 0.9rem;
    }
    
    /* Improve readability on mobile */
    .stChatMessage [data-testid="column"] {
        padding: 0.5rem;
    }
    
    /* Mobile-optimized spacing */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Better mobile input */
    .stTextInput input {
        font-size: 16px; /* Prevents zoom on iOS */
        padding: 0.75rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Get background image as base64 (only for desktop)
    background_path = os.path.join(ROOT_DIR, "assets", "background.png")
    background_b64 = get_background_image_base64(background_path)
    
    # Add subtle background for desktop only
    if background_b64:
        st.markdown(f"""
        <style>
        @media (min-width: 769px) {{
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
                opacity: 0.1;
                z-index: -1;
                pointer-events: none;
            }}
            
            .main .block-container {{
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 10px;
                padding: 2rem;
                margin-top: 1rem;
            }}
        }}
        </style>
        """, unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.markdown("## Settings")
        
        mode = st.radio(
            "**Response Mode**",
            options=["Interview", "Storytelling", "Fast Facts", "Humble Brag", "Reflective", "Humorous"],
            index=0,  # This is already set to Interview mode (first option)
        )
        
        st.markdown("---")
        
        temperature = st.slider("**Temperature**", 0.0, 1.0, 0.4, 0.05, 
                               help="Lower = more consistent, Higher = more creative")
        
        reflective = st.toggle("**Reflective mode**", value=(mode == "Reflective"),
                              help="Prioritize self-reflection documents")
        
        show_sources = st.toggle("**Show sources**", value=True,
                                help="Display source attribution")
        
        st.markdown("---")
        
        # Model info
        st.markdown("""
        **Current Setup:**
        - **Chat**: Groq API (LLaMA 3)
        - **Embeddings**: Pre-built (deployment-ready)
        - **Cost**: Free!
        """)

    # Header with dynamic greeting
    st.markdown("# AskJonty")
    st.markdown("*AI-powered JontyBot grounded in real documents, HALLUCINATIONS OCCUR, confirm with the real one*")

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

    # Mobile-friendly button layout - stack on mobile, grid on desktop
    # Check if we should use mobile layout (this is a simple heuristic)
    cols_per_row = 2 if len(sample_qs) > 3 else len(sample_qs)
    
    # Create rows of buttons for better mobile experience
    for i in range(0, len(sample_qs), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(sample_qs):
                q = sample_qs[i + j]
                with col:
                    if st.button(q, key=f"q_{i+j}", use_container_width=True):
                        st.session_state["last_question"] = q

    st.markdown("---")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg_idx, m in enumerate(st.session_state["messages"]):
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
                    with st.expander("Related Questions", expanded=False):
                        for rq_idx, rq in enumerate(m["related_questions"]):
                            # Make key unique with message index and question index
                            if st.button(rq, key=f"related_history_{msg_idx}_{rq_idx}"):
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
                        st.markdown(f"\n\n**Sources:** {sources_footer}")
                    
                    # Show related questions
                    if related_questions:
                        with st.expander("You might also ask:", expanded=False):
                            for rq_idx, rq in enumerate(related_questions):
                                # Make key unique for new questions
                                if st.button(rq, key=f"related_new_{len(st.session_state['messages'])}_{rq_idx}"):
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
