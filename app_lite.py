from __future__ import annotations

import os
import sys
import streamlit as st
import datetime
import random
import time
import uuid
from typing import List, Dict
import requests
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv  # type: ignore
from groq import Groq

from utils.loader import load_profile
from utils.logging_utils import get_logger
# Updated import to use deployment-ready retriever
from rag.retriever_deployment import retrieve_with_prebuilt as retrieve

logger = get_logger("codex.app")

ROOT_DIR = os.path.dirname(__file__)
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

def log_question(question: str):
    """Log question to Google Sheets via Google Forms"""
    try:
        # Local logging (for development)
        logs_dir = os.path.join(ROOT_DIR, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(logs_dir, f"{today}.txt")
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {question}\n")
        
        # Google Sheets logging (for production)
        form_url = os.environ.get("GOOGLE_FORM_URL")
        timestamp_field = os.environ.get("GOOGLE_FORM_TIMESTAMP_FIELD")
        question_field = os.environ.get("GOOGLE_FORM_QUESTION_FIELD")
        
        if form_url and timestamp_field and question_field:
            timestamp_full = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            form_data = {
                timestamp_field: timestamp_full,
                question_field: question
            }
            
            # Submit to Google Form
            response = requests.post(form_url, data=form_data, timeout=5)
            
            if response.status_code == 200:
                logger.info("Question logged to Google Sheets successfully")
            else:
                logger.warning(f"Google Sheets logging failed: {response.status_code}")
                
    except Exception as e:
        logger.error(f"Failed to log question: {e}")

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

def get_background_image_base64(image_path: str) -> str:
    """Convert image to base64 string for CSS background"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

def main():
    st.set_page_config(
        page_title="AskJonty", 
        page_icon="😎", 
        layout="wide"
    )

    load_dotenv()

    # Check if pre-built index exists
    if not check_prebuilt_index():
        st.stop()

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

    # Create button layout for quick start questions
    cols = st.columns(len(sample_qs))
    for i, (col, q) in enumerate(zip(cols, sample_qs)):
        with col:
            if st.button(q, key=f"q_{i}"):
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
                            if st.button(rq, key=f"related_history_{msg_idx}_{rq_idx}"):
                                st.session_state["last_question"] = rq

    # Input
    q = st.chat_input("Ask about Jonty...")
    if st.session_state.get("last_question") and not q:
        q = st.session_state.pop("last_question")

    if q:
        # LOG THE QUESTION HERE
        log_question(q)
        
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
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=1024,
                    )

                assistant_content = response.choices[0].message.content

                # Simple typing effect
                typing_animation(assistant_content, typing_container)

                # Generate related questions
                related_questions = generate_related_questions(q, r)

                # Store message with metadata
                msg_data = {
                    "role": "assistant", 
                    "content": assistant_content,
                    "related_questions": related_questions
                }

                if show_sources and r.get("results"):
                    sources = ", ".join([os.path.basename(x['source']) for x in r["results"]])
                    msg_data["sources"] = sources

                st.session_state["messages"].append(msg_data)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
