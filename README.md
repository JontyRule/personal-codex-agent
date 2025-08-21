# Personal Codex Agent

A simple, local-first Streamlit chatbot that answers evaluator questions about a candidate using only your own documents and voice.

##  Quickstart

1) **Install dependencies** (Python 3.11+):

```bash
cd personal-codex-agent
python -m venv .venv
source .venv/bin/activate (macOS)
pip install -r requirements.txt #(requirements-dev.txt for local deployment)
```

2) **Get free Groq API key**:
   - Go to https://console.groq.com/
   - Sign up (free)
   - Create API key
   - also create a google form with Timestamo and Question fields

3) **Configure environment**:

```bash
cp .env.example .env
# Edit .env and add your `GROQ_API_KEY`, `GOOGLE_FORM_URL`, `GOOGLE_FORM_TIMESTAMP_FIELD`, `GOOGLE_FORM_QUESTION_FIELD` 
```

4) **Build the index**:
- Place .md files in /data, this is where info for answers will be sourced

```bash
python scripts/build_embeddings_local.py
```

5) **Run the app**:

```bash
streamlit run app_lite.py
```

```bash
alternative for macOS issues:
# Set macOS compatibility environment variables
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run the app
streamlit run app_lite.py
```
## Dataset & Voice

- Edit `data/profile.yaml` to define your tone, values, strengths, and style.
- Put your `.md` docs in `data/`. The indexer loads all `.md` files.

**Components:**
- **Retriever Agent**: FAISS cosine similarity over chunked docs (sentence-transformers)
- **Chat Agent**: Composes prompts with system + profile + mode + context  
- **Chat Model**: Groq API with LLaMA 3 (free tier)

## Guardrails

- Answers must be grounded in retrieved chunks + profile
- If retrieval is weak, admits gaps and suggests which doc to add
- Sources always shown as filenames
- No hallucination - refuses to answer without sufficient context

## Deployment

**Streamlit Community Cloud:**
1. Push this repo to GitHub
2. Connect to Streamlit Cloud
3. Set `GROQ_API_KEY`, `GOOGLE_FORM_URL`, `GOOGLE_FORM_TIMESTAMP_FIELD`, `GOOGLE_FORM_QUESTION_FIELD`  in secrets
4. Deploy `app_lite.py` on streamlit

## Show your thinking artifacts
Found in the /artifacts folder 

## Cost & Performance

- **Embeddings**: Free (runs locally)
- **Chat**: Groq free tier (6,000 requests/minute)
- **Storage**: ~80MB for embedding model
- **Speed**: Sub-second responses via Groq API

##  License

- MIT (personal use). Customize as needed.

## Logging
- Questions and timestamps are logged to a google form/sheet
