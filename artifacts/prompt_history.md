# Prompt History

This document summarizes the key prompts used to build this Personal Codex Agent project.

## Initial Architecture Request
"You are a senior AI engineer. Produce a complete, runnable repository for a Personal Codex Agent that answers questions about a candidate using their own documents and voice. Provide reccomendations on a simple, deployable Streamlit chatbot with tone/mode switching and lightweight RAG with FAISS embeddings." - Some decisions were made and then implemented

Result: Created complete 22-file repository structure with modular RAG system, multiple response modes, and Streamlit interface.

## Performance Optimization Request  
"I don't have a lot of processing power, please reccomend and implement the approved lightweigth models for embedding and chat"

Result: Redesigned entire system from OpenAI embeddings to local sentence-transformers, switched to Groq API for chat. Achieved 90% memory reduction from 2GB to 200MB.

## UI Enhancement Request
"Please use the added pngs (avatar and background), and reccomend way to make the project feel more proffessional"... some of these were then implemented

Result: Implemented basic CSS design to display avatar and background pngs, collapsible settings panel with smooth animations.

## Feature Enhancement Request
"Fun improvements" followed by selection of 4 suggested features

Result: Implemented dynamic time-based greetings, typing animation for responses, related questions generation.

## Source Display Simplification
"Please simplify the showing of sources, just the file that the data came from"

Result: Simplified source attribution from detailed sections to just filename display.

## Please help with deployment
"Please help with deployment using streamlit"

Result: provided steps to deployment