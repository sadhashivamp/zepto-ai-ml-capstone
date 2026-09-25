# Zepto Support Assistant

A simple policy-based customer support assistant built using Sentence Transformers, ChromaDB, LangGraph, Pydantic and FastAPI.

## Architecture

```text
8 Zepto Policy Documents
        ↓
Document Embeddings
        ↓
Sentence Transformer
all-MiniLM-L6-v2
        ↓
ChromaDB
        ↓
User Query
        ↓
LangGraph
        ↓
classify_intent
     ↙          ↘
policy_question  general_question
      ↓                ↓
retrieve_and_answer  direct_answer
      ↓                ↓
   MOCK_LLM         MOCK_LLM
      ↓                ↓
        Pydantic Response
                ↓
          FastAPI /ask