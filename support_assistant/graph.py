import os
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

from support_assistant.models import AnswerResponse


class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_ids: list[str]
    retrieved_docs: list[str]
    answer: str


model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="support_assistant/vector_store"
)

collection = client.get_collection(
    name="zepto_policies"
)


def mock_llm_enabled():
    return os.getenv("MOCK_LLM", "1") != "0"


def classify_intent(state: SupportState):
    query = state["query"]

    if not mock_llm_enabled():
        raise NotImplementedError("Real LLM classification is optional.")

    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    query_lower = query.lower()

    if any(keyword in query_lower for keyword in keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {"intent": intent}


def retrieve_and_answer(state: SupportState):
    query = state["query"]

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    documents = results["documents"][0]
    ids = results["ids"][0]

    top_chunk_snippet = documents[0][:200]

    if mock_llm_enabled():
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
    else:
        raise NotImplementedError("Real LLM answer generation is optional.")

    return {
        "retrieved_ids": ids,
        "retrieved_docs": documents,
        "answer": answer
    }


def direct_answer(state: SupportState):
    if mock_llm_enabled():
        answer = "I can only answer questions about Zepto policies right now."
    else:
        raise NotImplementedError("Real LLM answer generation is optional.")

    return {"answer": answer}


def route_query(state: SupportState):
    return state["intent"]


builder = StateGraph(SupportState)

builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)

builder.add_edge(START, "classify_intent")

builder.add_conditional_edges(
    "classify_intent",
    route_query,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)

builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)

graph = builder.compile()


def make_response(result):
    if result["intent"] == "policy_question":
        return AnswerResponse(
            answer=result["answer"],
            sources=result["retrieved_ids"],
            confidence=1.0
        )

    return AnswerResponse(
        answer=result["answer"],
        sources=[],
        confidence=1.0
    )