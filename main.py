"""
Streamlit-powered Deep Research Agentic System
Run with:
    streamlit run app.py

Requirements:
    pip install -U streamlit langchain-core langchain-community langgraph tavily-python \
                   langchain_huggingface python-dotenv
"""
from __future__ import annotations

# ──────────────────────────────────────────────────
# Standard library imports
# ──────────────────────────────────────────────────
import os
import asyncio
import textwrap
from typing import Dict, Any, TypedDict

# ──────────────────────────────────────────────────
# Third‑party imports
# ──────────────────────────────────────────────────
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langgraph.graph import StateGraph, END, START

# ──────────────────────────────────────────────────
# Environment setup
# ──────────────────────────────────────────────────
load_dotenv()  # load .env variables as early as possible

# Validate critical API keys at startup so the user sees clear feedback
REQUIRED_ENV_VARS = [
    "TAVILY_API_KEY",
    # OPENAI_API_KEY and HUGGINGFACEHUB_API_TOKEN are optional
]
missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing:
    missing_list = ", ".join(missing)
    raise EnvironmentError(
        f"Missing required env var(s): {missing_list}. "
        "Add them to a .env file or export in your shell."
    )

# Instantiate shared external tools once to avoid redundant network calls
TAVILY = TavilySearchResults()

# ──────────────────────────────────────────────────
# LLM factory
# ──────────────────────────────────────────────────

def build_llm(provider: str):
    """Return an LLM instance based on the selected provider."""
    provider = provider.lower()

    if provider == "openai":
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    if provider == "huggingface":
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise EnvironmentError(
                "HUGGINGFACEHUB_API_TOKEN not set. "
                "Create a personal access token, give it access to "
                "'TinyLlama/TinyLlama-1.1B-Chat-v1.0', and add it to your env."
            )
        return HuggingFaceEndpoint(
            repo_id="tiiuae/falcon-rw-1b",
            task="text-generation",
            huggingfacehub_api_token=hf_token,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
        )

    raise ValueError("provider must be 'openai' or 'huggingface'")

# ──────────────────────────────────────────────────
# LangGraph state and helper utilities
# ──────────────────────────────────────────────────

class ResearchState(TypedDict, total=False):
    query: str
    llm: Any
    research_notes: str
    answer: str


def _clean_lines(text: str) -> list[str]:
    """Return non-empty, de-bulleted lines from a block of text."""
    cleaned: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        # remove leading bullets or numbering
        cleaned.append(line.lstrip("1234567890-•*. \t"))
    return cleaned


async def _safe_llm_call(llm, prompt: str, default: str = "") -> str:
    """Safely invoke an LLM. On error, return *default* and log to Streamlit."""
    try:
        if isinstance(llm, HuggingFaceEndpoint):
            return await llm.ainvoke(prompt)  # HF returns raw string
        response = await llm.ainvoke(HumanMessage(content=prompt))
        return response.content
    except Exception as exc:
        st.error(f"LLM error: {exc}")
        return default

# ──────────────────────────────────────────────────
# LangGraph node definitions
# ──────────────────────────────────────────────────

async def researcher_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    query, llm = state["query"], state["llm"]

    # 1. Break the question into sub‑queries
    planning_prompt = (
        "Break the user's research question into 3-5 distinct web-search queries, "
        "one per line.\n"
        f"User question: {query}"
    )
    sub_qs_raw = await _safe_llm_call(llm, planning_prompt, query)
    sub_queries = _clean_lines(sub_qs_raw) or [query]
    sub_queries = sub_queries[:3]  # cap to avoid rate limits

    # 2. Perform web searches via Tavily
    docs: list[str] = []
    for q in sub_queries:
        try:
            results = TAVILY.invoke({"query": q, "max_results": 3})
            docs.extend(r["content"] for r in results)
        except Exception as exc:
            st.warning(f"Search error for '{q}': {exc}")

    # 3. Summarize findings
    if docs:
        summary_prompt = (
            "Summarize the following sources into concise bullet points:\n\n" +
            "\n\n".join(docs[:5])
        )
        notes = await _safe_llm_call(llm, summary_prompt, "No relevant information found.")
    else:
        notes = "No search results were found."

    return {"research_notes": notes}


async def drafter_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    llm, query = state["llm"], state["query"]
    research_notes = state.get("research_notes", "No research notes available.")

    draft_prompt = textwrap.dedent(
        f"""
        You are an expert analyst. Using **only** the provided notes, craft a clear,
        concise answer for the user.

        Notes:
        {research_notes}

        Question: {query}
        """
    )
    answer = await _safe_llm_call(llm, draft_prompt, "Unable to generate an answer.")
    return {"answer": answer}

# Build the graph once at startup
_graph = StateGraph(ResearchState)
_graph.add_node("research", researcher_agent)
_graph.add_node("draft", drafter_agent)
_graph.add_edge(START, "research")
_graph.add_edge("research", "draft")
_graph.add_edge("draft", END)
EXEC_GRAPH = _graph.compile()

# Public async wrapper used by the Streamlit UI
async def deep_research(query: str, provider: str) -> str:
    initial = {"query": query, "llm": build_llm(provider)}
    final_state = await EXEC_GRAPH.ainvoke(initial)
    return final_state.get("answer", "Research completed but no answer was generated.")

# ──────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────

st.set_page_config(page_title="Deep Research Agent", layout="wide")
st.title("🔍 Deep Research AI Agent")

with st.sidebar:
    provider = st.radio("LLM provider", ["openai", "huggingface"], index=0)

    # Key visibility for quick debugging
    for var in ["OPENAI_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "TAVILY_API_KEY"]:
        st.write(f"{var} found: {bool(os.getenv(var))}")

question = st.text_input("Enter your research question")

if st.button("Run") and question:
    with st.spinner("Researching..."):
        try:
            answer = asyncio.run(deep_research(question, provider))
            st.subheader("Answer")
            st.markdown(answer)
        except Exception as exc:
            st.error(f"Error: {exc}")
            if provider == "huggingface":
                st.info("Try using the OpenAI provider or check your Hugging Face token.")
