"""
Ansible Log Reader - LLM-based Q&A over Ansible execution logs.

Place this file at: tools/ansible_log_reader.py
"""

import os
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate

from tools.ansible_log_chunker import AnsibleLogChunker


# ============================================================
# STATE
# ============================================================

class LogState(TypedDict):
    file_path: str
    question: str
    chunks: List[Dict[str, Any]]
    summaries: List[str]
    answer: str


# ============================================================
# READER
# ============================================================

class AnsibleLogReader:
    """LLM-powered Q&A over Ansible logs."""

    def __init__(self, model: str | None = None):
        llm_name = model or os.getenv("LLM_NAME", "anthropic/bedrock-sonnet-4-5")
        api_key = os.getenv("LITELLM_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
        api_url = os.getenv("LITELLM_API_BASE", os.getenv("ANTHROPIC_API_URL", ""))

        llm_kwargs = {
            "model": llm_name,
            "api_key": api_key,
            "max_tokens": 4096,
            "drop_params": True,
        }
        if api_url:
            llm_kwargs["api_base"] = api_url

        self.llm = ChatLiteLLM(**llm_kwargs)
        self.chunker = AnsibleLogChunker()
        self.graph = self._build_graph()

    # --------------------------------------------------------
    # NODE 1: Parse log
    # --------------------------------------------------------
    def parse_log(self, state: LogState) -> LogState:
        """Parse the Ansible log into structured chunks."""
        print(f"📂 Parsing Ansible log: {state['file_path']}")

        chunks = self.chunker.chunk_log_file(state["file_path"])
        summaries = [c.get("summary", "") for c in chunks]

        print(f"   Found {len(chunks)} chunks ({sum(1 for c in chunks if c.get('status') == 'failed')} failed)")

        return {
            **state,
            "chunks": chunks,
            "summaries": summaries,
        }

    # --------------------------------------------------------
    # NODE 2: Answer question
    # --------------------------------------------------------
    def answer_question(self, state: LogState) -> LogState:
        """Use LLM to answer the question based on parsed chunks."""
        print(f"💭 Answering: {state['question']}")

        summaries_text = "\n".join(state["summaries"])

        question_lower = state["question"].lower()

        relevant_chunks = []
        if "fail" in question_lower or "error" in question_lower:
            relevant_chunks = [c for c in state["chunks"] if c.get("status") == "failed"]
        elif "change" in question_lower:
            relevant_chunks = [c for c in state["chunks"] if c.get("status") == "changed"]
        elif "summary" in question_lower or "recap" in question_lower:
            relevant_chunks = [c for c in state["chunks"] if c.get("type") == "recap"]

        if not relevant_chunks:
            relevant_chunks = state["chunks"][:10]

        context_parts = []
        for chunk in relevant_chunks[:5]:
            context_parts.append(
                f"[{chunk.get('type', 'unknown').upper()}] "
                f"{chunk.get('summary', '')}\n"
                f"{chunk.get('content', '')[:1000]}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = ChatPromptTemplate.from_template(
            """You are analyzing an Ansible execution log.

Overall execution summary:
{summaries}

Relevant sections:
{context}

Question: {question}

Provide a clear, concise answer:"""
        )

        answer = self.llm.invoke(
            prompt.format(
                summaries=summaries_text[:3000],
                context=context[:8000],
                question=state["question"],
            )
        ).content

        print("✓ Answer generated")

        return {**state, "answer": answer}

    # --------------------------------------------------------
    # Build Graph
    # --------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        """Create the workflow."""
        workflow = StateGraph(LogState)

        workflow.add_node("parse", self.parse_log)
        workflow.add_node("answer", self.answer_question)

        workflow.set_entry_point("parse")
        workflow.add_edge("parse", "answer")
        workflow.add_edge("answer", END)

        return workflow.compile()

    # --------------------------------------------------------
    # Public Interface
    # --------------------------------------------------------
    def query(self, file_path: str, question: str) -> str:
        """Ask a question about an Ansible log file."""
        result = self.graph.invoke(
            {
                "file_path": file_path,
                "question": question,
                "chunks": [],
                "summaries": [],
                "answer": "",
            }
        )
        return result["answer"]
