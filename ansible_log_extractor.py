"""
Ansible Log Extractor - Extract structured data from Ansible log chunks.

Place this file at: tools/ansible_log_extractor.py
"""

from typing import List, Dict, Any

from tools.ansible_log_chunker import AnsibleLogChunker


class AnsibleLogExtractor:
    """Extract and filter structured data from Ansible log chunks."""

    def __init__(self, chunker: AnsibleLogChunker | None = None):
        self.chunker = chunker or AnsibleLogChunker()

    def chunk_log_content(self, content: str) -> List[Dict[str, Any]]:
        """Delegate to chunker."""
        return self.chunker.chunk_log_content(content)

    def extract_failed_tasks_only(self, content: str) -> List[Dict[str, Any]]:
        """Get only failed tasks for quick error analysis."""
        chunks = self.chunk_log_content(content)
        return [c for c in chunks if c.get("status") == "failed"]

    def extract_by_host(self, content: str, host: str) -> List[Dict[str, Any]]:
        """Get all tasks affecting a specific host."""
        chunks = self.chunk_log_content(content)
        return [
            c
            for c in chunks
            if any(r["host"] == host for r in c.get("results", []))
        ]

    def extract_by_play(self, content: str, play_name: str) -> List[Dict[str, Any]]:
        """Get all tasks from a specific play."""
        chunks = self.chunk_log_content(content)
        return [c for c in chunks if c.get("play") == play_name]

    def get_execution_timeline(self, content: str) -> List[Dict[str, Any]]:
        """Create chronological timeline of execution."""
        chunks = self.chunk_log_content(content)
        timeline = []

        for i, chunk in enumerate(chunks):
            if chunk["type"] == "task":
                timeline.append(
                    {
                        "step": i + 1,
                        "play": chunk["play"],
                        "task": chunk["task"],
                        "status": chunk["status"],
                        "summary": chunk["summary"],
                    }
                )

        return timeline
