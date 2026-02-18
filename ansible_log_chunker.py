"""
Ansible Log Chunker - Parse Ansible logs into structured chunks by play/task.

Place this file at: tools/ansible_log_chunker.py
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class AnsibleTask:
    """Represents a single Ansible task."""

    play_name: str
    task_name: str
    results: List[Dict[str, Any]]
    status: str  # 'ok', 'changed', 'failed', 'skipped'
    raw_output: str


class AnsibleLogChunker:
    """Chunk Ansible logs by logical execution units."""

    def __init__(self):
        self.play_pattern = re.compile(r"^PLAY \[(.*?)\]")
        self.task_pattern = re.compile(r"^TASK \[(.*?)\]")
        self.recap_pattern = re.compile(r"^PLAY RECAP")
        self.host_result_pattern = re.compile(
            r"^(ok|changed|failed|skipping|fatal|unreachable): \[(.*?)\]"
        )

    def chunk_log_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Chunk Ansible log into plays and tasks."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return self.chunk_log_content(content)

    def chunk_log_content(self, content: str) -> List[Dict[str, Any]]:
        """Parse log content into structured chunks."""
        lines = content.split("\n")
        chunks = []

        current_play = None
        current_task = None
        current_block = []

        for i, line in enumerate(lines):
            play_match = self.play_pattern.match(line)
            if play_match:
                if current_task and current_block:
                    chunks.append(
                        self._create_task_chunk(
                            current_play, current_task, current_block
                        )
                    )
                    current_block = []

                current_play = play_match.group(1)
                current_task = None
                continue

            task_match = self.task_pattern.match(line)
            if task_match:
                if current_task and current_block:
                    chunks.append(
                        self._create_task_chunk(
                            current_play, current_task, current_block
                        )
                    )
                    current_block = []

                current_task = task_match.group(1)
                continue

            if self.recap_pattern.match(line):
                if current_task and current_block:
                    chunks.append(
                        self._create_task_chunk(
                            current_play, current_task, current_block
                        )
                    )

                recap_lines = lines[i:]
                chunks.append(
                    {
                        "type": "recap",
                        "content": "\n".join(recap_lines),
                        "summary": "Execution summary",
                    }
                )
                break

            if current_task:
                current_block.append(line)

        return chunks

    def _create_task_chunk(
        self, play_name: str, task_name: str, lines: List[str]
    ) -> Dict[str, Any]:
        """Create a structured chunk for a task."""
        content = "\n".join(lines)

        results = []
        status = "ok"

        for line in lines:
            host_match = self.host_result_pattern.match(line)
            if host_match:
                result_status = host_match.group(1)
                host = host_match.group(2)

                results.append({"host": host, "status": result_status})

                if result_status in ["failed", "fatal"]:
                    status = "failed"
                elif result_status == "changed" and status != "failed":
                    status = "changed"

        error_msg = None
        if status == "failed":
            error_match = re.search(r'"msg": "(.*?)"', content)
            if error_match:
                error_msg = error_match.group(1)

        return {
            "type": "task",
            "play": play_name,
            "task": task_name,
            "status": status,
            "results": results,
            "error_message": error_msg,
            "content": content,
            "summary": self._create_task_summary(
                play_name, task_name, status, results, error_msg
            ),
        }

    def _create_task_summary(
        self,
        play: str,
        task: str,
        status: str,
        results: List[Dict],
        error: str,
    ) -> str:
        """Create human-readable summary."""
        summary = f"[{status.upper()}] {play} → {task}"

        if results:
            host_summary = ", ".join(
                [f"{r['host']}:{r['status']}" for r in results[:3]]
            )
            summary += f" ({host_summary})"

        if error:
            summary += f" | Error: {error[:50]}"

        return summary
