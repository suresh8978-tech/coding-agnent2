"""
Large File Handling Tools - LangChain tool wrappers for the agent.

These tools are imported by agent.py and added to ALL_TOOLS.

Place this file at: tools/large_file_tools.py

Dependencies (these must also be in tools/):
    - tools/large_file_reader.py    → AgenticFileReader
    - tools/ansible_log_reader.py   → AnsibleLogReader
    - tools/ansible_log_chunker.py  → AnsibleLogChunker
    - tools/ansible_log_extractor.py → AnsibleLogExtractor
"""

import json
from typing import Annotated
from langchain_core.tools import tool

from tools.large_file_reader import AgenticFileReader
from tools.ansible_log_reader import AnsibleLogReader
from tools.ansible_log_chunker import AnsibleLogChunker
from tools.ansible_log_extractor import AnsibleLogExtractor


@tool
def query_large_file(
    file_path: Annotated[str, "Path to the large file to analyze"],
    question: Annotated[str, "Question to ask about the file content"],
) -> str:
    """Analyze a large file (that might exceed context limits) by chunking and summarizing it to answer a question.

    Use this tool when:
    - You need to read a file that is too large for standard read_file
    - You need to ask complex questions about a large document
    - Standard analysis fails due to token limits
    """
    try:
        reader = AgenticFileReader()
        return reader.query(file_path, question)
    except Exception as e:
        return f"Error analyzing large file: {str(e)}"


@tool
def query_ansible_log(
    file_path: Annotated[str, "Path to the Ansible log file"],
    question: Annotated[
        str,
        "Question about the Ansible execution (e.g. 'what failed?', 'summary')",
    ],
) -> str:
    """Analyze an Ansible log file to answer questions about the execution.

    Use this tool for:
    - determining why a playbook failed
    - finding which hosts changed
    - getting a summary of the execution
    - analyzing specific tasks or errors
    """
    try:
        reader = AnsibleLogReader()
        return reader.query(file_path, question)
    except Exception as e:
        return f"Error analyzing Ansible log: {str(e)}"


@tool
def extract_ansible_metrics(
    file_path: Annotated[str, "Path to the Ansible log file"],
    extraction_type: Annotated[
        str,
        "Type of extraction: 'failed_tasks', 'timeline', 'by_host:<hostname>', 'by_play:<playname>'",
    ],
) -> str:
    """Extract specific structured data from an Ansible log file.

    This tool does NOT use LLM calls - it parses the log directly.

    Args:
        file_path: Path to the Ansible log file.
        extraction_type: What to extract. Options:
            - 'failed_tasks': Get all failed tasks with error messages
            - 'timeline': Get chronological execution timeline
            - 'by_host:<hostname>': Get all tasks for a specific host
            - 'by_play:<playname>': Get all tasks from a specific play
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunker = AnsibleLogChunker()
        extractor = AnsibleLogExtractor(chunker)

        if extraction_type == "failed_tasks":
            result = extractor.extract_failed_tasks_only(content)
        elif extraction_type == "timeline":
            result = extractor.get_execution_timeline(content)
        elif extraction_type.startswith("by_host:"):
            host = extraction_type.split(":", 1)[1]
            result = extractor.extract_by_host(content, host)
        elif extraction_type.startswith("by_play:"):
            play = extraction_type.split(":", 1)[1]
            result = extractor.extract_by_play(content, play)
        else:
            return (
                f"Unknown extraction type: {extraction_type}. "
                f"Valid options: 'failed_tasks', 'timeline', 'by_host:<hostname>', 'by_play:<playname>'"
            )

        if not result:
            return f"No results found for extraction type '{extraction_type}'."

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"Error extracting metrics: {str(e)}"
