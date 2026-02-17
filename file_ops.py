"""
File operations tools for the coding agent.

NOTE: Only read_file is shown here with the new line-range and large-file
handling modifications. Copy this read_file function into your existing
tools/file_ops.py, replacing the old read_file. Keep your other functions
(write_file, list_directory, file_exists) unchanged.
"""

import os
from langchain_core.tools import tool


@tool
def read_file(
    path: str,
    start_line: int = None,
    end_line: int = None,
) -> str:
    """Read file content, optionally a specific line range.

    For large files (>300 lines), automatically returns a summary with head/tail
    unless a specific line range is requested. This prevents token overflow.

    Args:
        path: Path to the file to read.
        start_line: Optional start line (1-indexed). If omitted, reads from beginning.
        end_line: Optional end line (1-indexed, inclusive). If omitted, reads to end.

    Returns:
        File content as a string, with metadata about line counts and truncation.

    Examples:
        read_file("main.py")                        # Full file (auto-truncated if large)
        read_file("main.py", start_line=1, end_line=50)   # Lines 1-50
        read_file("main.py", start_line=100)         # Line 100 to end
    """
    try:
        # Resolve path
        resolved_path = os.path.abspath(path)

        if not os.path.isfile(resolved_path):
            return f"Error: File not found: {path}"

        # Check file size first (skip binary/huge files)
        file_size = os.path.getsize(resolved_path)
        if file_size > 5_000_000:  # 5MB
            return (
                f"Error: File is too large ({file_size:,} bytes / {file_size // 1024:,} KB). "
                f"This is likely a binary or data file. Use run_shell_command to inspect it "
                f"(e.g., 'head -100 {path}' or 'file {path}')."
            )

        with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)

        # --- Line range mode ---
        if start_line is not None or end_line is not None:
            s = max((start_line or 1) - 1, 0)
            e = min(end_line or total_lines, total_lines)

            if s >= total_lines:
                return f"Error: start_line {start_line} exceeds file length ({total_lines} lines)."

            selected = lines[s:e]
            content = "".join(selected)

            # Even in range mode, cap at a reasonable size
            if len(content) > 50000:
                content = (
                    content[:48000]
                    + f"\n\n... [TRUNCATED at 48,000 chars. Requested range has {len(''.join(selected)):,} chars. "
                    + "Use a smaller line range.] ...\n"
                )

            return (
                f"[Lines {s + 1}-{min(e, total_lines)} of {total_lines} total lines]\n"
                f"{content}"
            )

        # --- Full file mode ---
        content = "".join(lines)

        # Small files: return as-is
        if total_lines <= 300 and len(content) <= 12000:
            return f"[{total_lines} lines, {len(content):,} chars]\n{content}"

        # Medium files (300-800 lines): return with a note
        if total_lines <= 800 and len(content) <= 30000:
            return (
                f"[{total_lines} lines, {len(content):,} chars - consider using line ranges for targeted reading]\n"
                f"{content}"
            )

        # Large files: return structured summary + head + tail
        head_lines = 100
        tail_lines = 50

        head = "".join(lines[:head_lines])
        tail = "".join(lines[-tail_lines:])

        # Try to extract a structural overview for code files
        structure_info = ""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".py", ".yaml", ".yml", ".json", ".js", ".ts", ".java", ".go", ".rb"):
            structure_info = _extract_structure_overview(lines, ext)

        result = (
            f"[LARGE FILE: {total_lines} lines, {len(content):,} chars]\n"
            f"[Showing first {head_lines} and last {tail_lines} lines. "
            f"Use start_line/end_line for specific sections.]\n"
        )

        if structure_info:
            result += f"\n--- STRUCTURE OVERVIEW ---\n{structure_info}\n"

        result += (
            f"\n--- FIRST {head_lines} LINES ---\n"
            f"{head}\n"
            f"\n--- LAST {tail_lines} LINES ---\n"
            f"{tail}\n"
        )

        return result

    except UnicodeDecodeError:
        return f"Error: File appears to be binary or uses an unsupported encoding: {path}"
    except PermissionError:
        return f"Error: Permission denied reading file: {path}"
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"


def _extract_structure_overview(lines: list[str], ext: str) -> str:
    """Extract a quick structural overview of a code file.

    Returns a compact summary of classes, functions, and key sections
    to help the agent know which line ranges to request.
    """
    overview_items = []

    if ext == ".py":
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("class ") and ":" in stripped:
                overview_items.append(f"  Line {i}: {stripped.split(':')[0].strip()}")
            elif stripped.startswith("def ") and ":" in stripped:
                # Only top-level and class-level functions
                indent = len(line) - len(line.lstrip())
                if indent <= 4:
                    overview_items.append(f"  Line {i}: {stripped.split(':')[0].strip()}")
            elif stripped.startswith("# ===") or stripped.startswith("# ---"):
                # Section headers
                if i + 1 <= len(lines):
                    next_line = lines[i].strip() if i < len(lines) else ""
                    if next_line.startswith("#"):
                        overview_items.append(f"  Line {i}: [Section] {next_line.lstrip('#').strip()}")

    elif ext in (".yaml", ".yml"):
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            # Top-level keys only
            if indent == 0 and stripped and not stripped.startswith("#") and ":" in stripped:
                key = stripped.split(":")[0].strip()
                if key.startswith("- "):
                    key = key[2:]
                overview_items.append(f"  Line {i}: {key}")

    if not overview_items:
        return ""

    # Cap the overview
    if len(overview_items) > 40:
        shown = overview_items[:30]
        shown.append(f"  ... and {len(overview_items) - 30} more definitions")
        overview_items = shown

    return "\n".join(overview_items)
