#!/usr/bin/env python3
"""
Coding Agent with Ansible and Python capabilities.

A LangGraph-based agent that can analyze and modify Ansible and Python codebases
with an approval-based workflow for all modifications.

Usage:
    python agent.py                          # Interactive mode
    python agent.py --query "your question"  # Non-interactive mode
    python agent.py --mop path/to/mop.docx   # Load MOP document
    python agent.py --mop mop.docx --query "implement step 1"
"""

import argparse
import os
import sys
import logging
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# Import all tools
from tools.file_ops import read_file, write_file, list_directory, file_exists
from tools.git_ops import (
    git_fetch_all,
    git_create_branch,
    git_checkout,
    git_add,
    git_commit,
    git_push,
    git_diff,
    git_status,
    get_current_branch,
)
from tools.mop_parser import read_mop_document
from tools.python_analysis import (
    analyze_python_file,
    find_python_pattern,
    find_functions,
    find_classes,
    find_imports,
)
from tools.python_coding import modify_python_code, add_import, add_function
from tools.ansible_analysis import (
    scan_ansible_project,
    analyze_playbook,
    analyze_role,
    find_tasks_using_module,
    get_variable_usage,
)
from tools.ansible_coding import modify_task, add_task, modify_variable, modify_yaml_file
from tools.shell_ops import run_shell_command, find_files, search_in_files
from tools.approval import (
    PendingChange,
    create_modification_plan,
    generate_unified_diff,
    format_changes_for_display,
    format_push_request,
)

# Import large file handling tools
from tools.large_file_tools import query_large_file, query_ansible_log, extract_ansible_metrics

# Load environment variables
load_dotenv()
repo_path = os.getenv("REPO_PATH")
os.chdir(repo_path)

# Configure logger
logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging to file.
    
    The log file is always created in the same directory as agent.py,
    regardless of the current working directory.
    """
    # Get the directory where agent.py is located
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(agent_dir, 'agent.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='a'
    )


# =============================================================================
# Agent State Definition
# =============================================================================

class AgentState(TypedDict):
    """State for the coding agent."""
    # Conversation messages
    messages: Annotated[list, add_messages]
    
    # Pending changes awaiting approval
    pending_changes: list[dict[str, Any]]
    
    # Approval flags
    awaiting_modification_approval: bool
    awaiting_push_approval: bool
    modification_approved: bool
    push_approved: bool
    
    # Pending push call (blocked git_push tool call awaiting approval)
    pending_push_call: dict | None
    
    # Git state
    current_branch: str | None
    original_branch: str | None
    branch_created: bool
    
    # MOP content (if loaded)
    mop_content: dict | None
    
    # AGENT.md content (if found in repo root)
    agent_md_content: str | None
    
    # Path configuration
    repo_path: str | None
    mop_path: str | None
    
    # User feedback for revisions
    user_feedback: str | None



# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an intelligent coding agent specialized in Ansible and Python development.

## Your Capabilities:
1. **File Operations**: Read, write, list, and delete files
2. **Git Operations**: Fetch, create branches (with 'agent-' prefix), commit, push, diff, status
3. **MOP Analysis**: Read and parse Method of Procedure documents (DOCX format, up to 90 pages)
4. **Python Analysis**: Analyze Python code structure using ast-grep (functions, classes, imports, patterns)
5. **Ansible Analysis**: Analyze Ansible projects, playbooks, roles, tasks, and variables
6. **Python Coding**: Modify Python code, add imports, add functions
7. **Ansible Coding**: Modify tasks, add tasks, update variables, modify YAML files
8. **Large File Analysis**: Analyze large files that exceed context limits using chunking
9. **Ansible Log Analysis**: Parse and analyze Ansible execution logs (failures, timeline, per-host)

## Workflow Rules (CRITICAL):

### Before Making ANY Code Modifications:
1. ALWAYS run `git_fetch_all` first to ensure you have the latest code
2. Create a new branch using `git_create_branch` - provide a descriptive name (it will be auto-prefixed with 'agent-')
3. The branch name should describe what changes you're making

### Modification Approval Process:
1. When you want to modify code, use the appropriate coding tools (modify_python_code, add_function, modify_task, etc.)
2. These tools return a diff showing the proposed changes - DO NOT apply them yet
3. Present the modification plan with ALL diffs to the user and wait for approval
4. If the user requests changes, incorporate their feedback and present the updated plan
5. Only after explicit approval ("approve", "yes", "proceed", etc.), apply the changes using write_file

### Push Approval Process:
1. After changes are applied and committed, ask for push approval
2. Show the branch name, commit summary, and files changed
3. If the user requests changes to the commit, amend and ask again
4. Only after explicit push approval, execute git_push

### When Reading MOPs:
1. Use read_mop_document to load the entire document
2. Analyze all procedures, steps, and requirements
3. Create a comprehensive modification plan covering ALL changes from the MOP
4. Present the full plan and wait for approval before making any changes

## Large File Handling (CRITICAL):
- For files that are TOO LARGE for read_file: Use `query_large_file` - it chunks and summarizes automatically
- For ANSIBLE LOG files: Use `query_ansible_log` for Q&A or `extract_ansible_metrics` for structured data
- When analyzing large codebases, use analysis tools (find_functions, find_classes) BEFORE reading full files

### Tool Selection for Large Files:
| Situation | Tool to Use |
|-----------|-------------|
| Normal code files | `read_file` |
| Any file too large, need to answer a question | `query_large_file` |
| Ansible log - what failed? | `query_ansible_log` or `extract_ansible_metrics` with 'failed_tasks' |
| Ansible log - execution timeline | `extract_ansible_metrics` with 'timeline' |
| Ansible log - specific host results | `extract_ansible_metrics` with 'by_host:<hostname>' |
| Ansible log - specific play results | `extract_ansible_metrics` with 'by_play:<playname>' |

## Response Format:
- Be clear and concise
- Always show diffs when proposing changes
- Explain what each change does and why
- Group related changes together
- Number the changes for easy reference
- If no specific tool exists for a task, use run_shell_command to execute shell commands

## MOP Document Handling:
When a MOP (Method of Procedure) document is loaded:
- Prioritize answering questions based on the MOP content
- Reference specific sections, steps, or procedures from the MOP
- If asked to implement changes, follow the MOP procedures exactly
- Cite the MOP section when explaining actions

## Result Filtering (CRITICAL):
When dealing with large result sets:
- If a directory listing returns more than 30 items, focus on the most relevant ones
- If a search returns more than 20 matches, summarize patterns and show only key examples
- If file content is too long (>500 lines), analyze in sections
- NEVER try to process all items when there are too many - filter and prioritize
- When results are filtered, clearly state what was included and what was skipped
- For large codebases, work incrementally: analyze structure first, then dive into specifics

Remember: NEVER apply changes without explicit user approval!"""


# =============================================================================
# Tool Definitions
# =============================================================================

# All available tools for the agent
ALL_TOOLS = [
    # File operations
    read_file,
    write_file,
    list_directory,
    file_exists,
    # Git operations
    git_fetch_all,
    git_create_branch,
    git_checkout,
    git_add,
    git_commit,
    git_push,
    git_diff,
    git_status,
    # MOP parsing
    read_mop_document,
    # Python analysis
    analyze_python_file,
    find_python_pattern,
    find_functions,
    find_classes,
    find_imports,
    # Python coding
    modify_python_code,
    add_import,
    add_function,
    # Ansible analysis
    scan_ansible_project,
    analyze_playbook,
    analyze_role,
    find_tasks_using_module,
    get_variable_usage,
    # Ansible coding
    modify_task,
    add_task,
    modify_variable,
    modify_yaml_file,
    # Shell operations (for when no specific tool exists)
    run_shell_command,
    find_files,
    search_in_files,
    # Large file handling tools
    query_large_file,
    query_ansible_log,
    extract_ansible_metrics,
]


# =============================================================================
# Agent Node Functions
# =============================================================================

def create_agent(model_name: str | None = None):
    """Create the LLM agent with tools bound."""
    llm_name = model_name or os.getenv("LLM_NAME", "anthropic/bedrock-sonnet-4-5")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    api_url = os.getenv("ANTHROPIC_API_URL")
    
    # Build kwargs for ChatLiteLLM
    llm_kwargs = {
        "model": llm_name,
        "api_key": api_key,
        "max_tokens": 4096,
        "drop_params": True,
    }
    
    # Add api_base if a proxy URL is configured
    if api_url:
        llm_kwargs["api_base"] = api_url
    
    llm = ChatLiteLLM(**llm_kwargs)
    
    return llm.bind_tools(ALL_TOOLS)


def agent_node(state: AgentState) -> dict:
    """Main agent node that processes messages and decides actions."""
    messages = state["messages"]
    
    # Add system prompt if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        system_content = SYSTEM_PROMPT
        
        # Add AGENT.md context if available (highest priority)
        agent_md_content = state.get("agent_md_content")
        if agent_md_content:
            system_content += build_agent_md_context(agent_md_content)
        
        # Add MOP context if available
        mop_content = state.get("mop_content")
        if mop_content:
            system_content += build_context_message(mop_content)
            
        messages = [SystemMessage(content=system_content)] + list(messages)
    
    # Check if we're awaiting approval
    if state.get("awaiting_modification_approval"):
        # Add context about pending changes
        pending = state.get("pending_changes", [])
        if pending:
            context = "\n\n[SYSTEM: You have pending changes awaiting user approval. "
            context += "Wait for the user to approve, reject, or request modifications.]\n"
            context += format_changes_for_display(pending)
            messages = messages + [SystemMessage(content=context)]
    
    if state.get("awaiting_push_approval"):
        branch = state.get("current_branch", "unknown")
        context = f"\n\n[SYSTEM: Changes have been applied. Awaiting push approval for branch '{branch}'.]\n"
        messages = messages + [SystemMessage(content=context)]
    
    # Get LLM response
    agent = create_agent()
    response = agent.invoke(messages)
    
    return {"messages": [response]}


def setup_node(state: AgentState) -> dict:
    """Initialize the agent environment (paths, MOP loading, AGENT.md loading)."""
    # Handle repo path
    repo_path = state.get("repo_path")
    if repo_path:
        current_path = os.getcwd()
        if os.path.abspath(repo_path) != current_path:
            try:
                os.chdir(repo_path)
                logger.info(f"Working directory set to: {repo_path}")
            except Exception as e:
                logger.error(f"Error changing directory to {repo_path}: {e}")
    
    updates = {}
    
    # Handle AGENT.md loading
    if not state.get("agent_md_content"):
        agent_md_content = load_agent_md(repo_path)
        if agent_md_content:
            updates["agent_md_content"] = agent_md_content
    
    # Handle MOP loading
    mop_path = state.get("mop_path")
    if mop_path and not state.get("mop_content"):
        try:
            mop_content = load_mop_content(mop_path)
            if mop_content:
                updates["mop_content"] = mop_content
                logger.info(f"Loaded MOP content from {mop_path}")
        except Exception as e:
            logger.error(f"Failed to load MOP from {mop_path}: {e}")
    
    return updates


def should_continue(state: AgentState) -> Literal["tools", "approval_check", "push_approval", "end"]:
    """Determine the next step based on current state."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # If the last message has tool calls, execute them
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    # Check if we need push approval (interrupt-based)
    if state.get("awaiting_push_approval"):
        return "push_approval"
    
    # Check if we need modification approval
    if state.get("awaiting_modification_approval"):
        return "approval_check"
    
    return "end"


def should_continue_after_push_approval(state: AgentState) -> Literal["execute_push", "agent"]:
    """After push approval node, determine if we execute push or return to agent."""
    if state.get("push_approved"):
        return "execute_push"
    return "agent"


def approval_check_node(state: AgentState) -> dict:
    """Handle approval checking and user feedback processing."""
    messages = state["messages"]
    last_user_message = None
    
    # Find the last user message
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.lower().strip()
            break
    
    if not last_user_message:
        return {}
    
    # Check for modification approval
    if state.get("awaiting_modification_approval"):
        if last_user_message in ["approve", "yes", "proceed", "ok", "go ahead", "lgtm"]:
            return {
                "modification_approved": True,
                "awaiting_modification_approval": False,
            }
        elif last_user_message in ["reject", "no", "cancel", "abort"]:
            return {
                "modification_approved": False,
                "awaiting_modification_approval": False,
                "pending_changes": [],
            }
        else:
            # User wants changes - store feedback
            return {
                "user_feedback": last_user_message,
            }
    
    # Check for push approval
    if state.get("awaiting_push_approval"):
        if last_user_message in ["push", "yes", "proceed", "ok", "go ahead"]:
            return {
                "push_approved": True,
                "awaiting_push_approval": False,
            }
        elif last_user_message in ["cancel", "no", "skip", "abort"]:
            return {
                "push_approved": False,
                "awaiting_push_approval": False,
            }
        else:
            # User wants changes
            return {
                "user_feedback": last_user_message,
            }
    
    return {}


def tools_node(state: AgentState) -> dict:
    """Execute tool calls and process results with verbose logging.
    
    IMPORTANT: This node blocks git_push calls if push_approved is False.
    Blocked pushes are stored in pending_push_call and awaiting_push_approval is set.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
    
    # Log tool calls being made
    logger.info("-" * 60)
    logger.info("TOOL EXECUTION")
    logger.info("-" * 60)
    
    for tc in last_message.tool_calls:
        logger.info(f"Tool: {tc['name']}")
        logger.info(f"  Input: {_truncate_str(str(tc.get('args', {})), 200)}")
    
    # Separate git_push from other tools if not approved
    allowed_tool_calls = []
    blocked_push_call = None
    blocked_messages = []
    
    for tc in last_message.tool_calls:
        if tc["name"] == "git_push" and not state.get("push_approved"):
            # Block this push - requires approval
            logger.warning(f"Blocking unapproved git_push call: {tc['id']}")
            blocked_push_call = tc
            blocked_messages.append(
                ToolMessage(
                    tool_call_id=tc["id"],
                    content="[BLOCKED] git_push requires explicit user approval. Please wait for user confirmation.",
                    name=tc["name"]
                )
            )
        else:
            allowed_tool_calls.append(tc)
    
    # Execute allowed tools
    result_messages = []
    if allowed_tool_calls:
        # Create a filtered AI message with only allowed tool calls
        filtered_message = AIMessage(
            content=last_message.content,
            tool_calls=allowed_tool_calls,
            id=last_message.id,
            additional_kwargs=last_message.additional_kwargs,
            response_metadata=last_message.response_metadata
        )
        
        temp_messages = list(messages)
        temp_messages[-1] = filtered_message
        temp_state = dict(state)
        temp_state["messages"] = temp_messages
        
        tool_node = ToolNode(ALL_TOOLS)
        node_result = tool_node.invoke(temp_state)
        result_messages = node_result.get("messages", [])
    
    # Combine results
    all_result_messages = result_messages + blocked_messages
    
    # Log tool results
    for msg in all_result_messages:
        if isinstance(msg, ToolMessage):
            tool_name = "unknown"
            for tc in last_message.tool_calls:
                if tc["id"] == msg.tool_call_id:
                    tool_name = tc["name"]
                    break
            
            output = msg.content if isinstance(msg.content, str) else str(msg.content)
            logger.info(f"Output from {tool_name}:")
            logger.info(f"  {_truncate_str(output, 500)}")
    
    logger.info("-" * 60)
    
    # Build updates
    updates: dict[str, Any] = {"messages": all_result_messages}
    
    # Store blocked push call if any
    if blocked_push_call:
        updates["pending_push_call"] = blocked_push_call
        updates["awaiting_push_approval"] = True
    
    # Consume push approval if it was used
    if state.get("push_approved"):
        for tc in allowed_tool_calls:
            if tc["name"] == "git_push":
                updates["push_approved"] = False
                updates["pending_push_call"] = None
                break
    
    # Check if any modification tools were called (for diff collection)
    modification_tools = {
        "modify_python_code", "add_import", "add_function",
        "modify_task", "add_task", "modify_variable", "modify_yaml_file"
    }
    
    pending_changes = list(state.get("pending_changes", []))
    
    for msg in result_messages:
        if isinstance(msg, ToolMessage):
            # Find tool name
            tool_name = None
            for tc in allowed_tool_calls:
                if tc["id"] == msg.tool_call_id:
                    tool_name = tc["name"]
                    break
            
            if tool_name in modification_tools:
                try:
                    import json
                    tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(tool_result, dict) and "diff" in tool_result:
                        pending_changes.append(tool_result)
                except Exception:
                    pass
    
    # Update pending changes if changed
    if pending_changes != state.get("pending_changes", []):
        updates["pending_changes"] = pending_changes
        updates["awaiting_modification_approval"] = True
    
    return updates


def push_approval_node(state: AgentState) -> dict:
    """Node that uses LangGraph interrupt to request push approval from user."""
    pending_push = state.get("pending_push_call")
    branch = state.get("current_branch", "unknown")
    
    push_request = format_push_request(branch, 1, [])
    
    logger.info("Push approval node triggered - calling interrupt")
    
    user_response = interrupt(push_request)
    
    logger.info(f"Interrupt resumed with user response: {user_response}")
    
    response_lower = str(user_response).lower().strip()
    
    if response_lower in ["push", "yes", "proceed", "ok", "go ahead", "y"]:
        return {
            "push_approved": True,
            "awaiting_push_approval": False,
        }
    elif response_lower in ["cancel", "no", "skip", "abort", "n"]:
        return {
            "push_approved": False,
            "awaiting_push_approval": False,
            "pending_push_call": None,
            "messages": [
                ToolMessage(
                    tool_call_id=pending_push["id"] if pending_push else "cancelled",
                    content="Push cancelled by user.",
                    name="git_push"
                )
            ]
        }
    else:
        return {
            "push_approved": False,
            "awaiting_push_approval": False,
            "pending_push_call": None,
            "user_feedback": user_response,
        }


def execute_push_node(state: AgentState) -> dict:
    """Execute the pending git_push after approval."""
    pending_push = state.get("pending_push_call")
    
    if not pending_push:
        logger.warning("execute_push_node called but no pending_push_call found")
        return {}
    
    logger.info(f"Executing approved git_push: {pending_push}")
    
    push_message = AIMessage(
        content="",
        tool_calls=[pending_push]
    )
    
    temp_state = dict(state)
    temp_state["messages"] = [push_message]
    
    tool_node = ToolNode([git_push])
    result = tool_node.invoke(temp_state)
    
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            logger.info(f"git_push result: {msg.content}")
    
    return {
        "messages": result.get("messages", []),
        "push_approved": False,
        "pending_push_call": None,
        "awaiting_push_approval": False,
    }


def _truncate_str(s: str, max_len: int) -> str:
    """Truncate a string to max_len characters."""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


# =============================================================================
# Graph Construction
# =============================================================================

def create_graph():
    """Create the LangGraph workflow with interrupt-based push approval."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("setup", setup_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("approval_check", approval_check_node)
    workflow.add_node("push_approval", push_approval_node)
    workflow.add_node("execute_push", execute_push_node)
    
    # Set entry point
    workflow.set_entry_point("setup")
    
    # Add edges
    workflow.add_edge("setup", "agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "approval_check": "approval_check",
            "push_approval": "push_approval",
            "end": END,
        }
    )
    
    workflow.add_edge("tools", "agent")
    workflow.add_edge("approval_check", "agent")
    
    workflow.add_conditional_edges(
        "push_approval",
        should_continue_after_push_approval,
        {
            "execute_push": "execute_push",
            "agent": "agent",
        }
    )
    workflow.add_edge("execute_push", "agent")
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# =============================================================================
# Agent Runner Functions
# =============================================================================

def load_mop_content(mop_path: str) -> dict | None:
    """Load MOP document content."""
    if not mop_path:
        return None
    
    from tools.mop_parser import read_mop_document
    logger.info(f"Loading MOP document: {mop_path}")
    result = read_mop_document.invoke({"path": mop_path})
    
    if isinstance(result, dict) and "error" in result:
        logger.error(f"Error loading MOP: {result['error']}")
        return None
    
    logger.info(f"MOP loaded successfully. {result.get('stats', {}).get('word_count', 0)} words.")
    return result


def load_agent_md(repo_path: str | None = None) -> str | None:
    """Load AGENT.md content from the repo root directory."""
    base_path = repo_path or os.getcwd()
    agent_md_path = os.path.join(base_path, "AGENT.md")
    
    if not os.path.isfile(agent_md_path):
        logger.info(f"AGENT.md not found at {agent_md_path}")
        return None
    
    try:
        with open(agent_md_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Loaded AGENT.md from {agent_md_path} ({len(content)} chars)")
        return content
    except Exception as e:
        logger.error(f"Error reading AGENT.md: {e}")
        return None


def create_initial_state(repo_path: str | None = None, mop_path: str | None = None) -> AgentState:
    """Create initial agent state."""
    return {
        "messages": [],
        "pending_changes": [],
        "awaiting_modification_approval": False,
        "awaiting_push_approval": False,
        "modification_approved": False,
        "push_approved": False,
        "pending_push_call": None,
        "current_branch": None,
        "original_branch": None,
        "branch_created": False,
        "mop_content": None,
        "agent_md_content": None,
        "user_feedback": None,
        "repo_path": repo_path,
        "mop_path": mop_path,
    }


def build_agent_md_context(agent_md_content: str | None) -> str:
    """Build context message with AGENT.md instructions."""
    if not agent_md_content:
        return ""
    
    context = "\n\n" + "=" * 60 + "\n"
    context += "PROJECT-SPECIFIC INSTRUCTIONS (AGENT.md)\n"
    context += "=" * 60 + "\n"
    context += "\nThe following instructions were found in AGENT.md in the repository root.\n"
    context += "These instructions MUST be followed for ALL interactions with this codebase.\n"
    context += "When conflicting with general guidelines, AGENT.md takes precedence.\n"
    context += "\n--- AGENT.md ---\n"
    context += agent_md_content[:15000]
    if len(agent_md_content) > 15000:
        context += "\n... (content truncated)"
    context += "\n--- END AGENT.md ---\n"
    context += "=" * 60 + "\n"
    
    return context


def build_context_message(mop_content: dict | None) -> str:
    """Build context message with MOP content if available."""
    if not mop_content:
        return ""
    
    context = "\n\n[MOP DOCUMENT LOADED]\n"
    context += f"Title: {mop_content.get('title', 'Untitled')}\n"
    context += f"Sections: {mop_content.get('stats', {}).get('section_count', 0)}\n"
    context += f"Tables: {mop_content.get('stats', {}).get('table_count', 0)}\n"
    context += "\n--- MOP FULL CONTENT ---\n"
    context += mop_content.get("full_text", "")[:30000]
    if len(mop_content.get("full_text", "")) > 30000:
        context += "\n... (content truncated)"
    context += "\n--- END MOP CONTENT ---\n"
    context += "\nPrioritize responses based on this MOP document when applicable.\n"
    
    return context


def run_single_query(query: str, repo_path: str | None = None, mop_path: str | None = None) -> str:
    """Run a single query in non-interactive mode."""
    import uuid
    from langgraph.errors import GraphInterrupt
    
    setup_logging()
    graph = create_graph()
    state = create_initial_state(repo_path, mop_path)
    state["messages"] = [HumanMessage(content=query)]
    
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    
    try:
        result = graph.invoke(state, config)
        
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content
        
        return "No response generated."
    
    except GraphInterrupt as e:
        interrupt_value = e.args[0] if e.args else "Push approval required"
        return f"PUSH APPROVAL REQUIRED\n\n{interrupt_value}\n\nNote: Cannot approve pushes in non-interactive mode. Run in interactive mode to approve."


def run_interactive(repo_path: str | None = None, mop_path: str | None = None):
    """Run the agent in interactive mode with interrupt-based push approval."""
    import uuid
    from langgraph.errors import GraphInterrupt
    
    setup_logging()
    graph = create_graph()
    state = create_initial_state(repo_path, mop_path)
    
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    
    print("=" * 60)
    print("Coding Agent with Ansible & Python Capabilities")
    print("=" * 60)
    print("Commands:")
    print("  - Type your request to interact with the agent")
    print("  - 'approve' to approve pending changes")
    print("  - 'reject' to reject pending changes")
    print("  - 'push' to approve pushing to remote")
    print("  - 'cancel' to cancel a push request")
    print("  - 'quit' or 'exit' to exit")
    print("=" * 60)
    print()
    
    interrupted = False
    
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            try:
                if interrupted:
                    result = graph.invoke(Command(resume=user_input), config)
                    interrupted = False
                else:
                    state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]
                    result = graph.invoke(state, config)
                
                state = result
                
                for msg in reversed(state["messages"]):
                    if isinstance(msg, AIMessage):
                        print(f"\nAgent: {msg.content}\n")
                        break
                
                if state.get("awaiting_modification_approval") and state.get("pending_changes"):
                    print("\n" + format_changes_for_display(state["pending_changes"]))
                    print("\nType 'approve' to apply changes or describe what you'd like to change.\n")
                    
            except GraphInterrupt as e:
                interrupted = True
                interrupt_value = e.args[0] if e.args else "Push approval required"
                print(f"\n{interrupt_value}")
                print("\nType 'push' to push, or 'cancel' to abort.\n")
                
            except Exception as e:
                error_msg = str(e)
                print(f"\nError: {error_msg}\n")
                import traceback
                traceback.print_exc()
                interrupted = False
    
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Coding Agent with Ansible & Python capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py                              # Interactive mode
  python agent.py --query "list ansible files" # Non-interactive query
  python agent.py --mop procedure.docx         # Load MOP, interactive mode
  python agent.py --mop procedure.docx --query "implement step 1"
        """
    )
    parser.add_argument(
        "--mop",
        type=str,
        help="Path to MOP (Method of Procedure) DOCX document"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Query to run in non-interactive mode"
    )
    parser.add_argument(
        "--repo",
        type=str,
        help="Override REPO_PATH from .env"
    )
    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    
    repo_path = args.repo or os.getenv("REPO_PATH")
    
    if args.query:
        response = run_single_query(args.query, repo_path, args.mop)
        print(response)
    else:
        run_interactive(repo_path, args.mop)
