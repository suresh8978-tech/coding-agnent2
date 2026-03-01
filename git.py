"""Git operation tools for the coding agent."""

import os
from typing import Optional
from langchain_core.tools import tool
from tools.utils import safe_tool
import git


def _get_repo(cwd: Optional[str] = None) -> git.Repo:
    """Get a git.Repo instance for the current repository.
    
    Args:
        cwd: Working directory for the repository.
        
    Returns:
        git.Repo instance.
    """
    repo_path = cwd or os.environ.get("REPO_PATH", ".")
    return git.Repo(repo_path)


@tool
@safe_tool
def git_fetch_all() -> str:
    """Fetch all branches from all remotes.
    
    Returns:
        Success message or error description.
    """
    try:
        repo = _get_repo()
        for remote in repo.remotes:
            remote.fetch()
        return "Successfully fetched all branches."
    except git.GitCommandError as e:
        return f"Error fetching: {e}"
    except Exception as e:
        return f"Error fetching: {str(e)}"


@tool
@safe_tool
def git_create_branch(name: str) -> str:
    """Create a new branch with 'agent-' prefix.
    
    Args:
        name: The branch name (will be prefixed with 'agent-' automatically).
        
    Returns:
        Success message or error description.
    """
    branch_name = name if name.startswith("agent-") else f"agent-{name}"
    branch_name = branch_name.replace(" ", "-").lower()
    branch_name = "".join(c for c in branch_name if c.isalnum() or c in "-_/")
    
    try:
        repo = _get_repo()
        new_branch = repo.create_head(branch_name)
        new_branch.checkout()
        return f"Successfully created and switched to branch '{branch_name}'."
    except git.GitCommandError as e:
        return f"Error creating branch: {e}"
    except Exception as e:
        return f"Error creating branch: {str(e)}"


@tool
@safe_tool
def git_checkout(branch: str) -> str:
    """Switch to an existing branch.
    
    Args:
        branch: The name of the branch to switch to.
        
    Returns:
        Success message or error description.
    """
    try:
        repo = _get_repo()
        repo.git.checkout(branch)
        return f"Successfully switched to branch '{branch}'."
    except git.GitCommandError as e:
        return f"Error switching branch: {e}"
    except Exception as e:
        return f"Error switching branch: {str(e)}"


@tool
@safe_tool
def git_add(files: str) -> str:
    """Stage files for commit.
    
    Args:
        files: Comma-separated list of file paths to stage, or '.' for all files.
        
    Returns:
        Success message or error description.
    """
    file_list = [f.strip() for f in files.split(",")]
    try:
        repo = _get_repo()
        repo.index.add(file_list)
        return f"Successfully staged files: {', '.join(file_list)}"
    except git.GitCommandError as e:
        return f"Error staging files: {e}"
    except Exception as e:
        return f"Error staging files: {str(e)}"


@tool
@safe_tool
def git_commit(message: str) -> str:
    """Commit staged changes.
    
    Args:
        message: The commit message.
        
    Returns:
        Success message or error description.
    """
    try:
        repo = _get_repo()
        commit = repo.index.commit(message)
        return f"Successfully committed changes.\n{commit.hexsha[:7]} {message}"
    except git.GitCommandError as e:
        return f"Error committing: {e}"
    except Exception as e:
        return f"Error committing: {str(e)}"


@tool
@safe_tool
def git_push(remote: str = "origin", branch: str = "") -> str:
    """Push commits to remote repository.
    
    Args:
        remote: The remote name (default: 'origin').
        branch: The branch to push. If empty, pushes current branch.
        
    Returns:
        Success message or error description.
    """
    try:
        repo = _get_repo()
        remote_obj = repo.remote(remote)
        if branch:
            push_info = remote_obj.push(branch)
        else:
            current_branch = repo.active_branch.name
            push_info = remote_obj.push(f"{current_branch}:{current_branch}", set_upstream=True)
        
        # Check push results
        for info in push_info:
            if info.flags & info.ERROR:
                return f"Error pushing: {info.summary}"
        
        return f"Successfully pushed to {remote}."
    except git.GitCommandError as e:
        return f"Error pushing: {e}"
    except Exception as e:
        return f"Error pushing: {str(e)}"


@tool
@safe_tool
def git_diff(staged: bool = False) -> str:
    """Get the diff of changes.
    
    Args:
        staged: If True, show diff of staged changes. Otherwise show unstaged.
        
    Returns:
        The diff output or message if no changes.
    """
    try:
        repo = _get_repo()
        if staged:
            diff = repo.git.diff("--staged")
        else:
            diff = repo.git.diff()
        return diff if diff else "No changes to show."
    except git.GitCommandError as e:
        return f"Error getting diff: {e}"
    except Exception as e:
        return f"Error getting diff: {str(e)}"


@tool
@safe_tool
def git_status() -> str:
    """Get the current repository status.
    
    Returns:
        The git status output.
    """
    try:
        repo = _get_repo()
        status = repo.git.status("--short")
        return status if status else "Working tree clean, no changes."
    except git.GitCommandError as e:
        return f"Error getting status: {e}"
    except Exception as e:
        return f"Error getting status: {str(e)}"


def get_current_branch() -> str:
    """Get the name of the current branch.
    
    Returns:
        The current branch name or error message.
    """
    try:
        repo = _get_repo()
        return repo.active_branch.name
    except git.GitCommandError as e:
        return f"Error getting current branch: {e}"
    except Exception as e:
        return f"Error getting current branch: {str(e)}"
