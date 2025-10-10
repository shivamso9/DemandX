# pluginfactoryapp/services/agents/git_agent.py

import os
import subprocess
import datetime
from flask import current_app

def _run_git_command(command, check=True):
    """Helper to run a git command in the configured repository path."""
    repo_path = current_app.config['GIT_REPO_PATH']
    result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True, encoding='utf-8')
    if check and result.returncode != 0:
        error_message = f"Git command failed: {' '.join(command)}\nError: {result.stderr}"
        raise Exception(error_message)
    return result

def push_to_git_repository(plugin_name, repo_code, tenant_code, test_code):
    """
    Writes plugin files to the local git repository, commits, and pushes them.
    """
    repo_path = current_app.config['GIT_REPO_PATH']
    if not repo_path or not os.path.isdir(repo_path) or not os.path.isdir(os.path.join(repo_path, '.git')):
        raise FileNotFoundError(f"Configuration error: GIT_REPO_PATH ('{repo_path}') is not a valid Git repository.")

    # Define paths and write files
    repo_file_path = os.path.join(repo_path, 'helpers', f"{plugin_name}-Repo.py")
    tenant_file_path = os.path.join(repo_path, 'plugin_scripts', f"{plugin_name}-Tenant.py")
    test_file_path = os.path.join(repo_path, 'tests', f"test_{plugin_name}.py")

    os.makedirs(os.path.dirname(repo_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(tenant_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

    with open(repo_file_path, "w", encoding='utf-8') as f: f.write(repo_code)
    with open(tenant_file_path, "w", encoding='utf-8') as f: f.write(tenant_code)
    with open(test_file_path, "w", encoding='utf-8') as f: f.write(test_code)

    # Git operations
    branch_name = "2025.10.01"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"({plugin_name}): Generate plugin via Agentic Factory on {timestamp}"

    current_branch_result = _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], check=False)
    if current_branch_result.stdout.strip() != branch_name:
        _run_git_command(["git", "checkout", "-B", branch_name])

    _run_git_command(["git", "add", "."])
    
    status_result = _run_git_command(["git", "status", "--porcelain"])
    if status_result.stdout:
        print(f"✅ Changes detected. Committing with message: '{commit_message}'")
        _run_git_command(["git", "commit", "-m", commit_message])
    else:
        print("✅ No changes to commit. Working tree clean.")
        return {
            "success": True, 
            "message": "No changes detected; push not required.",
            "branch": branch_name,
            "commit_message" : "N/A"
        }

    _run_git_command(["git", "fetch", "origin", branch_name])
    merge_result = _run_git_command(["git", "merge", f"origin/{branch_name}"], check=False)
    
    if merge_result.returncode != 0:
        raise ConnectionRefusedError(f"Merge conflicts detected. Please resolve them manually in the repository. Details: {merge_result.stderr}")
    
    _run_git_command(["git", "push", "origin", branch_name])
    
    return {
        "success": True, 
        "message": "Successfully committed and pushed changes.",
        "branch": branch_name,
        "commit_message": commit_message
    }