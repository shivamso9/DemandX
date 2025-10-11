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
        # Provide stdout and stderr for better debugging
        error_message = (
            f"Git command failed: {' '.join(command)}\n"
            f"Exit Code: {result.returncode}\n"
            f"Stdout: {result.stdout.strip()}\n"
            f"Stderr: {result.stderr.strip()}"
        )
        raise Exception(error_message)
    return result

def push_to_git_repository(plugin_name, repo_code, tenant_code, test_code):
    """
    Writes plugin files to the local git repository, commits, and pushes them.
    """
    repo_path = current_app.config['GIT_REPO_PATH']
    if not repo_path or not os.path.isdir(repo_path) or not os.path.isdir(os.path.join(repo_path, '.git')):
        raise FileNotFoundError(f"Configuration error: GIT_REPO_PATH ('{repo_path}') is not a valid Git repository.")
    # 1. Define branch and commit message
    branch_name = current_app.config['GIT_BRANCH']# This should ideally be dynamic
    commit_message = f"feat({plugin_name}): Generate plugin via Agentic Factory"

    try:
        # 2. Get the latest state from the remote repository
        print("Step 1: Fetching latest updates from origin...")
        _run_git_command(["git", "fetch", "origin"])

        # 3. Switch to the target branch
        print(f"Step 2: Checking out branch '{branch_name}'...")
        _run_git_command(["git", "checkout", branch_name])

        # 4. **THE FIX**: Pull latest changes to sync the local branch with the remote
        # This prevents non-fast-forward errors. Using --rebase avoids merge commits.
        print(f"Step 3: Pulling latest changes from 'origin/{branch_name}'...")
        _run_git_command(["git", "pull", "--rebase", "origin", branch_name])

    except Exception as e:
        # If any of the setup steps fail, it's a critical error.
        raise ConnectionRefusedError(f"Failed to synchronize with Git remote before making changes. Please check repository access and branch status. Details: {e}")

    # 5. Now that the branch is clean and synced, write the new files
    print("Step 4: Writing new plugin files...")
    repo_file_path = os.path.join(repo_path, 'helpers', f"{plugin_name}-Repo.py")
    tenant_file_path = os.path.join(repo_path, 'plugin_scripts', f"{plugin_name}-Tenant.py")
    test_file_path = os.path.join(repo_path, 'tests', f"test_{plugin_name}.py")

    os.makedirs(os.path.dirname(repo_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(tenant_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

    with open(repo_file_path, "w", encoding='utf-8') as f: f.write(repo_code)
    with open(tenant_file_path, "w", encoding='utf-8') as f: f.write(tenant_code)
    with open(test_file_path, "w", encoding='utf-8') as f: f.write(test_code)

    # 6. Add and commit the new files
    print("Step 5: Committing new files...")
    _run_git_command(["git", "add", "."])
    
    # Check if there are any changes to commit
    status_result = _run_git_command(["git", "status", "--porcelain"])
    if not status_result.stdout:
        print("✅ No new changes to commit. The files may be identical to what's already in the repository.")
        return {
            "success": True, 
            "message": "No changes detected; push not required.",
            "branch": branch_name,
            "commit_message" : "N/A"
        }
    
    _run_git_command(["git", "commit", "-m", commit_message])

    # 7. Push the new commit to the remote repository
    print(f"Step 6: Pushing changes to 'origin/{branch_name}'...")
    _run_git_command(["git", "push", "origin", branch_name])
    
    print("✅ Successfully committed and pushed changes.")
    return {
        "success": True, 
        "message": "Successfully committed and pushed changes.",
        "branch": branch_name,
        "commit_message": commit_message
    }