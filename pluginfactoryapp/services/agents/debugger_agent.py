# pluginfactoryapp/services/agents/debugger_agent.py

import os
import tempfile
import shutil
import subprocess
import time
import re
from flask import current_app
from ... import prompts
from .. import vertex_service
from ...utils import clean_code_output, parse_test_results

def execute_and_debug_code(test_code, repo_code, tenant_code, proposed_test_cases, logic_summary):
    """
    Executes tests in a temporary directory and enters a debugging loop
    if they fail, using an AI model to propose fixes.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        pytest_path = shutil.which("pytest")
        if not pytest_path:
            raise FileNotFoundError("`pytest` command not found on the system.")

        current_test_code = clean_code_output(test_code)
        current_repo_code = clean_code_output(repo_code)
        current_tenant_code = clean_code_output(tenant_code)
        
        max_attempts = current_app.config['MAX_DEBUG_ATTEMPTS']
        timeout = current_app.config['PYTEST_TIMEOUT']
        full_log = ""
        attempt_details = []
        last_result = {}

        for attempt in range(max_attempts):
            # Setup files for the current attempt
            helpers_dir = os.path.join(temp_dir, "helpers")
            os.makedirs(helpers_dir, exist_ok=True)
            with open(os.path.join(helpers_dir, "__init__.py"), "w") as f: f.write("")
            with open(os.path.join(temp_dir, "test_plugin.py"), "w") as f: f.write(current_test_code)
            with open(os.path.join(helpers_dir, "plugin_module.py"), "w") as f: f.write(current_repo_code)
            with open(os.path.join(temp_dir, "plugin_tenant.py"), "w") as f: f.write(current_tenant_code)

            # Run pytest
            result = subprocess.run([pytest_path, "-v", "test_plugin.py"], cwd=temp_dir, capture_output=True, text=True, timeout=timeout)
            
            # Smart retry for flaky tests
            if result.returncode != 0:
                full_log += f"--- Attempt {attempt + 1}/{max_attempts} (Initial Run) ---\n{result.stdout + result.stderr}\n\n"
                full_log += "⚠️ Initial run failed. Performing one smart retry for potential flakiness...\n"
                time.sleep(1)
                result = subprocess.run([pytest_path, "-v", "test_plugin.py"], cwd=temp_dir, capture_output=True, text=True, timeout=timeout)
                if result.returncode == 0:
                    full_log += "✅ Passed on smart retry! Likely a transient issue.\n"
            
            attempt_log = result.stdout + result.stderr
            if "smart retry" not in full_log:
                full_log += f"--- Attempt {attempt + 1}/{max_attempts} ---\n{attempt_log}\n\n"
            
            passedTests, failedTests = parse_test_results(attempt_log, proposed_test_cases)
            attempt_details.append({"attempt": attempt + 1, "passed": passedTests, "failed": failedTests})
            last_result = {"passedTests": passedTests, "failedTests": failedTests}

            # If tests pass, break the loop
            if result.returncode == 0:
                full_log += "✅ Tests Passed. Exiting correction loop.\n"
                break

            full_log += f"❌ Tests Failed on Attempt {attempt + 1}. Activating Debugger Agent.\n"
            
            # If more attempts are left, try to fix
            if attempt < max_attempts - 1:
                error_log = result.stdout + result.stderr
                
                # Triage the error
                triage_prompt = prompts.get_triage_prompt(error_log)
                triage_response = vertex_service.generate_content(triage_prompt)
                triage_result = clean_code_output(triage_response.strip())
                full_log += f"🤖 Triage determined the issue is a **{triage_result}**. Activating agent...\n\n"

                if "TEST_ERROR" in triage_result:
                    prompt = prompts.get_test_corrector_prompt(current_test_code, error_log)
                    response_text = vertex_service.generate_content(prompt)
                    current_test_code = clean_code_output(response_text)
                else:  # PLUGIN_ERROR
                    previous_attempts_summary = ""
                    if attempt > 0:
                        last_failed_names = [f['name'] for f in attempt_details[-1]['failed']]
                        previous_attempts_summary = f"---\n**PREVIOUS FAILED ATTEMPT:**\nA fix was applied, but these tests still failed: {', '.join(last_failed_names)}.\nPlease provide a DIFFERENT and BETTER solution.\n---"
                    
                    prompt = prompts.get_debug_prompt(logic_summary, current_repo_code, error_log, previous_attempts_summary)
                    response_text = vertex_service.generate_content(prompt)
                    repo_match = re.search(r'### REPO FILE ###\s*(.*)', response_text, re.S)
                    
                    if repo_match:
                        current_repo_code = clean_code_output(repo_match.group(1))
                    else:
                        full_log += "⚠️ Debugger agent failed to parse code from response. Aborting.\n"
                        break
            else:
                full_log += "⚠️ Maximum repair attempts reached. Reporting final failure.\n"

        final_passedTests = last_result.get("passedTests", [])
        final_failedTests = last_result.get("failedTests", [])
        success = not final_failedTests

        return {
            "success": success,
            "passed": len(final_passedTests),
            "failed": len(final_failedTests),
            "total": len(final_passedTests) + len(final_failedTests),
            "output": full_log,
            "plugin_tenant": current_tenant_code,
            "plugin_repo": current_repo_code,
            "test_code": current_test_code,
            "attempt_details": attempt_details,
        }
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)