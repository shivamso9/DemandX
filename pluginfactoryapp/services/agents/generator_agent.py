# pluginfactoryapp/services/agents/generator_agent.py

import os
import re
from flask import current_app
from ... import prompts
from .. import vertex_service
from ...utils import get_reusable_code_context, clean_code_output

def generate_plugin_code(prompt, reference_plugin=None):
    """
    Generates plugin code based on a prompt, optionally including reference code.
    This is used for generating the main Repo and Tenant files.
    """
    final_prompt = prompt
    
    if reference_plugin and reference_plugin != "DP009OutlierCleansing (Demo)":
        try:
            ref_dir = current_app.config['REFERENCE_PLUGIN_DIR']
            tenant_dir = current_app.config['TENANT_PLUGIN_DIR']
            
            with open(os.path.join(ref_dir, f"{reference_plugin}.py"), "r") as f:
                repo_code = f.read()
            with open(os.path.join(tenant_dir, f"{reference_plugin}.py"), "r") as f:
                tenant_code = f.read()
            
            reusable_code_context = get_reusable_code_context()
            if reusable_code_context:
                final_prompt += prompts.get_reusable_prompt(reusable_code_context)
            
            final_prompt += f"""
            ---
            **Reference Tenant File Code (`{reference_plugin}-Tenant.py`):**
            ```python
            {tenant_code}
            ```
            ---
            **Reference Repo File Code (`{reference_plugin}-Repo.py`):**
            ```python
            {repo_code}
            ```
            """
        except FileNotFoundError:
            raise FileNotFoundError(f"Reference plugin '{reference_plugin}' not found on the server.")
        except Exception as e:
            raise IOError(f"Error reading reference plugin: {e}")

    generated_text = vertex_service.generate_content(final_prompt)
    return {"text": generated_text}

def generate_test_file_chunked(logic_summary, test_cases):
    """
    Generates a pytest file by generating functions in batches to optimize for speed and reliability.
    """
    all_imports = set([
        "import pandas as pd",
        "import io",
        "from helpers.plugin_module import main"
    ])
    code_pieces = []
    
    # --- BATCHING LOGIC ---
    batch_size = 4  # You can tune this number. 3-5 is a good starting point.
    test_batches = [test_cases[i:i + batch_size] for i in range(0, len(test_cases), batch_size)]
    # ----------------------

    assertion_helper = """
def assert_df_equal(left, right, key_cols=None):
    \"\"\"Compares two DataFrames for equality after sorting and resetting index.\"\"\"
    if key_cols:
        left_sorted = left.sort_values(by=key_cols).reset_index(drop=True)
        right_sorted = right.sort_values(by=key_cols).reset_index(drop=True)
    else:
        left_sorted = left.sort_values(by=left.columns.tolist()).reset_index(drop=True)
        right_sorted = right.sort_values(by=right.columns.tolist()).reset_index(drop=True)
    
    pd.testing.assert_frame_equal(left_sorted, right_sorted, check_dtype=True)
"""
    code_pieces.append(assertion_helper)

    # Loop through batches instead of individual cases
    for i, batch in enumerate(test_batches):
        batch_names = [case['name'] for case in batch]
        print(f"🤖 Generating Batch {i+1}/{len(test_batches)}: {', '.join(batch_names)}...")
        
        # Dynamically build the list of test cases for the prompt
        test_cases_prompt_str = ""
        for case in batch:
            test_cases_prompt_str += f"- Function Name: `{case['name']}`\n  - Description: `{case['description']}`\n"

        # The prompt now asks for a batch of functions
        prompt = f"""
        You are an expert Python developer specializing in pytest.
        Your task is to write a set of Python pytest functions based on the provided business logic and test case descriptions.

        **Business Logic Summary:**
        ---
        {logic_summary}
        ---
        **Test Cases to Implement in this Batch:**
        {test_cases_prompt_str}
        ---
        **CRITICAL INSTRUCTIONS:**
        1.  Write ONLY the Python code for the functions listed above.
        2.  For each function, include all necessary data setup (e.g., CSV strings) inside that function.
        3.  Each function must call the main plugin logic and end with an assertion.
        4.  Do NOT include any imports or helper functions (like assert_df_equal), only the test functions themselves.
        5.  Your entire output MUST be only the raw, valid Python code for the batch of functions, with each function separated by two newlines.
        """

        try:
            batch_code = vertex_service.generate_content(prompt)
            cleaned_code = clean_code_output(batch_code)
            
            imports_in_code = re.findall(r"^\s*import\s.*|^\s*from\s.*", cleaned_code, re.MULTILINE)
            for imp in imports_in_code:
                all_imports.add(imp.strip())
            
            code_without_imports = "\n".join([line for line in cleaned_code.split('\n') if not line.strip().startswith(('import ', 'from '))])
            code_pieces.append(code_without_imports)

        except Exception as e:
            print(f"⚠️ Failed to generate batch for {', '.join(batch_names)}. Error: {e}. Skipping.")
            error_placeholder = f"\n# Batch of tests failed to generate due to an API error.\n# Tests: {', '.join(batch_names)}\n# Error: {e}"
            code_pieces.append(error_placeholder)

    # Assemble the final file
    final_code = "\n".join(sorted(list(all_imports))) + "\n\n" + "\n\n".join(code_pieces)
    
    return {"text": final_code}

def fix_json_format(bad_json):
    """Uses the AI model to fix malformed JSON."""
    prompt = prompts.get_json_fixer_prompt(bad_json)
    fixed_json_text = vertex_service.generate_content(prompt)
    return {"fixed_json": clean_code_output(fixed_json_text)}