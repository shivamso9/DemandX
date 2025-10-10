# pluginfactoryapp/services/agents/generator_agent.py

import os
from flask import current_app
from ... import prompts
from .. import vertex_service
from ...utils import get_reusable_code_context,clean_code_output

def generate_plugin_code(prompt, reference_plugin=None):
    """
    Generates plugin code based on a prompt, optionally including reference code.
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

def fix_json_format(bad_json):
    """Uses the AI model to fix malformed JSON."""
    prompt = prompts.get_json_fixer_prompt(bad_json)
    fixed_json_text = vertex_service.generate_content(prompt)
    return {"fixed_json": clean_code_output(fixed_json_text)}