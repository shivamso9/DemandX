import os
import subprocess
import tempfile
import shutil
import re
import datetime
import glob
from flask import Flask, request, jsonify
from flask_cors import CORS
import vertexai
from vertexai.generative_models import GenerativeModel
import pandas as pd

# --- Configuration ---
PROJECT_ID = "o9-gemini-codeassist"
REGION = "us-central1"
MODEL_NAME = "gemini-2.5-pro"
OUTPUT_DIR = "generated_plugins" 
REFERENCE_PLUGIN_DIR = "reference_plugins" # Directory to store reference plugins

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Vertex AI Initialization ---
try:
    vertexai.init(project=PROJECT_ID, location=REGION)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    model = GenerativeModel(MODEL_NAME)
    print(f"✅ Vertex AI initialized successfully with {MODEL_NAME}.")
    # Create reference plugin directory if it doesn't exist
    os.makedirs(REFERENCE_PLUGIN_DIR, exist_ok=True)
except Exception as e:
    print(f"❌ Critical Error: Failed to initialize Vertex AI. Reason: {e}")
    model = None

# --- Helper function to clean AI-generated code ---
def clean_code_output(text):
    return re.sub(r'```(python|py)?\s*|\s*```', '', text).strip()

# --- NEW API to list available reference plugins ---
@app.route('/api/list_plugins', methods=['GET'])
def list_plugins():
    try:
        # For this example, we'll list the reference plugins you uploaded.
        # In a real system, this would scan a directory.
        # We'll simulate that by checking for the files we expect.
        simulated_plugin_list = ["DP009OutlierCleansing"]
        
        # You could also scan a real directory like this:
        # plugins = set()
        # for f in glob.glob(os.path.join(REFERENCE_PLUGIN_DIR, "*-Repo.py")):
        #     plugins.add(os.path.basename(f).replace("-Repo.py", ""))
        # for f in glob.glob(os.path.join(REFERENCE_PLUGIN_DIR, "*-Tenant.py")):
        #     plugins.add(os.path.basename(f).replace("-Tenant.py", ""))

        return jsonify({"plugins": sorted(list(simulated_plugin_list))})
    except Exception as e:
        print(f"❌ Error listing plugins: {e}")
        return jsonify({"error": str(e)}), 500


# --- API Route for File Validation (No changes needed) ---
@app.route('/api/validate', methods=['POST'])
def validate():
    # ... This function remains unchanged ...
    if model is None: return jsonify({"error": "Vertex AI model is not available."}), 500
    if 'file' not in request.files: return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    try:
        xls = pd.ExcelFile(file)
        required_sheets = ['Input Queries', 'Output Queries', 'Number Example']
        if not all(sheet in xls.sheet_names for sheet in required_sheets):
            return jsonify({"error": f"Invalid Excel file. Please ensure it contains the following sheets: {', '.join(required_sheets)}."}), 400
            
        df_input = pd.read_excel(xls, sheet_name='Input Queries', header=None)
        df_output = pd.read_excel(xls, sheet_name='Output Queries', header=None)
        df_logic = pd.read_excel(xls, sheet_name='Number Example', header=None)
        
        prompt = f"""
        You are an expert Data Analyst and Python developer, tasked with reverse-engineering the business logic from an Excel specification. Your primary goal is to act like a human analyst meticulously tracing data flow through a spreadsheet.

        **YOUR CORE TASK:**
        Your main objective is to understand how the tables listed in the 'Input Queries' sheet are transformed into the tables listed in the 'Output Queries' sheet by following the calculations laid out in the 'Number Example' sheet.

        **Step-by-Step Analysis Process (Follow this strictly):**

        1.  **Identify Key Tables:**
            * List the names of all input tables from the 'Input Queries' sheet.
            * List the names of all final output tables from the 'Output Queries' sheet.

        2.  **Locate Tables in 'Number Example':**
            * Scan the 'Number Example' sheet to find where the input tables are explicitly shown with their data.
            * Scan the 'Number Example' sheet to find where the final output tables are shown with their data.
            * If you cannot find a clear representation of these tables, you must fail validation and ask the user to clarify where the data is.

        3.  **Trace the Data Flow (Most Critical Step):**
            * Starting from the identified input tables in the 'Number Example', describe every single calculation, filter, join, or aggregation step you see that leads to the next intermediate table or calculation.
            * Continue this process, explaining how each intermediate step builds upon the last.
            * Your trace must conclude by explaining exactly how the final output tables are generated from the previous steps.
            * If there are gaps in the logic or steps that are not clear, you must fail validation and ask a specific question about the missing calculation.

        4.  **Synthesize the Business Logic Plan:**
            * Based on your detailed data flow trace, create a concise, step-by-step summary of the complete business logic. This summary will be used by other agents to write code.

        **NEW SPECIFICATION TO ANALYZE:**
        ---
        Input Queries Sheet:
        {df_input.to_string()}

        Output Queries Sheet:
        {df_output.to_string()}

        Number Example Sheet (Business Logic):
        {df_logic.to_string()}
        ---

        **FINAL INSTRUCTIONS:**
        - If you cannot find the input/output tables in the 'Number Example' sheet, or if the calculation steps are unclear, your entire response **MUST** start with "VALIDATION_FAIL:" followed by a clear question for the user.
        - If the logic is traceable and clear, your entire response **MUST** start with "VALIDATION_SUCCESS:" followed by the synthesized business logic plan.
        """
        response = model.generate_content(prompt)
        return jsonify({"text": response.candidates[0].content.parts[0].text})

    except Exception as e:
        print(f"❌ Error during file validation: {e}")
        return jsonify({"error": f"Failed to process Excel file. Ensure it has 'Input Queries', 'Output Queries', and 'Number Example' sheets. Error: {e}"}), 500


# --- MODIFIED /api/generate to handle reference plugins ---
@app.route('/api/generate', methods=['POST'])
def generate():
    if model is None: return jsonify({"error": "Vertex AI model is not available."}), 500
    data = request.get_json()
    if not data or 'prompt' not in data: return jsonify({"error": "Request must be JSON with a 'prompt' field"}), 400
    
    prompt = data['prompt']
    
    # Check if a reference plugin is requested
    if 'reference_plugin' in data and data['reference_plugin']:
        plugin_name = data['reference_plugin']
        try:
            # In a real system, you would read these from your reference plugin directory
            # For this demo, we'll hardcode the paths to the uploaded files
            with open(f"{plugin_name}-Repo.py", "r") as f:
                repo_code = f.read()
            with open(f"{plugin_name}-Tenant.py", "r") as f:
                tenant_code = f.read()

            # Append the reference code to the prompt
            prompt += f"""
            ---
            **Reference Tenant File Code (`{plugin_name}-Tenant.py`):**
            ```python
            {tenant_code}
            ```

            ---
            **Reference Repo File Code (`{plugin_name}-Repo.py`):**
            ```python
            {repo_code}
            ```
            """
        except FileNotFoundError:
            return jsonify({"error": f"Reference plugin '{plugin_name}' not found on the server."}), 404
        except Exception as e:
            return jsonify({"error": f"Error reading reference plugin: {e}"}), 500

    try:
        response = model.generate_content(prompt)
        return jsonify({"text": response.candidates[0].content.parts[0].text})
    except Exception as e:
        print(f"❌ Error during content generation: {e}")
        return jsonify({"error": "Failed to generate content from the model."}), 500

# --- FIXED /api/execute ---
@app.route('/api/execute', methods=['POST'])
def execute():
    data = request.get_json()
    if not data or 'test_code' not in data or 'plugin_repo_code' not in data or 'plugin_tenant_code' not in data:
        return jsonify({"error": "Missing test_code, plugin_repo_code, or plugin_tenant_code"}), 400

    temp_dir = tempfile.mkdtemp()
    
    try:
        pytest_path = shutil.which("pytest")
        if not pytest_path:
            raise FileNotFoundError("`pytest` command not found. Please ensure pytest is installed.")

        current_test_code = clean_code_output(data['test_code'])
        current_repo_code = clean_code_output(data['plugin_repo_code'])
        current_tenant_code = clean_code_output(data['plugin_tenant_code'])
        
        max_attempts = 3
        full_log = ""
        final_result_obj = None
        
        # Variables to store state between attempts
        previous_error_log = ""
        previous_repo_attempt = ""
        previous_tenant_attempt = ""
        previous_test_attempt = ""


        for attempt in range(max_attempts):
            # --- FIX: Ensure the 'helpers' directory exists before writing to it ---
            helpers_dir = os.path.join(temp_dir, "helpers")
            os.makedirs(helpers_dir, exist_ok=True)
            
            # Write all three files for the test run
            with open(os.path.join(temp_dir, "test_plugin.py"), "w") as f: f.write(current_test_code)
            with open(os.path.join(helpers_dir, "plugin_module.py"), "w") as f: f.write(current_repo_code) # Use a generic name the tenant can import
            with open(os.path.join(temp_dir, "plugin_tenant.py"), "w") as f: f.write(current_tenant_code)
            
            result = subprocess.run([pytest_path, "test_plugin.py"], cwd=temp_dir, capture_output=True, text=True, timeout=60)
            final_result_obj = result
            
            log_header = f"--- Attempt {attempt + 1}/{max_attempts} ---"
            full_log += f"{log_header}\n{result.stdout}{result.stderr}\n\n"

            if result.returncode == 0:
                full_log += "✅ Tests Passed. Exiting correction loop.\n"
                break
            
            full_log += f"❌ Tests Failed on Attempt {attempt + 1}.\n"
            if attempt < max_attempts - 1:
                error_log = result.stdout + result.stderr
                logic_summary = data.get('logic_summary', 'No logic summary provided.')

                triage_prompt = f"""Analyze the following pytest error log. Is the root cause more likely in the business logic (Repo or Tenant file) or in the test suite (`test_plugin.py`)?

                Pytest Log: --- {error_log} ---
                **Analysis Guidance:**
                - If the error is `AssertionError`, `ValueError`, `KeyError`, `TypeError`, or `AttributeError` within the plugin's function, it's a **PLUGIN_ERROR**.
                - If the error is a `SyntaxError` in the test file, an `ImportError` for a test library, or a `pytest`-specific error like 'Fixture ... called directly', it's a **TEST_ERROR**.

                Respond with only the string "PLUGIN_ERROR" or "TEST_ERROR". """
                triage_response = model.generate_content(triage_prompt)
                triage_result = triage_response.candidates[0].content.parts[0].text.strip()
                
                full_log += f"🤖 Triage determined the issue is a {triage_result}. Activating appropriate agent...\n\n"

                if triage_result == "TEST_ERROR":
                    test_corrector_prompt = f"""You are an expert Pytest developer. The previous test code failed. Analyze the error and provide a corrected version.
                    **Your PREVIOUS Incorrect Test Code:**\n{previous_test_attempt or current_test_code}
                    **Error from PREVIOUS Attempt:**\n{previous_error_log or error_log}
                    **CRITICAL:** Respond with ONLY the raw, corrected Python code for `test_plugin.py`. Do not include explanations."""
                    previous_test_attempt = current_test_code
                    response = model.generate_content(test_corrector_prompt)
                    current_test_code = clean_code_output(response.candidates[0].content.parts[0].text)
                
                else: # PLUGIN_ERROR
                    debug_prompt = f"""You are an expert Python Debugger Agent specializing in pandas and enterprise codebases. The previous code failed. Your task is to analyze the error and provide corrected versions of the Tenant and Repo files.

                    **Original Business Logic:** {logic_summary}
                    **Your PREVIOUS Incorrect Tenant Code:**\n{previous_tenant_attempt or current_tenant_code}
                    **Your PREVIOUS Incorrect Repo Code:**\n{previous_repo_attempt or current_repo_code}
                    **Error from PREVIOUS Attempt:**\n{previous_error_log or error_log}

                    **INSTRUCTIONS:**
                    1.  Analyze the error log to understand the root cause.
                    2.  Determine if the fix is needed in the Tenant file (data handling/imports), the Repo file (core logic), or both.
                    3.  Rewrite the necessary code to fix the bug.
                    4.  **You MUST provide the complete code for BOTH files**, even if you only change one.

                    **CRITICAL:** Your entire response must contain two distinct, clearly marked sections: one for the Tenant file and one for the Repo file.
                    - Start the Tenant file code with `### TENANT FILE ###`.
                    - Start the Repo file code with `### REPO FILE ###`.
                    """
                    previous_tenant_attempt = current_tenant_code
                    previous_repo_attempt = current_repo_code
                    
                    response = model.generate_content(debug_prompt)
                    full_corrected_code = response.candidates[0].content.parts[0].text
                    
                    tenant_match = re.search(r'### TENANT FILE ###\s*(.*?)### REPO FILE ###', full_corrected_code, re.S)
                    repo_match = re.search(r'### REPO FILE ###\s*(.*)', full_corrected_code, re.S)

                    if tenant_match and repo_match:
                        current_tenant_code = clean_code_output(tenant_match.group(1))
                        current_repo_code = clean_code_output(repo_match.group(1))
                
                previous_error_log = error_log

            else:
                full_log += "⚠️ Maximum repair attempts reached. Reporting final failure.\n"

        summary_line = next((line for line in reversed(full_log.splitlines()) if "===" in line and ("passed" in line or "failed" in line)), None)
        passed_count, failed_count = 0, 0
        if summary_line:
            passed_match = re.search(r'(\d+)\s+passed', summary_line)
            failed_match = re.search(r'(\d+)\s+failed', summary_line)
            if passed_match: passed_count = int(passed_match.group(1))
            if failed_match: failed_count = int(failed_match.group(1))
        total_count = passed_count + failed_count
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_plugin_name = data.get('plugin_name', 'NewPlugin')
        save_path = os.path.join(OUTPUT_DIR, f"{new_plugin_name}_{timestamp}")
        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path, "test_plugin.py"), "w") as f: f.write(current_test_code)
        with open(os.path.join(save_path, f"{new_plugin_name}-Repo.py"), "w") as f: f.write(current_repo_code)
        with open(os.path.join(save_path, f"{new_plugin_name}-Tenant.py"), "w") as f: f.write(current_tenant_code)

        return jsonify({
            "success": final_result_obj.returncode == 0,
            "passed": passed_count, "failed": failed_count, "total": total_count,
            "output": full_log,
            "saved_path": save_path
        })

    except Exception as e:
        print(f"❌ Error during code execution: {e}")
        return jsonify({"error": f"An error occurred while executing the code: {e}"}), 500
    finally:
        shutil.rmtree(temp_dir)

@app.route('/')
def index(): return "Backend server is running."

if __name__ == '__main__':
    # Make sure reference files are available for the demo
    # In a real system, these would be managed in a dedicated folder.
    # For now, we assume they are in the same directory as the script.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)

