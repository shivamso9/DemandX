# pluginfactoryapp/services/file_service.py

import os
import glob
import datetime
import json
from flask import current_app

def get_plugin_list():
    """
    Scans the tenant plugin directory and returns a sorted list of unique plugin names.
    """
    tenant_dir = current_app.config['TENANT_PLUGIN_DIR']
    if not os.path.isdir(tenant_dir):
        return []

    # Use a set to handle potential duplicates from -Repo and -Tenant files
    plugin_names = {
        os.path.basename(f).replace("-Repo.py", "").replace("-Tenant.py", "").replace(".py", "")
        for f in glob.glob(os.path.join(tenant_dir, "*.py"))
    }
    return sorted(list(plugin_names))

def save_generated_plugin(plugin_name, repo_code, tenant_code, test_code):
    """
    Saves the repo, tenant, and test code to a new versioned folder
    in the main output directory.
    
    Returns:
        str: The path to the newly created directory for the saved plugin.
    """
    output_dir = current_app.config['OUTPUT_DIR']
    
    # Create a unique, timestamped directory for this version of the plugin
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_path = os.path.join(output_dir, f"{plugin_name}_{timestamp}")

    # Define subdirectories consistent with the project structure
    helpers_path = os.path.join(saved_path, "helpers")
    scripts_path = os.path.join(saved_path, "plugin_scripts")
    tests_path = os.path.join(saved_path, "tests")

    # Create all necessary directories
    os.makedirs(helpers_path, exist_ok=True)
    os.makedirs(scripts_path, exist_ok=True)
    os.makedirs(tests_path, exist_ok=True)

    # Write the code to the appropriate files
    with open(os.path.join(helpers_path, "plugin_repo.py"), "w", encoding='utf-8') as f:
        f.write(repo_code)
        
    with open(os.path.join(scripts_path, "plugin_tenant.py"), "w", encoding='utf-8') as f:
        f.write(tenant_code)
        
    with open(os.path.join(tests_path, "test_plugin.py"), "w", encoding='utf-8') as f:
        f.write(test_code)

    print(f"✅ Plugin files saved successfully to: {saved_path}")
    return saved_path

def save_test_plan(plugin_name, test_cases_json):
    """Saves a list of test cases to a JSON file."""
    # NOTE: You must add 'TEST_PLAN_DIR' to your Flask app config.
    # e.g., app.config['TEST_PLAN_DIR'] = 'instance/test_plans'
    test_plan_dir = current_app.config['TEST_PLAN_DIR']
    os.makedirs(test_plan_dir, exist_ok=True)
    file_path = os.path.join(test_plan_dir, f"{plugin_name}.json")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_cases_json, f, indent=2)
        return {"status": "success", "path": file_path}
    except Exception as e:
        print(f"Error saving test plan for {plugin_name}: {e}")
        return {"status": "error", "message": str(e)}

def get_test_plan(plugin_name):
    """Retrieves a test plan if it exists."""
    test_plan_dir = current_app.config['TEST_PLAN_DIR']
    file_path = os.path.join(test_plan_dir, f"{plugin_name}.json")

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return {"status": "found", "plan": json.load(f)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    else:
        return {"status": "not_found"}

def create_initial_directories_and_files():
    """
    Ensures that all necessary directories and placeholder files exist on startup.
    This makes the application self-sufficient.
    """
    # Ensure base directories exist
    ref_dir = current_app.config['REFERENCE_PLUGIN_DIR']
    tenant_dir = current_app.config['TENANT_PLUGIN_DIR']
    output_dir = current_app.config['OUTPUT_DIR']
    
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(tenant_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create placeholder demo plugin if it doesn't exist
    repo_placeholder_path = os.path.join(ref_dir, "DP009OutlierCleansing-Repo.py")
    tenant_placeholder_path = os.path.join(ref_dir, "DP009OutlierCleansing-Tenant.py")

    if not os.path.exists(repo_placeholder_path):
        with open(repo_placeholder_path, "w", encoding='utf-8') as f:
            f.write("import pandas as pd\n\ndef main(df: pd.DataFrame):\n    # Placeholder function\n    return df\n")
        print(f"✅ Created placeholder file: {repo_placeholder_path}")

    if not os.path.exists(tenant_placeholder_path):
        with open(tenant_placeholder_path, "w", encoding='utf-8') as f:
            f.write("from helpers.plugin_module import main\n\nif __name__ == '__main__':\n    # This script calls the main logic\n    pass # In a real scenario, you'd load data and call main()\n")
        print(f"✅ Created placeholder file: {tenant_placeholder_path}")