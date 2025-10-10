# pluginfactoryapp/routes.py

import traceback
from flask import current_app, jsonify, request, render_template
from .services import vertex_service
from .services.agents import validator_agent, generator_agent, debugger_agent, git_agent
from .services import file_service

# Get the current Flask app instance to define routes
app = current_app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/list_plugins', methods=['GET'])
def list_plugins():
    try:
        plugins = file_service.get_plugin_list()
        return jsonify({"plugins": plugins})
    except Exception as e:
        print(f"❌ Error listing plugins: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/validate', methods=['POST'])
def validate():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        result = validator_agent.process_and_validate_spec(file)
        return jsonify(result)
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"A critical error occurred: {e}"}), 500

@app.route('/api/plan', methods=['POST'])
def plan():
    data = request.get_json()
    if not data or 'parsed_schemas' not in data:
        return jsonify({"error": "Missing required data for Planner Agent."}), 400
    # Current logic is a passthrough; can be expanded later.
    return jsonify({"status": "PLAN_SUCCESS", "message": "Planner Agent approved. Proceeding to test generation."})

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Request must include a 'prompt' field"}), 400
    try:
        result = generator_agent.generate_plugin_code(
            prompt=data['prompt'],
            reference_plugin=data.get('reference_plugin')
        )
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/fix_json', methods=['POST'])
def fix_json():
    data = request.get_json()
    if not data or 'bad_json' not in data:
        return jsonify({"error": "Request must include a 'bad_json' field"}), 400
    try:
        result = generator_agent.fix_json_format(data['bad_json'])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to fix JSON: {e}"}), 500

@app.route('/api/execute', methods=['POST'])
def execute():
    data = request.get_json()
    required_keys = ['test_code', 'plugin_repo_code', 'plugin_tenant_code']
    if not all(k in data for k in required_keys):
        return jsonify({"error": "Missing required code fields"}), 400
    try:
        result = debugger_agent.execute_and_debug_code(
            test_code=data['test_code'],
            repo_code=data['plugin_repo_code'],
            tenant_code=data['plugin_tenant_code'],
            proposed_test_cases=data.get('proposed_test_cases', []),
            logic_summary=data.get('logic_summary')
        )
        # Conditionally save on success
        if result.get("success"):
            saved_path = file_service.save_generated_plugin(
                plugin_name=data.get('plugin_name', 'NewPlugin'),
                repo_code=result['plugin_repo'],
                tenant_code=result['plugin_tenant'],
                test_code=result['test_code']
            )
            result['saved_path'] = saved_path
            result['output'] += f"\n✅ Successfully saved plugin files to {saved_path}"
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during execution: {e}"}), 500

@app.route('/api/save_plugin', methods=['POST'])
def save_plugin():
    data = request.get_json()
    required_keys = ['plugin_name', 'repo_code', 'tenant_code', 'test_code']
    if not all(k in data for k in required_keys):
        return jsonify({"error": "Missing required data to save plugin"}), 400
    try:
        saved_path = file_service.save_generated_plugin(
            plugin_name=data.get('plugin_name', 'NewPlugin_ForceSaved'),
            repo_code=data['repo_code'],
            tenant_code=data['tenant_code'],
            test_code=data['test_code']
        )
        return jsonify({"success": True, "saved_path": saved_path})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An error occurred while saving: {e}"}), 500

@app.route('/api/push_to_git', methods=['POST'])
def push_to_git():
    data = request.get_json()
    required_keys = ['plugin_name', 'plugin_repo_code', 'plugin_tenant_code', 'test_code']
    if not all(key in data for key in required_keys):
        return jsonify({"error": "Missing required data for Git push."}), 400
    try:
        result = git_agent.push_to_git_repository(
            plugin_name=data['plugin_name'],
            repo_code=data['plugin_repo_code'],
            tenant_code=data['plugin_tenant_code'],
            test_code=data['test_code']
        )
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except ConnectionRefusedError as e: # For merge conflicts
        return jsonify({"error": str(e)}), 409
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500