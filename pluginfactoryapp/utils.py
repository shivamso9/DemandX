# pluginfactoryapp/utils.py

import re
import os
import glob
import pandas as pd

def clean_code_output(text):
    """Removes markdown code fences from a string."""
    return re.sub(r'```(json|python|py)?\s*|\s*```', '', text).strip()

def parse_test_results(log, proposed_test_cases):
    """Parses pytest output to identify passed and failed tests."""
    passedTests, failedTests = [], []
    test_pattern = re.compile(r"^(test_plugin\.py::(?:[a-zA-Z0-9_]+::)?(test_[a-zA-Z0-9_]+(?:\[.*?\])?))\s+(PASSED|FAILED|ERROR|SKIPPED)", re.MULTILINE)
    matches = test_pattern.findall(log)
    test_status_map = {name: status for _, name, status in matches}
    test_descriptions = {tc['name']: tc['description'] for tc in proposed_test_cases}
    all_tests_in_log = re.findall(r"test_plugin\.py::(?:[a-zA-Z0-9_]+::)?(test_[a-zA-Z0-9_]+(?:\[.*?\])?)", log)
    unique_test_names = sorted(list(set(all_tests_in_log)))
    for name in unique_test_names:
        status = test_status_map.get(name)
        base_name = name.split('[')[0]
        description = test_descriptions.get(base_name, "Description not available.")
        if status == "PASSED":
            passedTests.append({"name": name, "description": description})
        else:
            reason = "Test did not run or failed. See full log for details."
            failure_block_pattern = re.compile(r"_{{10,}}\s+(?:[a-zA-Z0-9_]+::)?{}\s+_{{10,}}(.*?)((?:^E\s+.*$)|(?:> .*$))".format(re.escape(name)), re.DOTALL | re.MULTILINE)
            match = failure_block_pattern.search(log)
            if match:
                reason_text = match.group(1).strip() + "\n" + match.group(2).strip()
                error_lines = reason_text.splitlines()
                detailed_error_lines = [line for line in error_lines if line.strip().startswith('E ') or line.strip().startswith('> ')]
                reason = "\n".join(detailed_error_lines[-5:]) if detailed_error_lines else "\n".join(error_lines[-5:])
            elif status:
                reason = f"Test ended with status: {status}"
            failedTests.append({"name": name, "description": description, "reason": reason.strip()})
    return passedTests, failedTests

def parse_input_queries(df_input):
    """Parses the 'Input Queries' DataFrame to extract table names and columns."""
    parsed_schemas = {}
    for _, row in df_input.iterrows():
        variable_name, query_string = row.get('VariableName'), row.get('Query')
        if pd.isna(variable_name) or pd.isna(query_string) or not isinstance(query_string, str):
            continue
        columns = [item.split('.')[-1].strip() if '.' in item else item.strip() for item in re.findall(r'\[(.*?)\]', query_string)]
        parsed_schemas[variable_name] = list(dict.fromkeys(filter(None, columns)))
    return parsed_schemas

def get_reusable_code_context():
    """Builds context by reading files from 'helpers' to provide to the AI."""
    helper_filenames_to_consider = ["o9Constants.py", "utils.py"]
    context_str = ""
    helpers_dir = 'helpers'
    for simple_filename in helper_filenames_to_consider:
        full_path = os.path.join(helpers_dir, simple_filename)
        if os.path.isfile(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    relative_path = os.path.relpath(full_path).replace(os.sep, '/')
                    context_str += f"\n\n# --- From: {relative_path} ---\n"
                    context_str += f.read()
            except Exception as e:
                print(f"❌ Error reading specified file {full_path}: {e}")
        else:
            print(f"⚠️ Warning: Specified helper file not found: {full_path}")
    return context_str