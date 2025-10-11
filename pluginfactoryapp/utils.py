# pluginfactoryapp/utils.py

import re
import os
import glob
import pandas as pd

def clean_code_output(text):
    """Removes markdown code fences from a string."""
    return re.sub(r'```(json|python|py)?\s*|\s*```', '', text).strip()

def extract_test_data_from_code(test_code):
    """
    Parses the full text of a pytest file to extract input/expected data for each test.
    Returns a dictionary mapping test names to their data.
    """
    test_data_map = {}
    function_pattern = re.compile(r"def\s+(test_[a-zA-Z0-9_]+)\s*\([^)]*\):\s*([\s\S]*?)(?=\ndef\s|\Z)", re.DOTALL)
    data_pattern = re.compile(r"(\w+_csv)\s*=\s*\"\"\"([\s\S]*?)\"\"\"")

    for match in function_pattern.finditer(test_code):
        test_name, function_body = match.groups()
        data_vars = data_pattern.findall(function_body)
        
        test_data_map[test_name] = {
            "input_data": "Not Found",
            "expected_data": "Not Found"
        }

        for var_name, var_content in data_vars:
            if 'input' in var_name.lower():
                test_data_map[test_name]["input_data"] = var_content.strip()
            elif 'expected' in var_name.lower():
                test_data_map[test_name]["expected_data"] = var_content.strip()
                
    return test_data_map

def parse_test_results(log, proposed_test_cases, test_data_map=None):
    """Parses pytest output to identify passed/failed tests, now with more detail."""
    if test_data_map is None:
        test_data_map = {}
        
    passedTests, failedTests = [], []
    test_pattern = re.compile(r"^(?:test_plugin|test_custom)\.py::(?:[a-zA-Z0-9_]+::)?(test_[a-zA-Z0-9_]+(?:\[.*?\])?)\s+(PASSED|FAILED|ERROR|SKIPPED)", re.MULTILINE)
    matches = test_pattern.findall(log)
    
    test_status_map = {name: status for name, status in matches}
    test_descriptions = {tc['name']: tc['description'] for tc in proposed_test_cases}
    
    all_tests_in_log = re.findall(r"(?:test_plugin|test_custom)\.py::(?:[a-zA-Z0-9_]+::)?(test_[a-zA-Z0-9_]+(?:\[.*?\])?)", log)
    unique_test_names = sorted(list(set(all_tests_in_log)))

    for name in unique_test_names:
        status = test_status_map.get(name)
        base_name = name.split('[')[0]
        description = test_descriptions.get(base_name, "Description not available.")
        
        data = test_data_map.get(base_name, {
            "input_data": "N/A",
            "expected_data": "N/A"
        })

        test_details = {
            "name": name,
            "description": description,
            "input_data": data["input_data"],
            "expected_data": data["expected_data"]
        }

        if status == "PASSED":
            passedTests.append(test_details)
        else:
            reason = "Test did not run or failed. See full log for details."
            
            # --- THIS LINE IS CORRECTED ---
            # The regex curly braces {10,} are now escaped as {{10,}} to prevent a .format() KeyError.
            failure_block_pattern = re.compile(r"_{{10,}}\s+(?:[a-zA-Z0-9_]+::)?{}\s+_{{10,}}(.*?)((?:^E\s+.*$)|(?:> .*$))".format(re.escape(name)), re.DOTALL | re.MULTILINE)
            
            match = failure_block_pattern.search(log)
            if match:
                reason_text = (match.group(1) + match.group(2)).strip()
                assertion_error = re.search(r"E\s+AssertionError: ([\s\S]*)", reason_text)
                if assertion_error:
                    reason = assertion_error.group(1).strip()
                else:
                    error_lines = reason_text.splitlines()
                    reason = "\n".join(line.strip() for line in error_lines if line.strip())
            elif status:
                reason = f"Test ended with status: {status}"
            
            test_details["reason"] = reason.strip()
            failedTests.append(test_details)
            
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
    if not os.path.isdir(helpers_dir):
        helpers_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'helpers')

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