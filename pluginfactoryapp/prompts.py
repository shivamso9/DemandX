# prompts.py
import pandas as pd
import numpy as np
import io

def get_validation_prompt(schema_review_string, df_output_string, df_logic_string):
    """Generates the prompt for the Inspector Agent to validate the spec file."""
    return f"""
    You are a meticulous data analyst. Your task is to analyze the provided Excel sheets and create a detailed and accurate business logic plan.

    **TASK:**
    Generate three sections: PSEUDOCODE, FLOWCHART, and OUTPUT SCHEMA.

    **CRITICAL INSTRUCTIONS & RULES:**
    1.  **PSEUDOCODE SECTION:**
        * Start with an "Inputs:" list showing each input table along with all the columns/fields.
        * Describe step-by-step how inputs are transformed into outputs, explicitly listing all input and intermediate tables with the column/field used, detailing each calculation, transformation, or business rule so that a coder or tester can implement and verify it fully and accurately.
        * Ends with "Inputs:" list showing each output table along with all the columns/fields.
        
    2.  **FLOWCHART SECTION:** Create a simple, numbered list of the main 5-8 high-level logical steps.

    3.  **OUTPUT SCHEMA SECTION:** Create a JSON object where keys are the 'VariableName' from the 'Output Queries' sheet and values are arrays of column names for that output.

    4.  **FINAL VERIFICATION:** Before finishing, you must double-check that every column listed in the 'Output Queries Sheet' is present in your 'Column Mapping' section and in your 'OUTPUT SCHEMA' JSON.

    5.  **FORMAT:** The entire response MUST start with "VALIDATION_SUCCESS:" and contain the three sections clearly marked with `### SECTION_NAME ###`.

    ---
    **Input Data Schema (from 'Input Queries'):**
    {schema_review_string}
    ---
    **Output Queries Sheet:**
    {df_output_string}
    ---
    **Logic (from 'Number Example'):**
    {df_logic_string}
    ---
    """

def get_test_architect_prompt(logic_plan):
    """Generates the prompt for the Test Architect Agent."""
    return f"""
    You are an expert Python developer specializing in pytest. Your task is to generate a single, complete, runnable pytest file to test a pandas-based plugin.

    **Business Logic Summary:**
    {logic_plan}

    ---
    ### CRITICAL INSTRUCTIONS
    1.  **Generate a Full Pytest File**: Your output must be a single, complete Python file.
    2.  **Self-Contained Assertion Helper**:
        * The file **MUST** include a helper function named `assert_df_equal`.
        * This function must compare two pandas DataFrames for equality by sorting both, resetting their indexes, and handling `object` and date column inconsistencies before comparison.
        * This function should be robust enough to prevent common `dtype` mismatch errors.
    3.  **DataFrames from CSV Strings**:
        * All test data must be embedded within the test functions as multi-line CSV strings.
        * **Crucially**, when creating DataFrames with `pd.read_csv(io.StringIO(...))`, you **MUST** use the `dtype` parameter to explicitly specify the data types for all columns that might be ambiguous (e.g., nullable integers, object/string columns with empty values). This is the most important step to prevent `dtype` errors.
    4.  **Test Case Design**:
        * Create a comprehensive suite of tests covering the "Golden Record" (main success path), "Edge Cases", and complex "Scenarios".
        * Each test function should be clearly named (e.g., `test_golden_record_...`, `test_edge_...`).
    5.  **Imports**: The test file must import the function to be tested (`main`) from `helpers.plugin_module`.
    6.  **Code Only**: Your entire output MUST be only the raw, valid Python code. Do not include any explanations or markdown formatting.

    ---
    ### EXAMPLE TEST STRUCTURE

    ```python
    import pandas as pd
    import io
    from helpers.plugin_module import main

    def assert_df_equal(left, right):
        # ... implementation that sorts, resets index, and handles dtypes ...
        pd.testing.assert_frame_equal(left_sorted, right_sorted)

    def test_golden_record_example():
        input_csv = \"\"\"ItemID,Category,Value
    1,A,100
    2,B,
    \"\"\"
        expected_csv = \"\"\"ItemID,Category,Value
    1,A,100.0
    2,B,
    \"\"\"
        # CRITICAL: Use the dtype parameter to prevent inference errors
        input_df = pd.read_csv(
            io.StringIO(input_csv),
            dtype={{'ItemID': 'Int64', 'Category': 'object', 'Value': 'float64'}}
        )
        expected_df = pd.read_csv(
            io.StringIO(expected_csv),
            dtype={{'ItemID': 'Int64', 'Category': 'object', 'Value': 'float64'}}
        )

        results = main(input_df)
        assert_df_equal(results['output'], expected_df)

    # ... more tests for edge cases and scenarios ...
    ```
    """

def get_coder_agent_prompt(logic_plan, test_descriptions, tenant_code, repo_code):
    """Generates the prompt for the Coder Agent."""
    base_prompt = f"""You are a senior Python developer specializing in pandas. Your task is to generate two Python files based on the provided business logic and test plan.

    **CRITICAL INSTRUCTIONS:**
    1.  **Function Signature:** The Repo file's main function must accept multiple pandas DataFrames as arguments (e.g., `def main(ItemMaster, AssortmentFinal, ...):`).
    2.  **Return Type:** The main function MUST return a dictionary where keys are the output table names (e.g., "AssortmentFinal") and values are the corresponding pandas DataFrames.
    3.  **Data Types:** Ensure numeric columns that could contain nulls are explicitly cast to a nullable type. Use `Int64` for integers and `float64` for floats.
    4.  **Generate Two Files:** Your output must contain two distinct sections: `### TENANT FILE ###` and `### REPO FILE ###`.
    5.  **Code Quality:** Write clean, readable, and robust code. Avoid common pandas pitfalls like `SettingWithCopyWarning`.
    ---
    **Business Logic to Implement:**
    ---
    {logic_plan}
    ---
    **Pytest Plan to Satisfy:**
    ---
    {test_descriptions}
    ---
    **STRICT OUTPUT FORMAT:** Your response must contain ONLY the two markers and their corresponding code blocks. No other text.
    """

    if tenant_code and repo_code:
        base_prompt += f"""
        ---
        **Reference Tenant File Code:**
        ```python
        {tenant_code}
        ```
        ---
        **Reference Repo File Code:**
        ```python
        {repo_code}
        ```
        """
    return base_prompt


def get_triage_prompt(error_log):
    """Generates the prompt to determine if an error is in the test or the plugin."""
    return f"""
    Analyze the pytest error log to determine the root cause.
    - If the error is a syntax issue, typo, missing import, or incorrect assertion logic in the test file itself (`test_plugin.py`), it is a TEST_ERROR.
    - If the error is a logical flaw, incorrect data processing, or structural output mismatch in the repo/tenant code being tested, it is a PLUGIN_ERROR.
    - If the error is `fixture not found`, it is a TEST_ERROR.

    Pytest Log: ---
    {error_log}
    ---

    Respond with only the string "PLUGIN_ERROR" or "TEST_ERROR".
    """

def get_test_corrector_prompt(test_code, error_log):
    """Generates the prompt to fix a broken test file."""
    return f"""
    You are an expert Pytest developer. The provided test code has failed. The root cause is an error within the test file itself, such as a syntax error, a missing import, a typo, or an incorrect use of a fixture. Analyze the error log and provide a corrected version of the test code.

    **Your PREVIOUS Incorrect Test Code:**
    ```python
    {test_code}
    ```

    **Critical Pytest Error Log:**
    ```
    {error_log}
    ```

    **CRITICAL:** Respond with ONLY the complete, raw, corrected Python code for the `test_plugin.py` file. Do not include explanations or markdown.
    """

def get_debug_prompt(logic_summary, repo_code, error_log, previous_attempts_summary=""):
    """
    Generates a highly structured, multi-stage prompt for a powerful Debugger Agent.
    """
    return f"""
    You are a world-class Python and pandas Debugging Agent. Your task is to meticulously analyze the provided code and Pytest failure log, identify every single error, and provide a complete, corrected version of all necessary files in a single response.

    {previous_attempts_summary}

    ### Business Logic Summary
    {logic_summary}

    ### Previous Incorrect Code
    ```python
    {repo_code} # This may contain both the plugin and test code.
    ```

    ### Critical Pytest Failure Log
    ```
    {error_log}
    ```

    ---

    ### MANDATORY DEBUGGING INSTRUCTIONS

    **Step 1: Comprehensive Error Triage & Classification**
    In a `### DEBUGGING ANALYSIS ###` section, begin your response by scanning the **entire log** from top to bottom. Identify, list, and classify **ALL** unique errors. Do not stop after finding the first one. For each error, specify its file of origin (`plugin_module.py` or `test_plugin.py`).

    Classify each error as one of the following:
    * **(A) Syntax Error:** A fatal error like `SyntaxError` that prevents Python from parsing the file.
    * **(B) Environment/Import Error:** `ModuleNotFoundError`, `ImportError`, or `fixture not found`.
    * **(C) Runtime NameError:** A `NameError` from a typo in a variable or function name.
    * **(D) Structural Output Error:** A `TypeError` or `KeyError` on the direct `results` object (e.g., `results["my_key"]`). This indicates the plugin is returning the wrong data structure (e.g., a raw DataFrame or tuple instead of a dictionary like `{{'my_key': df}}`).
    * **(E) Column Not Found Error:** A `KeyError` when selecting columns from a DataFrame (e.g., `my_df[['col_A']]`). This means the expected columns do not exist at that stage of the logic.
    * **(F) Core Data Logic Error:** An `AssertionError` (e.g., shape mismatch, wrong values, wrong dtype) or `InvalidIndexError`. This means the code runs but the pandas logic is flawed.
    * **(G) Object API Misuse Error:** An `AttributeError` (e.g., calling `.iloc` on an object that doesn't have it) or a `TypeError` from passing wrong arguments to a function.

    **Step 2: Multi-File Root Cause Analysis & Fix Strategy**
    For each error you classified, provide a brief root cause analysis and your fix strategy.
    * **For Syntax/NameError:** State the exact typo and the file to be corrected.
    * **For Environment/Import Error:** For `ModuleNotFoundError`, identify and plan to remove non-essential imports. For `fixture not found`, the test file requires a missing dependency like `pytest-mock`.
    * **For Structural Output Error:** This is a critical error. Confirm the expected output structure from the test code (e.g., `results["assortment_final"]` expects a dictionary). The fix is almost always to modify the `return` statement in the main plugin function to match this dictionary structure.
    * **For Column Not Found Error:** Hypothesize why the columns are missing. Did a previous step fail to create them? Were they dropped? Did a `pd.merge` add suffixes (e.g., `_x`, `_y`) that were not handled? Is the code correctly handling empty input DataFrames which might not have the expected columns? State the exact column name fix.
    * **For Core Data Logic Error:** Analyze the pandas logic.
        - **`AssertionError: Attribute "dtype" are different`**: This is a critical and common error. The cause can be in the plugin OR the test.
            - **Is the plugin wrong?** The plugin might be outputting `int64` when `float64` is expected. The fix is to use `.astype('float64')` on the specific column in the plugin code before returning the result.
            - **Is the test wrong?** The test's `expected` DataFrame, often created with `pd.read_csv`, might infer a `float64` type because of `NaN` values, while the plugin correctly produces a nullable integer `Int64`. The fix is to modify the test file by adding an explicit `dtype` parameter to `pd.read_csv`, like `dtype={{'column_name': 'Int64'}}` or `dtype={{'column_name': str}}` to enforce consistency.
            - **Your Strategy:** Prioritize fixing the test if the plugin's output type seems more logically correct (e.g., IDs should be integers). Otherwise, fix the plugin.
        - **`AssertionError: DataFrame shape mismatch`**: The logic is producing the wrong number of rows. Is a `.merge()` or `.groupby()` incorrect? Is `.drop_duplicates()` too aggressive?
        - **`InvalidIndexError`**: This often happens on `pd.concat`. The fix is usually to use `ignore_index=True` or to reset the index of the dataframes before concatenation.
    * **For Object API Misuse Error:** State the exact object type (e.g., `'Index' object`), the incorrect method/attribute being used (`.iloc`), and the file where it occurs. The fix is to replace the incorrect call with the correct API for that object (e.g., "A pandas Index should be accessed with `my_index[i]` instead of `my_index.iloc[i]`).

    **Step 3: Provide Batched Code Fixes**
    Based on your complete analysis, provide the full, corrected code for ALL files that need changes.
    * Use a `### REPO FILE ###` section for the main plugin code (`plugin_module.py`).
    * Use a `### TEST FILE ###` section for the test code (`test_plugin.py`).
    * If a file does not need changes, do not include its section.
    """

def get_json_fixer_prompt(bad_json):
    """Generates the prompt to fix malformed JSON."""
    return f"""
    The following text is not valid JSON. Correct all syntax errors (such as trailing commas, single quotes, or missing brackets) and return ONLY the raw, valid JSON text.

    ---INVALID JSON---
    {bad_json}
    ---END INVALID JSON---
    """

def get_reusable_prompt(reusable_code_context):
    return f""" # <--- ADDED
                    ### Instructions for Reusing Library Code
                    You are an expert Python developer. Your most important goal is to **reuse existing code** from the provided library. **Write as little new code as possible.**
                    You **MUST** follow this exact workflow:

                    **1. Understand the Goal:**
                    First, read the business logic you need to implement.

                    **2. Scan the Library:**
                    Next, carefully review all the code snippets provided below under "Library Code". For each function or class, understand what it does and where it is located.

                    **3. Plan Your Code with Imports:**
                    Before writing, decide which library functions you will use.
                    * **Identify** a useful function in the library.
                    * **Find its file path** from the comment above it, like `# --- From: helpers/data_cleaner.py ---`.
                    * **Create the correct import statement** for it, like `from helpers.data_cleaner import function_name`.

                    **4. Write the Final Code:**
                    * Begin your script with all the necessary import statements you planned.
                    * Implement the required business logic by **calling the functions you imported** from the library.
                    * **CRITICAL:** Only write a new function if the required functionality is **completely missing** from the library. Do not write a new function that is similar to an existing one.
                    
                    **5. Use Constants and decorators 
                    * reuse O9Constants while referencing constants.
                    * make use of decoartors from o9references.
                    ---

                    ### **Example Scenario:**

                    * **Your Task:** "Remove outliers from the sales data."
                    * **You See in the Library Code:**
                        ```python
                        # --- From: helpers/data_cleaner.py ---
                        def remove_outliers(dataframe, column):
                            # ... code to remove outliers ...
                            return dataframe
                        ```

                    * **Your CORRECT Action:** Your code should import and use the library function.
                        ```python
                        from helpers.data_cleaner import remove_outliers

                        def my_plugin_logic(sales_df):
                            # Correctly reusing the library function
                            cleaned_df = remove_outliers(sales_df, 'SalesValue')
                            return cleaned_df
                        ```

                    * **Your INCORRECT Action (DO NOT DO THIS):** Writing your own logic instead of importing.
                        ```python
                        # This is WRONG because you are re-writing logic that already exists.
                        def my_plugin_logic(sales_df):
                            q1 = sales_df['SalesValue'].quantile(0.25)
                            q3 = sales_df['SalesValue'].quantile(0.75)
                            # ... etc ...
                            return sales_df
                        ```
                    ---
                    **Final Review:** Before you finish, double-check your code. If you have written a new function, make absolutely sure that no function in the library could have been used instead.
                    {reusable_code_context}"""