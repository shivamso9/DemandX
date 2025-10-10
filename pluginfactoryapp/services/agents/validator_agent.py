# pluginfactoryapp/services/agents/validator_agent.py

import pandas as pd
import json
from ... import prompts
from .. import vertex_service
from ...utils import parse_input_queries

def process_and_validate_spec(file_storage):
    """
    Parses the uploaded Excel file, validates its structure, and uses an AI model
    to generate a validation summary and output schema.
    """
    xls = pd.ExcelFile(file_storage)
    required_sheets = ['Input Queries', 'Output Queries', 'Number Example']
    if not all(sheet in xls.sheet_names for sheet in required_sheets):
        missing = ', '.join(s for s in required_sheets if s not in xls.sheet_names)
        raise ValueError(f"Invalid Excel file. Missing sheets: {missing}.")

    df_input_raw = pd.read_excel(xls, sheet_name='Input Queries', header=None)
    header_row_index = next((i for i, row in df_input_raw.iterrows() if 'VariableName' in str(row.values)), -1)
    if header_row_index == -1:
        raise ValueError("Could not find 'VariableName' header in 'Input Queries' sheet.")

    df_input = pd.read_excel(xls, sheet_name='Input Queries', header=header_row_index)
    df_input.columns = [str(col).strip() for col in df_input.columns]
    df_output = pd.read_excel(xls, sheet_name='Output Queries')
    df_logic = pd.read_excel(xls, sheet_name='Number Example', header=None)

    parsed_schemas = parse_input_queries(df_input)
    schema_review_string = "\n".join([f"  - {table}: {', '.join(cols)}" for table, cols in parsed_schemas.items()])
    output_schema_json = df_output.to_json(orient='records')

    prompt = prompts.get_validation_prompt(
        schema_review_string=schema_review_string,
        df_output_string=df_output.to_string(),
        df_logic_string=df_logic.to_string()
    )
    
    response_text = vertex_service.generate_content(prompt)
    
    if "### OUTPUT SCHEMA ###" not in response_text:
        response_text += f"\n### OUTPUT SCHEMA ###\n```json\n{output_schema_json}\n```"

    return {"text": response_text, "parsed_schemas": parsed_schemas}