# pluginfactoryapp/services/vertex_service.py

import vertexai
from vertexai.generative_models import GenerativeModel
from flask import current_app

model = None

def initialize_vertex_ai():
    """Initializes the Vertex AI model."""
    global model
    if model is None:
        try:
            project_id = current_app.config['PROJECT_ID']
            region = current_app.config['REGION']
            model_name = current_app.config['MODEL_NAME']
            
            vertexai.init(project=project_id, location=region)
            model = GenerativeModel(model_name)
            print(f"✅ Vertex AI initialized successfully with {model_name}.")
        except Exception as e:
            print(f"❌ Critical Error: Failed to initialize Vertex AI. Reason: {e}")
            model = None # Ensure model is None on failure

def get_model():
    """Returns the initialized model instance."""
    if model is None:
        raise ConnectionError("Vertex AI model is not initialized or failed to initialize.")
    return model

def generate_content(prompt_text):
    """Generates content using the configured model."""
    try:
        llm_model = get_model()
        response = llm_model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        print(f"❌ Error during content generation: {e}")
        # Depending on desired error handling, you could return None or re-raise
        raise e