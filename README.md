# DemandX
# o9 Agentic Plugin Factory

This project is a web-based application, "o9 Agentic Plugin Factory," that uses a series of AI agents to automate the creation of data transformation plugins from an Excel-based specification.

## Features

- **Stage 1: Inspector Agent**: Validates the uploaded Excel specification (`plugin_template.xlsx`) and generates a business logic plan.
- **Stage 1a: Reference Plugin Selection**: Allows the user to select an existing plugin as a structural reference for the new one.
- **Stage 2: Test Architect Agent**: Automatically generates `pytest` tests based on the approved logic.
- **Stage 3: Coder Agent**: Writes the plugin code (Repo and Tenant files) based on the business logic and the structure of the selected reference plugin.
- **Stage 4: Reporter & Debugger Agents**: Executes the generated tests against the generated code, attempts to debug any failures, and reports the final results.

## Prerequisites

1.  **Python 3.8+**: Ensure you have a modern version of Python installed.
2.  **Google Cloud Project**:
    - A Google Cloud project with the Vertex AI API enabled.
    - The `gcloud` CLI installed and authenticated.
3.  **Reference Plugin Files**: The application requires at least one reference plugin for the code generation step. For the demo, you must have the following two files in the project's root directory:
    - `DP009OutlierCleansing-Repo.py`
    - `DP009OutlierCleansing-Tenant.py`

## Setup and Installation

1.  **Clone the repository (or download the files):**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**

    A `requirements.txt` file is provided for convenience.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Authenticate with Google Cloud:**

    Log in with your Google Cloud account to provide Application Default Credentials for the backend to use Vertex AI.

    ```bash
    gcloud auth application-default login
    ```

5.  **Configure the Backend (Optional):**

    Open `backend.py` and update the following variables if you are not using the default project:

    ```python
    PROJECT_ID = "your-gcp-project-id"
    REGION = "us-central1" # Or your preferred region
    ```

## Running the Application

1.  **Start the Flask Backend Server:**

    Run the `backend.py` script from your terminal.

    ```bash
    python backend.py
    ```

    You should see output indicating that the Flask server is running and Vertex AI has been initialized successfully. The server will run on `http://127.0.0.1:8080`.

2.  **Open the Frontend:**

    Open the `index.html` file directly in your web browser (e.g., Chrome, Firefox). The frontend is self-contained and will automatically connect to the backend server running locally.

3.  **Use the Application:**
    - Follow the on-screen stages, starting with uploading a `plugin_template.xlsx` file.