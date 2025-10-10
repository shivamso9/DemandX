# Agentic Plugin Factory

**Agentic Plugin Factory** is an AI-powered web application that automates the creation, testing, and debugging of software plugins using a multi-agent system. Powered by **Google Vertex AI**, the application interprets business requirements from an Excel file, generates Python code, writes unit tests, and iteratively debugs the code until it passes all tests.

---

## 🚀 Core Features

- **Specification Ingestion**  
  Upload an Excel file containing business logic, input schemas, and output schemas.

- **AI-Powered Validation**  
  A `Validator Agent` reviews the specification for clarity, consistency, and completeness.

- **Automated Code Generation**  
  A `Generator Agent` writes both the plugin’s core logic and tenant-specific implementation.

- **Automated Test Generation**  
  A `Tester Agent` creates `pytest` unit tests based on the specification.

- **Iterative Debugging Loop**  
  A `Debugger Agent` runs tests, analyzes failures, and uses AI to fix bugs in code or tests until all tests pass.

- **Git Integration**  
  A `Git Agent` commits and pushes the validated plugin to a specified Git repository.

- **Web Interface**  
  Simple Flask-based UI to manage the plugin generation workflow.

---

## 🧱 Tech Stack

| Component       | Technology             |
|----------------|-------------------------|
| **Backend**     | Flask                   |
| **AI Platform** | Google Vertex AI (Gemini Pro) |
| **Data Handling** | Pandas               |
| **Testing**     | Pytest                  |
| **Language**    | Python 3.x              |

---

## 📁 Project Structure

~~~plaintext
DemandX/
├── run.py                   # Entry point for the Flask app
├── config.py                # Configuration settings
├── requirements.txt         # Project dependencies
├── .env                     # (Optional) Environment variables
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── routes.py            # API routes
│   ├── utils.py             # Utility functions
│   ├── prompts.py           # AI prompts used by agents
│   ├── static/              # CSS, JS, and images
│   ├── templates/           # HTML templates (Jinja2)
│   └── services/
│       ├── vertex_service.py    # Interface with Vertex AI
│       ├── file_service.py      # File handling logic
│       └── agents/              # Agent implementations
│           ├── validator_agent.py
│           ├── generator_agent.py
│           ├── debugger_agent.py
│           └── git_agent.py
├── generated_plugins/       # Output directory for completed plugins
├── helpers/                 # Reference or example plugins
└── plugin_scripts/          # Tenant-specific plugin scripts
~~~



## ⚙️ Setup and Installation

Follow these steps to set up the project on your local machine.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/shivamso9/DemandX.git
   cd DemandX
   ```

2. **Create Virtual Environment**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### 3. ⚙️ Configure the Application

Open the `config.py` file and update the following variables with your specific paths and settings:

| Variable              | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `PROJECT_ID`          | Your **Google Cloud Project ID**                                            |
| `REGION`              | The **Google Cloud region** for your project (e.g., `us-central1`)         |
| `MODEL_NAME`          | The **Vertex AI model** to use (e.g., `gemini-2.5-pro`)                     |
| `GIT_REPO_PATH`       | The **absolute local path** to the Git repository where plugins will be saved |
| `MAX_DEBUG_ATTEMPTS`  | Number of times the **Debugger Agent** will retry fixing failed tests       |
| `PYTEST_TIMEOUT`      | Maximum time (in seconds) to wait for unit tests to complete                |

### 4 🔐 Google Cloud Authentication

Ensure your environment is authenticated with Google Cloud. For local development, run the following command:

```bash
gcloud auth application-default login
```

### ▶️ How to Run the Application

Once the setup is complete, start the Flask server from the root directory (`/Flaskapp/`):

```bash
python run.py
```

This will launch the app at:

http://127.0.0.1:8080

## 🤖 Agent Workflow

The application employs a series of specialized agents that work in sequence to generate, test, debug, and deploy plugins:

### Validator Agent
- Parses the uploaded Excel file.
- Uses a language model to confirm that the business requirements are clear, consistent, and complete.

### Generator (Coder) Agent
- Takes the validated logic and generates:
  - Python plugin code
  - Tenant-specific implementation
  - Pytest unit tests

### Debugger Agent
- Enters an execution loop:
  1. Runs the generated pytest tests.
  2. If tests **pass**: the plugin is validated.
  3. If tests **fail**:
     - Analyzes error logs.
     - Identifies whether the problem is in the plugin or test code.
     - Invokes the LLM to generate a fix.
  4. Repeats until all tests pass or `MAX_DEBUG_ATTEMPTS` is reached.

### Git Agent
- Once validation is complete:
  - Writes the final plugin files to the configured Git repository.
  - Commits and pushes the changes to the remote branch.
