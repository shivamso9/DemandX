import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

# --- Core Configuration ---
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
MODEL_NAME = os.getenv("MODEL_NAME")


# --- Directory Paths ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
REFERENCE_PLUGIN_DIR = os.getenv("REFERENCE_PLUGIN_DIR")
TENANT_PLUGIN_DIR = os.getenv("TENANT_PLUGIN_DIR")
TEST_PLAN_DIR = os.getenv("TEST_PLAN_DIR")

# --- Git Configuration ---
GIT_REPO_PATH = os.getenv("GIT_REPO_PATH")
GIT_BRANCH = os.getenv("GIT_BRANCH")

# --- Execution Configuration ---
MAX_DEBUG_ATTEMPTS = int(os.getenv("MAX_DEBUG_ATTEMPTS", default=3))

# Get the timeout from the environment variable.
# The `int()` converts the string value to an integer.
# The second argument to os.getenv is a default value if the variable isn't found.
PYTEST_TIMEOUT = int(os.getenv("PYTEST_TIMEOUT", 120))