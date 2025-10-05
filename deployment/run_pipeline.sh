#!/bin/bash

# --- 1. Determine Script Location ---
# This block robustly finds the absolute directory where this script (run_pipeline.sh)
# is located.
SCRIPT_DIR=$(dirname "$0")

# If $0 is a relative path, this converts it to an absolute path
if [ "${SCRIPT_DIR:0:1}" != "/" ]; then
    SCRIPT_DIR="$(pwd)/$SCRIPT_DIR"
fi

# Define the absolute path to the directory containing orchestrate.py.
# Structure: <project_dir>/deployment (SCRIPT_DIR) -> <project_dir>/src/main/transcriptgen
PYTHON_BASE_DIR="$SCRIPT_DIR/../src/main/transcriptgen"

# Define the absolute path to the project's source root directory for imports (e.g., 'llms' package).
# Structure: <project_dir>/deployment (SCRIPT_DIR) -> <project_dir>/src/main
SOURCE_ROOT_DIR="$SCRIPT_DIR/../src/main"

# Define the absolute path to the Virtual Environment.
# Structure: <project_dir>/deployment (SCRIPT_DIR) -> <project_dir> -> .venv
VENV_PATH="$SCRIPT_DIR/../.venv"

# Define the absolute path to the Python script
PYTHON_SCRIPT="$PYTHON_BASE_DIR/orchestrate.py"


# --- 2. Check and Activate the Virtual Environment ---

# Check if the virtual environment directory exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment directory not found at the project root: $VENV_PATH"
    echo "Please ensure the .venv is located directly under <project_dir> (one level above 'deployment')."
    exit 1
fi

# Activation command differs between OSes/shells.
# We explicitly source the absolute path to the activation script.

ACTIVATE_SCRIPT=""
if [ -f "$VENV_PATH/bin/activate" ]; then
    ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"
elif [ -f "$VENV_PATH/Scripts/activate" ]; then
    ACTIVATE_SCRIPT="$VENV_PATH/Scripts/activate"
fi

if [ -z "$ACTIVATE_SCRIPT" ]; then
    echo "Error: Could not find venv activation script at expected paths."
    exit 1
fi

# Source the activation script
# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"

echo "Activated venv: $VENV_PATH"

# --- 3. Set up PYTHONPATH for custom modules (like 'llms') ---
# Add the source root to the PYTHONPATH so Python can find local packages
# regardless of the current working directory. This replicates IntelliJ's behavior.
export PYTHONPATH="$SOURCE_ROOT_DIR:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"


# --- 4. Run the Python Script ---
# Run the script using the full path. The Python script's Path.cwd()
# will still correctly refer to the directory where the user executed this shell script.
python "$PYTHON_SCRIPT" "$@"

# --- 5. Deactivate the Environment ---
# Important: Deactivate when done to return to the previous shell environment.
deactivate

# Capture the exit status of the Python script to ensure the shell script
# reports success or failure correctly.
EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo "Orchestration pipeline failed."
fi

exit $EXIT_STATUS
