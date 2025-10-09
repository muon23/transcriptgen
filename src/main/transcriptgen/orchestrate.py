import argparse
import logging
import sys
from pathlib import Path
from typing import List
import subprocess

# --- Configuration ---
DOC_SCRIPT = "mkdocs.py"
Q_SCRIPT = "mkquestions.py"
TRANSCRIPT_SCRIPT = "mktranscripts.py"

# Define the absolute directory where this script (and its siblings) reside
SCRIPT_HOME_DIR = Path(__file__).parent.resolve()

# ---------------------

def run_script(script_name: str, args: List[str]):
    """
    Helper function to run subprocesses and handle errors.

    For mktranscripts.py, we redirect stderr to the parent's stderr (None)
    to show the tqdm progress bar. For others, we capture it.
    """

    try:
        # Use the absolute path to the script file
        script_path = SCRIPT_HOME_DIR / script_name

        # Construct the full command: python /path/to/script.py arg1 arg2 ...
        command = [sys.executable, str(script_path)] + args

        logging.info(f"Running command: {' '.join(command)}")

        # For mktranscripts, stderr is inherited (shows tqdm).
        # For others, stderr is captured.
        result = subprocess.run(
            command,
            check=True
        )

        # Log standard output for all scripts
        if result.stdout:
            logging.info(f"--- {script_name} Output Start ---")
            logging.info(result.stdout.strip())
            logging.info(f"--- {script_name} Output End ---")

    except subprocess.CalledProcessError as e:
        logging.error(f"{script_name} failed with return code {e.returncode}.")

        # Display the captured stderr if available
        if e.stderr:
            logging.error(f"Captured STDERR:\n{e.stderr.strip()}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error(
            f"Could not find required script file: {script_name}. Ensure all scripts are in the current directory.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while running {script_name}: {e}")
        sys.exit(1)


def main():
    """
    Main function to parse arguments and orchestrate the document generation pipeline.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    parser = argparse.ArgumentParser(
        description="Master script to generate doctor profiles, interview questions, and transcripts in one go.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- REQUIRED POSITIONAL ARGUMENTS ---
    parser.add_argument(
        'specialty',
        type=str,
        help='The medical specialty of the doctors (e.g., Cardiology, Dermatology).'
    )

    parser.add_argument(
        'illness',
        type=str,
        help='The specific condition or illness being treated (e.g., Hypertension, Eczema).'
    )

    # --- OPTIONAL ARGUMENTS ---
    parser.add_argument(
        '-w', '--working',
        type=str,
        default=None,
        help='The base working directory for all input and output files. Defaults to the current directory.'
    )

    parser.add_argument(
        '-d', '--doctors',
        type=int,
        default=20,
        help='Number of doctor profiles to generate (default: 20).'
    )

    parser.add_argument(
        '-q', '--questions',
        type=int,
        default=30,
        help='Number of interview questions to generate (default: 30).'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        default='gpt-4o',
        help='LLM model to use for all generation steps (default: gpt-4o).'
    )

    # Optional arguments that should be propagated to all sub-scripts
    parser.add_argument(
        '-k', '--mock-mode',
        action='store_true',
        help='If set, uses a mock LLM client for all scripts (recommended for testing).'
    )

    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Force overwrite of existing files in the transcripts step.'
    )

    # Optional arguments that should only be propagated to mktranscripts.py
    parser.add_argument(
        '-c', '--chunk-size',
        type=int,
        default=None, # None so we only propagate if explicitly set
        help='Question chunk size for mktranscripts.py. (e.g., -c 5).'
    )

    parser.add_argument(
        '-r', '--random',
        type=int,
        default=None, # None so we only propagate if explicitly set
        help='Random sample N doctors for mktranscripts.py.'
    )

    parser.add_argument(
        '-l', '--lookup',
        action='store_true',
        default=False,
        help='If set, turn on web search tool.'
    )

    args = parser.parse_args()

    # --- Directory Setup ---
    # Determine the working directory
    working_dir = Path(args.working).resolve() if args.working else Path.cwd()

    # Create the highly specific output directory structure
    output_base = working_dir / args.model / args.specialty.lower() / args.illness.lower()

    output_base.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory created/verified: {output_base}")

    # Define file paths
    doctors_file = output_base / "doctors.json"
    questions_file = output_base / "questions.txt"
    transcripts_dir = output_base / "transcripts"  # mktranscripts.py expects an output directory

    # --- Common Arguments for all scripts ---
    common_args = []
    if args.mock_mode:
        common_args.append('-k')
    if args.model:
        common_args.extend(['-m', args.model])

    # --- Arguments only for mktranscripts.py ---
    transcript_specific_args = []
    if args.force:
        transcript_specific_args.append('-f')
    if args.chunk_size is not None:
        transcript_specific_args.extend(['-c', str(args.chunk_size)])
    if args.random is not None:
        transcript_specific_args.extend(['-r', str(args.random)])

    # --- Whether to search the web
    if args.lookup:
        common_args.append('-l')

    # --- Execution Pipeline ---

    # 1. Run mkdocs.py to generate doctor profiles
    logging.info(f"\n--- Step 1/3: Generating doctor profiles using {DOC_SCRIPT} ---")

    doc_args = [
                   '-s', args.specialty,
                   '-i', args.illness,
                   '-o', str(doctors_file),
                   '-d', str(args.doctors),
               ] + common_args

    if doctors_file.is_file() and not args.force:
        logging.warning(f"Doctor profiles file already exists: {doctors_file}. Skipping {DOC_SCRIPT}. Use -f to force overwrite.")
    else:
        run_script(DOC_SCRIPT, doc_args)

    # 2. Run mkquestions.py to generate interview questions
    logging.info(f"\n--- Step 2/3: Generating interview questions using {Q_SCRIPT} ---")

    q_args = [
                 '-s', args.specialty,
                 '-i', args.illness,
                 '-o', str(questions_file),
                 '-q', str(args.questions),
            ] + common_args

    if questions_file.is_file() and not args.force:
        logging.warning(f"Questions file already exists: {questions_file}. Skipping {Q_SCRIPT}. Use -f to force overwrite.")
    else:
        run_script(Q_SCRIPT, q_args)


    # 3. Run mktranscripts.py to generate transcripts
    logging.info(f"\n--- Step 3/3: Generating transcripts using {TRANSCRIPT_SCRIPT} ---")

    transcript_args = [
                          '-s', args.specialty,
                          '-i', args.illness,
                          '-d', str(doctors_file),
                          '-q', str(questions_file),
                          '-o', str(transcripts_dir),
                      ] + common_args + transcript_specific_args

    # We always run mktranscripts.py, as it handles its own internal skipping/forcing for individual files.
    run_script(TRANSCRIPT_SCRIPT, transcript_args)

    logging.info("\n--- PIPELINE COMPLETE ---")
    logging.info(f"Generated data is available in: {output_base}")


if __name__ == '__main__':
    main()
