import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Pandas is now a required, direct import
import pandas as pd
from tqdm import tqdm  # Using standard tqdm for console progress bars

from llms import Llm


# --- MOCK LLM MODULE FOR TESTING (Required for runnable main function) ---
class MockLLM:
    """Simulates an LLM client that returns unique, structured text."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.call_count = 0

    def invoke(self, prompt_history: List[Tuple[str, str]], arguments: Dict[str, Any] = None) -> Llm.Response:
        """Generates a mock transcript snippet based on the current context."""
        self.call_count += 1

        # Extract doctor's name for unique content
        doctor_name = arguments.get('doctor_profile', {}).get('doctor_name', 'Dr. Mock')

        # Get the latest user message (questions)
        latest_user_content = prompt_history[-1][1]

        # Create a mock response
        response = f"{doctor_name} (Mock Reply #{self.call_count}): This is a simulated response based on the profile of {doctor_name} and the following questions/context:\n---\n{latest_user_content}\n---"

        return Llm.Response(text=response)


class MockLLMService:
    @staticmethod
    def of(model_name: str):
        return MockLLM(model_name)


# ----------------------------------------------------------------------


SUPPORTED_DOCTOR_FORMATS = ["json", "parquet", "tsv"]

# Fields considered essential for constructing the profile prompt
DOCTOR_PROFILE_ESSENTIAL_FIELDS = [
    "doctor_name",
    "gender",
    "years_in_practice",
    "location",
    "zip_code",
    # We include all fields here to provide rich context to the LLM
    "key_demographics",
    # Keys below are dynamic, but kept for reference
    # "specialty_key",
    # "disease_key",
    # "care_implications_key",
]

def load_doctors_profiles(filepath: Path) -> pd.DataFrame:
    """
    Loads doctor profiles from the specified file path into a pandas DataFrame,
    inferring the format from the file extension.
    """
    file_format = filepath.suffix[1:].lower() if filepath.suffix else None

    if not filepath.exists():
        logging.error(f"Doctor profiles file not found: {filepath}")
        sys.exit(1)

    if file_format == "json":
        try:
            df = pd.read_json(filepath)
        except Exception as e:
            logging.error(f"Failed to read JSON file '{filepath}': {e}")
            sys.exit(1)
    elif file_format == "parquet":
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logging.error(f"Failed to read Parquet file '{filepath}'. Ensure 'pyarrow' is installed: {e}")
            sys.exit(1)
    elif file_format == "tsv":
        try:
            df = pd.read_csv(filepath, sep='\t')
        except Exception as e:
            logging.error(f"Failed to read TSV file '{filepath}': {e}")
            sys.exit(1)
    else:
        logging.error(
            f"Unsupported format for doctor profiles: .{file_format}. Supported formats: {', '.join(SUPPORTED_DOCTOR_FORMATS)}")
        sys.exit(1)

    logging.info(f"Successfully loaded {len(df)} doctor profiles from {filepath}.")

    return df


def load_questions(filepath: Path) -> list[str]:
    """
    Loads interview questions from a plain text file, assuming one question per line.
    Empty or whitespace-only lines are ignored.
    """
    if not filepath.exists():
        logging.error(f"Questions file not found: {filepath}")
        sys.exit(1)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]

        if not questions:
            logging.error(f"Questions file '{filepath}' is empty or contains no valid questions.")
            sys.exit(1)

        logging.info(f"Successfully loaded {len(questions)} interview questions from {filepath}.")
        return questions

    except IOError as e:
        logging.error(f"Failed to read questions file '{filepath}': {e}")
        sys.exit(1)


def get_transcript_file_name(doctor_profile: Dict[str, Any]) -> str:
    """
    Creates a unique, file-safe name for the transcript based on the doctor's name (without titles)
    and their location (city and state).
    """
    # 1. Clean Doctor Name
    raw_name = doctor_profile.get("doctor_name", "unknown_doctor")

    # List of common titles/suffixes to remove
    titles_to_remove = ["Dr.", "MD", "D.O.", "PhD", "Jr.", "Sr.", "DO", "MD,", "D O", "PHD"]
    clean_name = raw_name

    for title in titles_to_remove:
        # Replace title (and potential space/comma) with nothing, handling case variations
        clean_name = clean_name.replace(title, "").replace(title.lower(), "")

        # Remove any extra spaces/punctuation, normalize spaces to underscores, and convert to lower
    clean_name = clean_name.strip()
    clean_name = "".join(c for c in clean_name if c.isalnum() or c.isspace()).strip()
    clean_name = clean_name.replace(" ", "_").lower()

    # Use the first 20 characters of the name part to prevent overly long filenames
    name_part = clean_name[:20].strip('_')
    if not name_part:
        name_part = "doctor"  # Fallback if name is empty after cleaning

    # 2. Clean Location (e.g., "Bronx, New York" -> "bronx_new_york")
    raw_location = doctor_profile.get("location", "Unknown_Location")

    # Split by comma, clean each part (city/state), and rejoin with a single underscore
    location_parts = [part.strip() for part in raw_location.split(',')]

    # Clean up spaces in each part and join them with a single underscore
    location_part = '_'.join([
        p.replace(' ', '_').replace('.', '').lower()
        for p in location_parts if p
    ])

    # 3. Combine and return
    return f"{name_part}_{location_part}.txt"


def make_transcript(
        doctor_profile: pd.Series,
        questions: List[str],
        bot: MockLLM,  # Type hinting with MockLLM for clarity, though it could be `Any`
        question_chunk_size: int,
        specialty: str,
        illness: str,
) -> str:
    """
    Generates a full interview transcript for a single doctor by iteratively
    sending question chunks to the LLM.
    """
    # 1. Construct the System Prompt (The instructions for the LLM's role)
    doctor_name = doctor_profile['doctor_name']
    transcript_system_prompt = textwrap.dedent(f"""
        <task>
        Simulate a realistic interview transcript between an interviewer and a {specialty}, focusing on how the physician manages {illness} in their specific regional and demographic context.
        </task>

        <instructions>
        - Use web search or external medical knowledge to understand how regional demographics, socioeconomic factors, and healthcare infrastructure influence {illness} care in the physician’s area.
        - Incorporate these contextual factors naturally into the physician’s responses (e.g., access to care, climate, environment, patient population characteristics, insurance coverage, public health resources, or cultural attitudes toward treatment).
        - The transcript must read as a natural, professional, back-and-forth conversation between the interviewer and the physician.
        - The physician’s replies should sound grounded in practice experience — balancing clinical knowledge, local realities, and personal insight.
        - Use the provided physician profile to guide tone, expertise, and geographic perspective.
        - The interviewer will ask roughly {len(questions)} questions, provided iteratively by the user in chunks.
        - Maintain conversational continuity across iterations, referencing previous remarks where appropriate.
        - Rephrase, merge, or slightly expand user-provided questions when needed for smoother flow or deeper insight.
        - Only generate dialogue lines (no narration, commentary, or stage directions).
        - Use the following structure exactly:
          Interviewer: [question or comment]
          {doctor_name}: [response]
        - The user will mark the interview boundaries with [start interview] and [end interview].
          Continue the conversation naturally between these markers as directed.
        </instructions>

        <doctor_profile>
        {{doctor_profile}}
        </doctor_profile>
        """)

    # Initialize the prompt history with the system instruction
    transcript_prompt: List[Tuple[str, str]] = [
        ("system", transcript_system_prompt),
    ]

    # 2. Format the output header
    transcript = "## Physician's Profile:\n"
    # Show specialty and illness in the header for clarity
    transcript += f"   Specialty: {specialty}\n"
    transcript += f"   Illness:   {illness}\n"

    for key, value in doctor_profile.items():
        transcript += f"   {key}: {value}\n"

    transcript += "\n## Transcript:\n\n"

    # 3. Iterate through questions in chunks and interact with the LLM
    question_chunks = range(0, len(questions), question_chunk_size)

    for i, start_index in enumerate(tqdm(question_chunks, desc=f"Interviewing {doctor_name}", leave=False)):
        end_index = start_index + question_chunk_size
        next_questions = questions[start_index: end_index]

        start_interview = "[start interview]\n" if i == 0 else ""
        end_interview = "[end interview]\n" if end_index >= len(questions) else ""

        user_message = start_interview + '\n'.join(next_questions) + end_interview

        transcript_prompt.append(("user", user_message))

        answer = bot.invoke(transcript_prompt, arguments={"doctor_profile": doctor_profile.to_dict()})

        transcript_prompt.append(("assistant", answer.text))
        transcript += answer.text + "\n\n"

    return transcript


def main():
    """
    Main function to parse arguments and drive the transcript generation process.
    """
    # Configure basic logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Generates mock interview transcripts between an interviewer and a doctor, synthesizing responses based on doctor profiles and a question list.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- REQUIRED ARGUMENTS ---

    # 1. Required Argument: Specialty
    parser.add_argument(
        '-s', '--specialty',
        type=str,
        required=True,
        help='The medical specialty of the doctors (e.g., Cardiology, Dermatology).'
    )

    # 2. Required Argument: Illness/Condition (used as 'disease' in the prompt)
    parser.add_argument(
        '-i', '--illness',
        type=str,
        required=True,
        help='The specific condition or illness being treated (e.g., Hypertension, Eczema).'
    )

    # 3. Optional Working Directory
    parser.add_argument(
        '-w', '--working',
        type=str,
        required=False,
        help='The base working directory for all input and output files.'
    )

    # 4. Input: Doctor Profiles File (default: doctors.json)
    parser.add_argument(
        '-d', '--doctors',
        type=str,
        required=False,  # Now optional if -w is provided
        help='Path to the doctor profiles file (JSON, TSV, or Parquet). Defaults to <working>/doctors.json.'
    )

    # 5. Input: Interview Questions File (default: questions.txt)
    parser.add_argument(
        '-q', '--questions',
        type=str,
        required=False,  # Now optional if -w is provided
        help='Path to the text file containing the interview questions. Defaults to <working>/questions.txt.'
    )

    # 6. Output: Output Directory (default: transcripts)
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=False,  # Now optional if -w is provided
        help='Path to the directory where transcript files are stored. Defaults to <working>/transcripts.'
    )

    # --- OPTIONAL ARGUMENTS (General) ---

    # 7. Optional Argument: LLM Model
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='gpt-4o',
        help='LLM model to use for transcript generation (default: gpt-4o).'
    )

    # 8. Optional Argument: Mock Mode (for testing)
    parser.add_argument(
        '-k', '--mock-mode',
        action='store_true',
        help='If set, uses a mock LLM client for testing instead of calling the external LLM API.'
    )

    # 9. Optional Argument: Question Chunk Size
    parser.add_argument(
        '-c', '--chunk-size',
        type=int,
        default=5,
        help='Number of questions to send to the LLM in each conversational turn (default: 5).'
    )

    # 10. Optional Argument: Random Sample Size
    parser.add_argument(
        '-r', '--random',
        type=int,
        default=None,
        help='Randomly sample N doctors from the input file instead of iterating through all of them.'
    )

    # 11. Optional Argument: Force Overwrite
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Force generation, overwriting any existing transcript files in the output directory.'
    )

    args = parser.parse_args()

    # --- PATH RESOLUTION AND SETUP ---

    # Set the base directory: use working directory if provided, otherwise current working directory
    base_dir = Path(args.working) if args.working else Path.cwd()
    logging.info(f"Base Directory set to: {base_dir.resolve()}")

    # 1. Resolve Doctor Profiles Path
    doctors_file_arg = args.doctors if args.doctors else 'doctors.json'
    doctors_path = (base_dir / doctors_file_arg).resolve()

    # 2. Resolve Questions File Path
    questions_file_arg = args.questions if args.questions else 'questions.txt'
    questions_path = (base_dir / questions_file_arg).resolve()

    # 3. Resolve Output Directory Path
    output_dir_arg = args.output if args.output else 'transcripts'
    output_dir = (base_dir / output_dir_arg).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    # 4. Set Chunk Size (Crucial for conversation flow)
    # Validate and use the new argument
    question_chunk_size = args.chunk_size
    if question_chunk_size < 1:
        logging.error("Chunk size must be 1 or greater.")
        sys.exit(1)

    logging.info("--- mktranscripts.py Execution Summary ---")
    logging.info(f"Specialty:            {args.specialty}")
    logging.info(f"Illness:              {args.illness}")
    logging.info(f"Doctor Profiles File: {doctors_path}")
    logging.info(f"Questions File:       {questions_path}")
    logging.info(f"Output Directory:     {output_dir}")
    logging.info(f"LLM Model:            {args.model}")
    logging.info(f"Mock Mode:            {args.mock_mode}")
    logging.info(f"Question Chunk Size:  {question_chunk_size}")
    logging.info(f"Random Sample Size:   {args.random if args.random is not None else 'All'}")
    logging.info(f"Force Overwrite:      {args.force}")
    logging.info("------------------------------------------")

    # 1. Load Data
    doctor_df = load_doctors_profiles(doctors_path)
    questions_list = load_questions(questions_path)

    # Apply random sampling if requested
    if args.random is not None:
        if args.random <= 0:
            logging.error("Random sample size must be a positive integer.")
            sys.exit(1)
        if args.random > len(doctor_df):
            logging.warning(
                f"Requested sample size ({args.random}) is larger than the total number of doctors ({len(doctor_df)}). Using all doctors.")
            # If greater, we proceed with the full DataFrame.
        else:
            logging.info(f"Randomly sampling {args.random} doctors.")
            # Use pandas sample method to get a random subset, using a fixed state for potential reproducibility
            doctor_df = doctor_df.sample(n=args.random, random_state=42)

    # 2. Initialize LLM Client
    if args.mock_mode:
        logging.warning("Running in MOCK mode (--mock-mode/-k). No external API calls will be made.")
        bot = MockLLMService.of(args.model)
    else:
        try:
            import llms
            logging.info("Using real LLM service.")
            bot = llms.of(args.model)
        except ImportError:
            logging.error("Could not import the external 'llms' package.")
            logging.warning("Falling back to MOCK mode for execution.")
            llms = None  # To supress IntelliJ warning
            bot = MockLLMService.of(args.model)

    # 3. Main Generation Loop
    logging.info(f"Starting generation of {len(doctor_df)} transcripts...")

    for _, doctor_profile in tqdm(doctor_df.iterrows(), total=len(doctor_df), desc="Physician Progress"):
        doctor_name = doctor_profile['doctor_name']

        transcript_filename = get_transcript_file_name(doctor_profile)
        transcript_file_path = output_dir / transcript_filename

        # Check for existence and skip if not forcing
        if transcript_file_path.is_file() and not args.force:
            logging.warning(
                f"Transcript for {doctor_name} already exists. Skipping: {transcript_filename} (Use --force/-f to overwrite)")
            continue

        try:
            transcript = make_transcript(
                doctor_profile=doctor_profile,
                questions=questions_list,
                bot=bot,
                question_chunk_size=question_chunk_size,
                specialty=args.specialty,
                illness=args.illness  # Pass the required arguments to the generator
            )

            transcript_file_path.write_text(transcript, encoding='utf-8')
            logging.info(f"Successfully wrote transcript: {transcript_filename}")

        except Exception as e:
            logging.error(f"Failed to generate/write transcript for {doctor_name}: {e}")

    logging.info("Transcript generation complete.")


if __name__ == '__main__':
    main()
