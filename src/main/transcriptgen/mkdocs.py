import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path


# --- MOCK LLM MODULE FOR TESTING ---
# So I don't have to pay OpenAI or Google :)
class MockLLM:
    """Simulates an LLM client that returns structured JSON."""

    def __init__(self, model_name):
        self.model_name = model_name
        # Note: The illness key dynamically changes, so we use a generic placeholder here
        self.illness_key = "Eczema"

    def invoke(self, prompt):
        logging.debug(f"MOCK: Invoking model '{self.model_name}'...")
        mock_data = [
            {"zip_code": "10001", "location": "New York, NY", "key_demographics": "High density, diverse, high income.",
             f"{self.illness_key}_care_implications": "Access to specialists high, but costs are a barrier.",
             "doctor_name": "Dr. Sarah Chen", "gender": "Female", "years_in_practice": 7},
            {"zip_code": "78701", "location": "Austin, TX", "key_demographics": "Young, rapid growth, tech-focused.",
             f"{self.illness_key}_care_implications": "Good tech utilization for follow-ups, focus on allergy triggers.",
             "doctor_name": "Dr. David Garcia", "gender": "Male", "years_in_practice": 15},
            {"zip_code": "99501", "location": "Anchorage, AK",
             "key_demographics": "Sparse population, specific Native Alaskan demographics, high seasonal variation.",
             f"{self.illness_key}_care_implications": "Logistical challenges for specialist referrals; telemedicine crucial.",
             "doctor_name": "Dr. Ben Kasuk", "gender": "Male", "years_in_practice": 22},
        ]
        return {"content": json.dumps(mock_data)}


class MockLLMService:
    @staticmethod
    def of(model_name):
        return MockLLM(model_name)
# -----------------------------------------------------------


# Define the supported output formats for validation
SUPPORTED_FORMATS = ["json", "tsv", "parquet"]


def determine_format(output_filepath: str, explicit_format: str) -> tuple[str, str]:
    """
    Determines the final output format based on the explicit_format argument
    or the file extension of the output_filepath.
    The format overrides the extension.
    Defaults to 'json' if neither is provided/inferred.

    Raises:
        NotImplementedError: If the determined format is not in SUPPORTED_FORMATS.

    Returns: A tuple (reconciled_output_file_path, format_string).
    """
    path_obj = Path(output_filepath)

    if explicit_format:
        # Explicit format overrides any inferred extension
        determined_format = explicit_format.lower()
    elif path_obj.suffix:
        # Use file extension if no explicit format is given
        # path_obj.suffix includes the dot (e.g., '.json'), so we strip it
        determined_format = path_obj.suffix[1:].lower()
    else:
        # Default to "json" if no extension was found and no explicit format was given
        determined_format = "json"

    # Validate against supported formats
    if determined_format not in SUPPORTED_FORMATS:
        raise NotImplementedError(
            f"Output format '{determined_format}' not supported. Supported formats are: {', '.join(SUPPORTED_FORMATS)}.")

    # Reconcile the output path using pathlib's methods
    # Get the path without the current suffix (e.g., 'data/file' from 'data/file.ext')
    base_path = path_obj.with_suffix('')

    # Append the new, reconciled suffix (e.g., '.json')
    new_output_path = base_path.with_suffix(f".{determined_format}")

    # Return the string representation of the new path and the determined format string
    return str(new_output_path), determined_format


def save_data(data: list, filepath: str, data_format: str):
    """
    Saves the generated data to the specified file path in the required format,
    creating parent directories if necessary. Uses pathlib and pandas/pyarrow for non-JSON formats.
    """
    path = Path(filepath)

    # 1. Create parent directories if they don't exist
    if not path.parent.exists():
        if path.parent.name:  # Check if it's not the current directory
            logging.info(f"Creating output directory: {path.parent}")
        path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Writing {len(data)} profiles to {filepath} in {data_format} format...")

    if data_format == "json":
        try:
            with open(path, 'w', encoding='utf-8') as f:
                # Use indent=4 for readable JSON output
                json.dump(data, f, indent=4)
            logging.info(f"JSON file written successfully to: {filepath}")
        except IOError as e:
            logging.error(f"Could not write JSON file: {e}")
            sys.exit(1)

    elif data_format in ["tsv", "parquet"]:
        # Check if pandas was imported successfully at the top level
        try:
            import pandas as pd
        except ImportError:
            logging.error(
                f"Pandas is required for {data_format} output but could not be imported. Please install it (e.g., pip install pandas pyarrow)."
            )
            sys.exit(1)

        try:
            # Convert list of dicts to DataFrame for easy export
            df = pd.DataFrame(data)

            if data_format == "tsv":
                # Use tab separator and no index
                df.to_csv(path, sep='\t', index=False)
                logging.info(f"TSV file written successfully to: {filepath}")

            elif data_format == "parquet":
                # Requires pyarrow, which is often installed alongside pandas
                df.to_parquet(path, index=False)
                logging.info(f"Parquet file written successfully to: {filepath}")

        except Exception as e:
            logging.error(f"Could not write {data_format} file: {e}")
            # Add a specific hint for parquet files
            if data_format == "parquet":
                logging.warning("If writing parquet failed, ensure 'pyarrow' is installed.")
            sys.exit(1)

    else:
        # Should be caught by determine_format, but serves as a failsafe
        logging.error(f"Unsupported format encountered: {data_format}")
        sys.exit(1)


def main():
    """
    Main function to parse arguments and simulate the document generation process.
    """
    # Configure basic logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Generates synthetic profiles for fictional doctors, including their specialty and treated illnesses, for use in mock documentation or data analysis.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- REQUIRED ARGUMENTS ---

    # 1. Required Argument: Specialty of the doctors
    parser.add_argument(
        '-s', '--specialty',
        type=str,
        required=True,
        help='The medical specialty of the doctors to generate (e.g., Cardiology, Dermatology).'
    )

    # 2. Required Argument: Illness/Condition
    parser.add_argument(
        '-i', '--illness',
        type=str,
        required=True,
        help='The specific condition or illness being treated (e.g., Hypertension, Eczema).'
    )

    # 3. Required Argument: Output File Name
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='The path and name of the output file (e.g., data/output.json or results.tsv).'
    )

    # --- OPTIONAL ARGUMENTS ---

    # 4. Optional Argument: LLM Model
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='gpt-4o',
        help='LLM model to use for generation (default: gpt-4o).'
    )

    # 5. Optional Argument: Number of Doctors
    parser.add_argument(
        '-d', '--doctors',
        type=int,
        default=20,
        help='Number of doctor profiles to generate (default: 20).'
    )

    # 6. Optional Argument: Output Format
    parser.add_argument(
        '-f', '--format',
        type=str,
        required=False,
        help=(
            f'Format of the output file. Supported: {", ".join(SUPPORTED_FORMATS)}.\n'
            'Default is determined by the OUTPUT file extension. If no extension '
            'is present and --format is not given, defaults to "json".'
        )
    )

    # 7. Optional Argument: Mock Mode (for testing)
    parser.add_argument(
        '-k', '--mock-mode',
        action='store_true',
        help='If set, uses a mock LLM client for testing instead of calling the external LLM API.'
    )

    args = parser.parse_args()

    # Determine the final output format and reconcile the output file path.
    try:
        output_file, final_format = determine_format(args.output, args.format)
    except NotImplementedError as e:
        logging.error(f"Error: {e}")
        sys.exit(1)

    # --- LLM Client Initialization ---
    if args.mock_mode:
        logging.warning("Running in MOCK mode (--mock-mode/-k). No external API calls will be made.")
        bot = MockLLMService.of(args.model)
    else:
        try:
            # Assume the real llms package is available for normal execution
            import llms
            logging.info("Using real LLM service.")
            bot = llms.of(args.model)
        except ImportError:
            logging.error("Could not import the external 'llms' package.")
            logging.warning("Falling back to MOCK mode for execution.")
            llms = None  # To supress IntelliJ warning
            bot = MockLLMService.of(args.model)

    logging.info("--- mkdocs.py Execution Summary ---")
    logging.info(f"Specialty:      {args.specialty}")
    logging.info(f"Illness:        {args.illness}")
    logging.info(f"Output File:    {output_file}")
    logging.info(f"LLM Model:      {args.model}")
    logging.info(f"Num Doctors:    {args.doctors}")
    logging.info(f"Target Format:  {final_format}")
    logging.info("----------------------------------")

    #
    # Starts generation
    #

    prompt = textwrap.dedent(f"""
        <task>
        Generate a mock dataset of {args.doctors} {args.specialty} physicians practicing in diverse regions across the United States.
        </task>
        
        <instructions>
        - Use web search or external knowledge to identify a variety of U.S. ZIP codes and locations with different demographics (e.g., urban, rural, coastal, Midwest, South).
        - For each physician:
          * Assign a plausible name (fictional, but realistic).
          * Specify gender and years in practice (e.g., "12 years").
          * Provide the physician’s ZIP code, city, and state.
          * Summarize key local demographics (e.g., age distribution, socioeconomic factors, ethnic diversity).
          * Describe potential implications for how {args.illness} may be treated or managed in that region, noting similarities and differences across locales.
        - Ensure diversity across the dataset in geography, demographics, and care implications.
        - Output only valid JSON (a list of physician objects) with the following fields:
          - zip_code
          - location
          - key_demographics
          - {args.illness}_care_implications
          - doctor_name
          - gender
          - years_in_practice
        - Do not include ```json or any code block delimiters.
        - Do not include explanations, comments, or extra text.
        </instructions>
        """)

    answer = bot.invoke(prompt)

    try:
        # Load the generated JSON string into a Python list/dict
        profiles = json.loads(answer.text)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON response from LLM: {e}")
        logging.error("Raw response content: %s", answer.get("content", "[None]"))
        sys.exit(1)

    # Write data to output file in the determined format
    save_data(profiles, output_file, final_format)


if __name__ == '__main__':
    # argparse now handles the required nature of -s, -i, and -o options.
    main()
