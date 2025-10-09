import argparse
import logging
import sys
import textwrap
from pathlib import Path


# --- MOCK LLM MODULE FOR TESTING ---
# Provides a fast, repeatable response without calling an external API.
class MockLLM:
    """Simulates an LLM client that returns a plain text string of questions."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.illness_key = "Eczema"

    def invoke(self, prompt):
        logging.debug(f"MOCK: Invoking model '{self.model_name}'...")
        # Simulate LLM response which is a plain text block (questions separated by newlines)
        mock_questions_text = (
            f"How do you typically approach a new patient presenting with {self.illness_key}?\n"
            "What is your preferred first-line treatment and why?\n"
            "How do local socioeconomic factors impact patient adherence to prescribed treatment plans in your area?\n"
            "What common geographical or environmental factors in your region exacerbate this condition?\n"
            f"Describe a challenging case of {self.illness_key} you managed and the multidisciplinary approach used."
        )
        # The real LLM will return raw text based on the prompt instructions.
        return {"content": mock_questions_text}


class MockLLMService:
    @staticmethod
    def of(model_name):
        return MockLLM(model_name)
# -----------------------------------------------------------


def save_data(data: str, filepath: str):
    """
    Saves the generated plain text questions (data) to the specified file path,
    creating parent directories if necessary.
    """
    path = Path(filepath)

    # 1. Create parent directories if they don't exist
    if not path.parent.exists():
        if path.parent.name:  # Check if it's not the current directory
            logging.info(f"Creating output directory: {path.parent}")
        path.parent.mkdir(parents=True, exist_ok=True)

    # Count lines/questions for logging
    question_count = data.count('\n') + 1 if data.strip() else 0
    logging.info(f"Writing {question_count} questions to {filepath} in plain text")

    try:
        with open(path, 'w', encoding='utf-8') as f:
            # Write the raw text content
            f.write(data)
        logging.info(f"Text file written successfully to: {filepath}")
    except IOError as e:
        logging.error(f"Could not write text file: {e}")
        sys.exit(1)


def main():
    """
    Main function to parse arguments and generate the doctor questionnaire.
    """
    # Configure basic logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Generates a synthetic questionnaire for a doctor based on their specialty and a specific illness.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- REQUIRED ARGUMENTS ---

    # 1. Required Argument: Specialty of the doctors
    parser.add_argument(
        '-s', '--specialty',
        type=str,
        required=True,
        help='The medical specialty of the doctor the questionnaire is for (e.g., Cardiology, Dermatology).'
    )

    # 2. Required Argument: Illness/Condition
    parser.add_argument(
        '-i', '--illness',
        type=str,
        required=True,
        help='The specific condition or illness the questions should target (e.g., Hypertension, Eczema).'
    )

    # 3. Required Argument: Output File Name
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='The path and name of the output text file (e.g., questions.txt).'
    )

    # --- OPTIONAL ARGUMENTS ---

    # 4. Optional Argument: Number of Questions (replaces --doctors)
    parser.add_argument(
        '-q', '--questions',
        type=int,
        default=30,
        help='Number of questions to generate (default: 30).'
    )

    # 5. Optional Argument: LLM Model
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='gpt-4o',
        help='LLM model to use for generation (default: gpt-4o).'
    )

    # 6. Optional Argument: Output Format (Removed -f since it is fixed to text)
    # The format argument is removed as it's no longer useful.

    # 7. Turn on web search
    parser.add_argument(
        '-l', '--lookup',
        action='store_true',
        default=False,
        help='If set, turn on web search tool.'
    )

    # 8. Optional Argument: Mock Mode (for testing)
    parser.add_argument(
        '-k', '--mock-mode',
        action='store_true',
        help='If set, uses a mock LLM client for testing instead of calling the external LLM API.'
    )

    args = parser.parse_args()

    # --- LLM Client Initialization ---
    if args.mock_mode:
        logging.warning("Running in MOCK mode (--mock-mode/-k). No external API calls will be made.")
        bot = MockLLMService.of(args.model)
    else:
        try:
            # Attempt to import the external llms package
            import llms
            logging.info("Using real LLM service.")
            bot = llms.of(args.model, web_search=args.lookup)
        except ImportError:
            logging.error("Could not import the external 'llms' package.")
            logging.warning("Falling back to MOCK mode for execution.")
            llms = None  # To supress IntelliJ warning
            bot = MockLLMService.of(args.model)

    logging.info("--- mkquestions.py Execution Summary ---")
    logging.info(f"Specialty:      {args.specialty}")
    logging.info(f"Illness:        {args.illness}")
    logging.info(f"Output File:    {args.output}")
    logging.info(f"LLM Model:      {args.model}")
    logging.info(f"Num Questions:  {args.questions}")
    logging.info("----------------------------------")

    #
    # Starts generation
    #
    num_questions = args.questions
    specialty = args.specialty
    disease = args.illness

    prompt = textwrap.dedent(f"""
        <task>
        Generate {num_questions} realistic and indepth interview questions for an interviewer to ask a {specialty} about their practices in managing {disease}.
        </task>

        <instructions>
        - Use web search or external knowledge to reflect current standards, challenges, and emerging trends in {disease} care.
        - Cover a broad range of themes, including: diagnosis methods, treatment options, patient management, preventive strategies, health disparities, and relevant public health policies.
        - Phrase each question exactly as it would be spoken in an interview.
        - Ensure questions are clear, self-contained, and understandable without additional context.
        - Arrange the questions so that consecutive items naturally build on or complement each other.
        - Output format:
          * Plain text only
          * One question per line
          * No numbering, bullet points, or extra characters
          * No explanations, comments, or code fences
        </instructions>

        <examples>
        How do you typically approach a new patient presenting with {disease}?
        What is your preferred first-line treatment and why?
        What has been your experience with [a_drug_name] in managing {disease}?
        </examples>
        """)

    answer = bot.invoke(prompt)

    if not answer.text:
        logging.error("Received empty response from the LLM. Cannot save data.")
        sys.exit(1)

    # Write data to output file
    save_data(answer.text, args.output)


if __name__ == '__main__':
    main()
