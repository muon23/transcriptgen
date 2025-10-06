"""
LLM Package Factory (__init__.py)

This module serves as the primary factory for creating Llm instances.
It transparently selects the correct concrete Llm subclass (e.g., GptLlm, GeminiLlm)
based on the requested model name.
"""
from llms.GeminiLlm import GeminiLlm
from llms.Llm import Llm
from llms.DeepInfraLlm import DeepInfraLlm
from llms.HuggingFaceLlm import HuggingFaceLlm
from llms.GptLlm import GptLlm


def of(model_name: str, **kwargs) -> Llm:
    """
    Factory function to instantiate the correct Llm subclass based on the model name.

    The function iterates through all known Llm subclasses and checks if the
    requested model_name is supported by that class.

    Args:
        model_name: The name of the LLM requested (e.g., 'gpt-4o', 'gemini-2.5').
        **kwargs: Arbitrary keyword arguments passed directly to the constructor
                  of the selected Llm subclass (e.g., API keys, temperature).

    Returns:
        An instantiated object of the correct Llm subclass.

    Raises:
        RuntimeError: If the provided model_name is not supported by any known subclass.
    """
    bots = [GptLlm, GeminiLlm, DeepInfraLlm, HuggingFaceLlm]

    for bot in bots:
        # Check if the model_name is in the list of supported models for this class
        if model_name in bot.get_supported_models():
            # Found the correct provider, instantiate and return
            return bot(model_name, **kwargs)

    # If the loop completes without finding a match
    raise RuntimeError(f"Model {model_name} not supported.")

