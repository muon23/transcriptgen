from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Any, List, Dict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, Runnable, RunnableConfig
from transformers import AutoTokenizer


class Llm(ABC):
    """
    Abstract class for accessing LLM models.

    This class provides a standardized interface for interacting with different LLM
    providers (like OpenAI or Gemini) using the LangChain framework, handling
    prompt preprocessing, token counting, and configuration management.
    Concrete subclasses must implement provider-specific logic.
    """

    class Role(Enum):
        """Enumeration for standard conversational roles."""
        SYSTEM = 0
        HUMAN = 1
        AI = 2

    def __init__(self, llm: Runnable, role_names: dict = None):
        """
        Initializes the LLM wrapper.

        Args:
            llm: The underlying LangChain Runnable object for the specific LLM.
            role_names: Optional dictionary to map internal Role enums to custom
                        string names (e.g., HUMAN -> "user", AI -> "assistant").
        """
        self.llm = llm
        # Default role names used for converting history into LangChain format
        self.role_names = {
            self.Role.SYSTEM: "system",
            self.Role.HUMAN: "user",
            self.Role.AI: "assistant",
        }
        if role_names:
            # Overwrite defaults if custom roles are provided
            self.role_names.update(role_names)

        # Use GPT-2 tokenizer as a robust, universal approximation for token counting
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def invoke(self, prompt: Sequence[tuple[Role | str, str] | str] | str, **kwargs) -> dict:
        """
        Processes the prompt, executes the LLM chain, and returns the cleaned response.

        Args:
            prompt: The input prompt. Can be a simple string or a sequence of
                    (Role/role_name, content) tuples representing chat history.
            **kwargs: Additional arguments passed to the chain, including
                      'arguments' (for prompt template values) and 'task' (for metadata).

        Returns:
            A dictionary containing the cleaned response from the LLM.
        """
        # Format the prompt into a LangChain ChatPromptTemplate object
        prompt = self.preprocess_prompt(prompt)

        # Prompt template parameters for filling holes in the prompt
        arguments = kwargs.get("arguments", {})

        # Create a sequential chain: Prompt -> LLM -> Response Cleanup
        chain = RunnableSequence(prompt | self.llm | self.clean_up_response)

        # Task, e.g., chat or completion.
        # Some LLM models need to distinguish chat or completion.
        # This is a way for the derived class to pass in its purpose.
        task = kwargs.get("task", self.get_default_task())
        config = RunnableConfig(metadata={"task": task})

        # Execute the chain
        response = chain.invoke(input=arguments, config=config, **kwargs)

        return response

    def preprocess_prompt(self, prompt: Sequence[tuple[Role | str, str] | str] | str) -> ChatPromptTemplate:
        """
        Converts various input prompt formats into a standardized ChatPromptTemplate.
        """
        # If the prompt is a simple string, wrap it as a single message
        if isinstance(prompt, str):
            # This handles single instruction prompts
            human_role_name = self.role_names[self.Role.HUMAN]
            return ChatPromptTemplate(messages=[(human_role_name, prompt)])
        else:
            # Reformat the sequence of (Role, content) tuples into LangChain messages
            messages = []
            for msg in prompt:
                role_name = self.role_names[msg[0]] if isinstance(msg[0], self.Role) else msg[0]
                messages.append((role_name, msg[1]))
            return ChatPromptTemplate(messages=messages)

    @abstractmethod
    def clean_up_response(self, response: Any) -> dict:
        """
        Abstract method to clean up and standardize the LLM's raw response.

        Concrete implementation must extract text, cost metadata, etc., and return a dict.
        """
        pass

    def get_num_tokens(self, text: str) -> int:
        """
        Estimates the number of tokens in a given text using the GPT-2 tokenizer.
        """
        return len(self.tokenizer.encode(text))

    @abstractmethod
    def get_max_tokens(self) -> int:
        """
        Abstract method to return the maximum context window size for the underlying model.
        """
        pass

    def get_default_task(self) -> str:
        """Returns the default task type for metadata."""
        return "chat"

    @classmethod
    @abstractmethod
    def get_supported_models(cls) -> List[str]:
        """Abstract method to return a list of model names supported by the subclass."""
        pass

    #
    # LangChain adapters (Allowing the Llm object to be used seamlessly in LangChain chains)
    #
    @abstractmethod
    def as_runnable(self) -> Runnable:
        """Returns the LLM instance as a LangChain Runnable."""
        pass

    @abstractmethod
    def as_language_model(self) -> BaseLanguageModel:
        """Returns the LLM instance as a LangChain BaseLanguageModel."""
        pass

    #
    # Helper functions
    #
    @classmethod
    def _alias2model(cls, models: Dict[str, dict]) -> Dict[str, str]:
        """Helper to create a mapping from model aliases to canonical model names."""
        a2m = dict()
        for model, properties in models.items():
            aliases = properties.get("aliases", [])
            for alias in aliases:
                a2m[alias] = model
        return a2m

    @classmethod
    def _model_token_limit(cls, models: Dict[str, dict], default: int) -> Dict[str, int]:
        """Helper to extract token limits from model configuration data."""
        limits = dict()
        for model, properties in models.items():
            limits[model] = properties.get("token_limit", default)
        return limits



