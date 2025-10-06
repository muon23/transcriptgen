import logging
import os
from typing import Any, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from llms.Llm import Llm


class GeminiLlm(Llm):
    """
    Concrete implementation of the Llm abstract class for interacting with Google's
    Gemini models (via the ChatGoogleGenerativeAI LangChain wrapper).

    This class handles model-specific configuration, API key management, token limits,
    Google Search grounding integration, and response cleanup.
    """

    __DEFAULT_TOKEN_LIMIT = 1_048_576

    # Model metadata, including token limits and aliases
    __MODELS = {
        # Note: Aliases are used to provide short or previous version names
        "gemini-2.5-pro": {"aliases": ["gemini-2.5"]},
        "gemini-2.5-flash": {},
        "gemini-2.0-flash": {"aliases": ["gemini-2"]},
        "gemini-2.0-flash-thinking-exp-01-21": {"aliases": ["gemini-2t"]},
    }

    # List of all canonical model names
    SUPPORTED_MODELS = list(__MODELS.keys())

    # Mapping from aliases (e.g., 'gemini-2') to canonical names
    MODEL_ALIASES = Llm._alias2model(__MODELS)

    # Dictionary mapping canonical model names to their context window size
    __MODEL_TOKEN_LIMITS = Llm._model_token_limit(__MODELS, __DEFAULT_TOKEN_LIMIT)

    def __init__(
            self,
            model_name: str = "gemini-2",
            model_key: str = None,
            web_search: bool = False,
            **kwargs):
        """
        Initializes the Gemini LLM client.

        Args:
            model_name: The requested model name or alias. Defaults to "gemini-2".
            model_key: The Gemini API key. Searches environment variable if None.
            web_search: If True, binds Google Search tool for grounding.
            **kwargs: Additional parameters passed directly to ChatGoogleGenerativeAI.

        Raises:
            ValueError: If the model is not supported.
            RuntimeError: If the API key is not found.
        """
        # Resolve aliases to the canonical model name
        self.model_name = model_name

        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"LLM model {model_name} not supported")

        logging.info(f"Using {self.model_name}")

        # API Key management
        self.model_key = model_key if model_key else os.environ.get("GEMINI_API_KEY", None)
        if not self.model_key:
            raise RuntimeError(f"Gemini API key not provided")

        # Initialize the base client model
        self.llm_model = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.model_key,
            **kwargs
        )

        # Client initialization (with or without Google Search grounding tool binding)
        # Note: Gemini uses 'google_search' tool specification
        self.llm = self.llm_model.bind_tools([{"google_search": {}}]) if web_search else self.llm_model

        # Call the abstract class constructor
        # Note: Gemini often expects SYSTEM prompts to be mapped to the 'user' role
        # and AI responses to the 'model' role.
        super().__init__(llm=self.llm, role_names={self.Role.SYSTEM: "user", self.Role.AI: "model"})

    def clean_up_response(self, response: Any) -> Llm.Response:
        """
        Cleans up the raw response from the LangChain wrapper.

        Args:
            response: The raw output from the ChatGoogleGenerativeAI runnable.

        Returns:


        Raises:
            TypeError: If the response object is not the expected AIMessage.
        """
        if isinstance(response, AIMessage):
            return Llm.Response(
                text=response.content,
                raw=response
            )

        else:
            raise TypeError(f"Unsupported return type for GptLlm.invoke() (was {type(response)})")

    def get_num_tokens(self, text: str) -> int:
        """
        Calculates the number of tokens in the given text using the ChatGoogleGenerativeAI's
        built-in token counting methods (if available), or defaults to the base class's
        estimation. Since the specific method is not visible here, we return the max limit.
        """
        return self.__DEFAULT_TOKEN_LIMIT

    def get_max_tokens(self) -> int:
        """Returns the maximum context window size for the currently selected model."""
        return self.__MODEL_TOKEN_LIMITS.get(self.model_name, 100_000)

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Returns all supported model names (canonical names and aliases)."""
        return list(cls.MODEL_ALIASES.keys()) + cls.SUPPORTED_MODELS

    def as_runnable(self) -> Runnable:
        """Returns the underlying ChatGoogleGenerativeAI instance as a LangChain Runnable."""
        return self.llm

    def as_language_model(self) -> BaseLanguageModel:
        """Returns the underlying ChatGoogleGenerativeAI instance as a LangChain BaseLanguageModel."""
        return self.llm_model

