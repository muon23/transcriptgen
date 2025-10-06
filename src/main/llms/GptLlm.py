import logging
import os
from typing import Any, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from llms.Llm import Llm


class GptLlm(Llm):
    """
    Concrete implementation of the Llm abstract class for interacting with OpenAI's
    GPT models (via the ChatOpenAI LangChain wrapper).

    This class handles model-specific configuration, API key management, token limits,
    web search integration, and response cleanup.
    """

    __DEFAULT_TOKEN_LIMIT = 4096

    # Model metadata, including token limits and aliases
    __MODELS = {
        "gpt-5": {"token_limit": 400_000},
        "gpt-4.1": {},
        "o3-mini": {"aliases": ["gpt-o3", "o3-mini"]},
        "o1": {"aliases": ["gpt-o1", "o1"]},
        "gpt-4o": {"token_limit": 16_384},
        "gpt-4o-mini": {"token_limit": 16_384},
        "gpt-4": {"token_limit": 8192},
        "gpt-3.5-turbo": {"aliases": ["gpt-3.5"]},
    }

    # List of all canonical model names
    SUPPORTED_MODELS = list(__MODELS.keys())

    # Models that support web search via LangChain tools binding
    WEB_SEARCH_SUPPORTED = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5"]

    # Mapping from aliases (e.g., 'gpt-3.5') to canonical names ('gpt-3.5-turbo')
    MODEL_ALIASES = Llm._alias2model(__MODELS)

    # Dictionary mapping canonical model names to their context window size
    __MODEL_TOKEN_LIMITS = Llm._model_token_limit(__MODELS, __DEFAULT_TOKEN_LIMIT)

    def __init__(self, model_name: str = "gpt-4", model_key: str = None, web_search: bool = False, **kwargs):
        """
        Initializes the GPT LLM client.

        Args:
            model_name: The requested model name or alias. Defaults to "gpt-4".
            model_key: The OpenAI API key. Searches environment variable if None.
            web_search: If True, binds web search tools to supported models.
            **kwargs: Additional parameters passed directly to ChatOpenAI (e.g., temperature).

        Raises:
            ValueError: If the model is not supported.
            RuntimeError: If the API key is not found.
            NotImplementedError: If web search is requested for an unsupported model.
        """
        # Resolve aliases to the canonical model name
        if model_name in self.MODEL_ALIASES:
            model_name = self.MODEL_ALIASES[model_name]

        self.model_name = model_name

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"LLM model {model_name} not supported")

        # Determine custom role names for specific OpenAI models if necessary
        # The 'o' series models sometimes require the system prompt to be mapped to 'user' role
        if self.model_name in ["o1", "o1-preview", "o3-mini"]:
            kwargs["temperature"] = 1  # These models removed temperature.  Setting it to anything else will fail.
            role_names = {Llm.Role.SYSTEM: "user"}
        else:
            role_names = {}

        logging.info(f"Using {self.model_name}")

        # API Key management
        self.model_key = model_key if model_key else os.environ.get("OPENAI_API_KEY", None)
        if not self.model_key:
            raise RuntimeError(f"OpenAI API key not provided")

        # Client initialization (with or without web search binding)
        if web_search:
            if self.model_name not in self.WEB_SEARCH_SUPPORTED:
                raise NotImplementedError(f"Web search is not supported by {self.model_name}")

            # Initialize ChatOpenAI and bind web search tool
            self.llm = ChatOpenAI(
                model_name=self.model_name, openai_api_key=self.model_key,
                use_responses_api=True, **kwargs
            ).bind_tools([{"type": "web_search"}])
        else:
            # Standard initialization
            self.llm = ChatOpenAI(model_name=self.model_name, openai_api_key=self.model_key, **kwargs)

        # Call the abstract class constructor
        super().__init__(llm=self.llm, role_names=role_names)

    def clean_up_response(self, response: Any) -> dict:
        """
        Cleans up the raw response from the LangChain wrapper.

        Args:
            response: The raw output from the ChatOpenAI runnable.

        Returns:
            A dictionary containing the cleaned response content and metadata.

        Raises:
            TypeError: If the response object is not the expected AIMessage.
        """
        if isinstance(response, AIMessage):
            return {
                "content": response.text(),
                "metadata": response
            }

        else:
            raise TypeError(f"Unsupported return type for GptLlm.invoke() (was {type(response)})")

    def get_num_tokens(self, text: str) -> int:
        """
        Calculates the number of tokens in the given text using the ChatOpenAI's built-in tokenizer.
        """
        return ChatOpenAI(temperature=0).get_num_tokens(text)

    def get_max_tokens(self) -> int:
        """Returns the maximum context window size for the currently selected model."""
        # Use the stored limits, defaulting to 100,000 for safety if lookup fails
        return self.__MODEL_TOKEN_LIMITS.get(self.model_name, 100_000)

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Returns all supported model names (canonical names and aliases)."""
        return list(cls.MODEL_ALIASES.keys()) + cls.SUPPORTED_MODELS

    def as_runnable(self) -> Runnable:
        """Returns the underlying ChatOpenAI instance as a LangChain Runnable."""
        return self.llm

    def as_language_model(self) -> BaseLanguageModel:
        """Returns the underlying ChatOpenAI instance as a LangChain BaseLanguageModel."""
        return self.llm

