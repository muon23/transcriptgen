import logging
import os
import re
from typing import Any, Tuple, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from llms.Llm import Llm
from llms.HuggingFaceChatRunnable import HuggingFaceChatRunnable
from llms.RunnableToLLMAdapter import RunnableToLLMAdapter


class HuggingFaceLlm(Llm):

    # Model metadata, including token limits and aliases
    __MODELS = {
        "deepseek-ai/DeepSeek-R1": {"aliases": ["deepseek-r1"], "token_limit": 8000},
    }

    # List of all canonical model names
    SUPPORTED_MODELS = list(__MODELS.keys())

    # Mapping from aliases (e.g., 'gpt-3.5') to canonical names ('gpt-3.5-turbo')
    MODEL_ALIASES = Llm._alias2model(__MODELS)

    # Dictionary mapping canonical model names to their context window size
    __MODEL_TOKEN_LIMITS = Llm._model_token_limit(__MODELS, 8000)

    __URL_PREFIX = "https://huggingface.co/"

    def __init__(self, model_name: str = "deepseek", model_key: str = None, **kwargs):
        self.model_name = model_name
        self.role_names = ["system", "user", "assistant"]

        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"LLM model {model_name} not supported")

        logging.info(f"Using {self.model_name}")

        self.model_key = model_key if model_key else os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)
        if not self.model_key:
            raise RuntimeError(f"HuggingFace API token is not provided")

        self.llm = HuggingFaceChatRunnable(
            model_name=self.model_name,
            api_key=self.model_key,
        )

        super().__init__(llm=self.llm)

    @classmethod
    def __separate_think_tag(self, text: str) -> Tuple[str, str]:
        match = re.search(r"(<think>)?(.*?)</think>(.*)", text, re.DOTALL)
        return (match.group(2).strip(), match.group(3).strip()) if match else ("", text)

    def clean_up_response(self, response: Any) -> Llm.Response:
        if isinstance(response, AIMessage):
            content = response.content

        elif isinstance(response, str):
            content = response

        else:
            raise TypeError(f"Unsupported return type for HfLlm.invoke() (was {type(response)})")

        thought, content = self.__separate_think_tag(content)

        return Llm.Response(
            text=content,
            thought=thought,
            raw=response
        )

    def get_max_tokens(self) -> int:
        limit = self.__MODEL_TOKEN_LIMITS.get(self.model_name, 4000)
        return limit

    @classmethod
    def get_supported_models(cls) -> List[str]:
        return list(cls.MODEL_ALIASES.keys()) + cls.SUPPORTED_MODELS

    def as_runnable(self) -> Runnable:
        return self.llm

    def as_language_model(self) -> BaseLanguageModel:
        return RunnableToLLMAdapter(self.as_runnable())
