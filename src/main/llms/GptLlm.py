import logging
import os
from typing import Any, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from llms.Llm import Llm


class GptLlm(Llm):
    __DEFAULT_TOKEN_LIMIT = 4096

    __MODELS = {
        "gpt-5": {},
        "gpt-4.1": {},
        "o3-mini": {"aliases": ["gpt-o3", "o3-mini"]},
        "o1": {"aliases": ["gpt-o1", "o1"]},
        "gpt-4o": {"token_limit": 16_384},
        "gpt-4o-mini": {"token_limit": 16_384},
        "gpt-4": {"token_limit": 8192},
        "gpt-3.5-turbo": {"aliases": ["gpt-3.5"]},
    }

    SUPPORTED_MODELS = list(__MODELS.keys())

    WEB_SEARCH_SUPPORTED = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5"]

    MODEL_ALIASES = Llm._alias2model(__MODELS)

    __MODEL_TOKEN_LIMITS = Llm._model_token_limit(__MODELS, __DEFAULT_TOKEN_LIMIT)

    def __init__(self, model_name: str = "gpt-4", model_key: str = None, web_search: bool = False, **kwargs):
        self.model_name = model_name
        self.role_names = ["system", "user", "assistant"]

        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"LLM model {model_name} not supported")

        if self.model_name in ["o1", "o1-preview", "o3-mini"]:
            kwargs["temperature"] = 1
            role_names = {Llm.Role.SYSTEM: "user"}
        else:
            role_names = {}

        logging.info(f"Using {self.model_name}")

        self.model_key = model_key if model_key else os.environ.get("OPENAI_API_KEY", None)
        if not self.model_key:
            raise RuntimeError(f"OpenAI API key not provided")

        if web_search:
            if self.model_name not in self.WEB_SEARCH_SUPPORTED:
                raise NotImplementedError(f"Web search is not supported by {self.model_name}")
            self.llm = ChatOpenAI(
                model_name=self.model_name, openai_api_key=self.model_key,
                use_responses_api=True, **kwargs
            ).bind_tools([{"type": "web_search"}])
        else:
            self.llm = ChatOpenAI(model_name=self.model_name, openai_api_key=self.model_key, **kwargs)

        super().__init__(llm=self.llm, role_names=role_names)

    def clean_up_response(self, response: Any) -> dict:
        if isinstance(response, AIMessage):
            return {
                "content": response.text(),
                "metadata": response
            }

        else:
            raise TypeError(f"Unsupported return type for GptLlm.invoke() (was {type(response)})")

    def get_num_tokens(self, text: str) -> int:
        return ChatOpenAI(temperature=0).get_num_tokens(text)

    def get_max_tokens(self) -> int:
        return self.__MODEL_TOKEN_LIMITS.get(self.model_name, 100_000)

    @classmethod
    def get_supported_models(cls) -> List[str]:
        return list(cls.MODEL_ALIASES.keys()) + cls.SUPPORTED_MODELS

    def as_runnable(self) -> Runnable:
        return self.llm

    def as_language_model(self) -> BaseLanguageModel:
        return self.llm
