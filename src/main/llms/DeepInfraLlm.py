import logging
import os
from typing import Any, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from llms import websearch
from llms.Llm import Llm
from llms.RunnableToLLMAdapter import RunnableToLLMAdapter
from llms.websearch.WebSearch import WebSearch


class DeepInfraLlm(Llm):
    __DEFAULT_TOKEN_LIMIT = 128_000

    __MODELS = {
        "Sao10K/L3.3-70B-Euryale-v2.3": {"aliases": ["euryale"], "token_limit": 131_072},
        "meta-llama/Llama-3.3-70B-Instruct": {"aliases": ["llama-3"], "web_search": True},
        "microsoft/WizardLM-2-8x22B": {"aliases": ["wizard-2"], "token_limit": 65_536},
        "deepseek-ai/DeepSeek-V3.1": {"aliases": ["deepseek-v3.1"], "token_limit": 163_840, "web_search": True},
        # "meta-llama/Llama-4-Maverick-17B-128E-Instruct-Turbo": {"aliases": ["llama-4"], "web_search": True}
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": {"aliases": ["llama-4"], "web_search": True}
    }

    SUPPORTED_MODELS = list(__MODELS.keys())

    MODEL_ALIASES = Llm._alias2model(__MODELS)

    __MODEL_TOKEN_LIMITS = Llm._model_token_limit(__MODELS, __DEFAULT_TOKEN_LIMIT)

    __API_URL = "https://api.deepinfra.com/v1/openai"

    __DEFAULT_WEB_SEARCH = "ddgs"

    def __init__(
            self,
            model_name: str = "llama-3",
            model_key: str = None,
            web_search: bool | str | WebSearch = False,
            **kwargs
    ):
        self.model_name = model_name
        self.role_names = ["system", "user", "assistant"]

        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"LLM model {model_name} not supported")

        logging.info(f"Using {self.model_name}")

        self.model_key = model_key if model_key else os.environ.get("DEEPINFRA_API_KEY", None)
        if not self.model_key:
            raise RuntimeError(f"DeepInfra API key is not provided")

        base_llm = ChatOpenAI(
            base_url=self.__API_URL,
            api_key=self.model_key,
            model=self.model_name,
            **kwargs
        )

        # Optional: bind web search tool
        support_web_search = self.__MODELS[self.model_name].get("web_search", False)
        logging.info(f"**** web_search={web_search} support_web_search={support_web_search}")
        if web_search and support_web_search:
            logging.info("Using web search")
            if web_search is True or web_search == "auto":
                web_search = self.__DEFAULT_WEB_SEARCH

            if isinstance(web_search, str):
                self.web_search = websearch.of(web_search)
            elif isinstance(web_search, WebSearch):
                self.web_search = web_search
            else:
                raise TypeError("web_search must be a bool or WebSearch instance")

            tool = self.web_search.as_tool()
            self.llm = self._make_agent_runnable(base_llm, [tool])
        else:
            self.llm = base_llm

        super().__init__(llm=self.llm)

    def clean_up_response(self, response: Any) -> Llm.Response:
        if isinstance(response, AIMessage):
            return Llm.Response(
                text=response.text(),
                citations=response.response_metadata.get("sources"),
                raw=response
            )

        else:
            raise TypeError(f"Unsupported return type for {self.__class__}.invoke() (was {type(response).__name__})")

    def get_max_tokens(self) -> int:
        return self.__MODEL_TOKEN_LIMITS.get(self.model_name, 1000)

    @classmethod
    def get_supported_models(cls) -> List[str]:
        return list(cls.MODEL_ALIASES.keys()) + cls.SUPPORTED_MODELS

    def as_runnable(self) -> Runnable:
        return self.llm

    def as_language_model(self) -> BaseLanguageModel:
        return RunnableToLLMAdapter(self.as_runnable())
