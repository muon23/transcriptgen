import logging
import os
from typing import Any, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable

from llms.Llm import Llm
from llms.DeepInfraChatRunnable import DeepInfraChatRunnable
from llms.RunnableToLLMAdapter import RunnableToLLMAdapter


class DeepInfraLlm(Llm):

    __DEFAULT_TOKEN_LIMIT = 128_000

    __MODELS = {
        "Sao10K/L3.3-70B-Euryale-v2.3": {"aliases": ["euryale"], "token_limit": 131_072},
        "meta-llama/Llama-3.3-70B-Instruct": {"aliases": ["llama-3"]},
        "microsoft/WizardLM-2-8x22B": {"aliases": ["wizard-2"], "token_limit": 65_536},
        "deepseek-ai/DeepSeek-V3-0324": {"aliases": ["deepseek-v3"], "token_limit": 163_840},
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {"aliases": ["llama-4"]}
    }

    SUPPORTED_MODELS = list(__MODELS.keys())

    MODEL_ALIASES = Llm._alias2model(__MODELS)

    __MODEL_TOKEN_LIMITS = Llm._model_token_limit(__MODELS, __DEFAULT_TOKEN_LIMIT)

    __API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

    def __init__(self, model_name: str = "llama-2", model_key: str = None, **kwargs):
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

        self.llm = DeepInfraChatRunnable(
            model_name=self.model_name,
            api_key=self.model_key,
            api_url=self.__API_URL,
        )

        super().__init__(llm=self.llm)

    def clean_up_response(self, response: Any) -> dict:
        if isinstance(response, str):
            return {
                "content": response,
                "metadata": {}
            }

        else:
            raise TypeError(f"Unsupported return type for GptLlm.invoke() (was {type(response)})")

    def get_max_tokens(self) -> int:
        return self.__MODEL_TOKEN_LIMITS.get(self.model_name, 1000)

    @classmethod
    def get_supported_models(cls) -> List[str]:
        return list(cls.MODEL_ALIASES.keys()) + cls.SUPPORTED_MODELS

    def as_runnable(self) -> Runnable:
        return self.llm

    def as_language_model(self) -> BaseLanguageModel:
        return RunnableToLLMAdapter(self.as_runnable())
