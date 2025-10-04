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


class DeepSeekLlm(Llm):

    SUPPORTED_MODELS = [
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ]

    MODEL_ALIASES = {
        "deepseek": "deepseek-ai/DeepSeek-R1",
        "deepseek-gwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    }

    __MODEL_TOKEN_LIMITS = {
        "deepseek-ai/DeepSeek-R1": 8000,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 32_768,
    }

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

    def clean_up_response(self, response: Any) -> dict:
        if isinstance(response, AIMessage):
            content = response.content
            metadata = {"response": response}

        elif isinstance(response, str):
            content = response
            metadata = dict()

        else:
            raise TypeError(f"Unsupported return type for HfLlm.invoke() (was {type(response)})")

        thought, content = self.__separate_think_tag(content)
        if thought:
            metadata["thought"] = thought

        return {
            "content": content,
            "metadata": metadata,
        }

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
