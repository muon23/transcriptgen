import logging
import os
from typing import Any, Sequence, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from llms.Llm import Llm
from llms.HuggingFaceChatRunnable import HuggingFaceChatRunnable
from llms.RunnableToLLMAdapter import RunnableToLLMAdapter


class LlamaLlm(Llm):

    SUPPORTED_MODELS = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-3.2-1B",
    ]

    MODEL_ALIASES = {
        "llama-2": "meta-llama/Llama-2-7b-chat-hf",
        "llama-3": "meta-llama/Llama-3.2-1B"
    }

    __MODEL_TOKEN_LIMITS = {
        "meta-llama/Llama-2-7b-chat-hf": 4096,
        "meta-llama/Llama-3.1-8B": 8000,
    }

    __CHAT_NOT_SUPPORTED = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-3.2-1B",
    ]

    __URL_PREFIX = "https://huggingface.co/"

    def __init__(self, model_name: str = "llama-2", model_key: str = None, **kwargs):
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
    def __convert_to_llama_prompt(cls, messages: Sequence[tuple[Llm.Role, str]]) -> str:
        prompt = ""
        for role, content in messages:
            if role == "system":
                prompt += f"<<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                prompt += f"[INST] {content} [/INST]\n"
            else:
                prompt += f"{content}\n"
        return prompt

    def preprocess_prompt(self, prompt: Sequence[tuple[Llm.Role, str] | str] | str) -> ChatPromptTemplate:
        # Reformat the prompt
        if not isinstance(prompt, str):
            prompt = self.__convert_to_llama_prompt(prompt)

        return ChatPromptTemplate(messages=[prompt])

    def clean_up_response(self, response: Any) -> dict:
        if isinstance(response, AIMessage):
            content = response.content
            metadata = {"response": response}

        elif isinstance(response, str):
            content = response
            metadata = dict()

        else:
            raise TypeError(f"Unsupported return type for LlamaLlm.invoke() (was {type(response)})")

        content = content.replace("[/INST]", "")

        return {
            "content": content,
            "metadata": metadata,
        }

    def get_max_tokens(self) -> int:
        limit = self.__MODEL_TOKEN_LIMITS.get(self.model_name, 4000)
        return limit

    def get_default_task(self) -> str:
        return "generation" if self.model_name in self.__CHAT_NOT_SUPPORTED else "chat"

    @classmethod
    def get_supported_models(cls) -> List[str]:
        return list(cls.MODEL_ALIASES.keys()) + cls.SUPPORTED_MODELS

    def as_runnable(self) -> Runnable:
        return self.llm

    def as_language_model(self) -> BaseLanguageModel:
        return RunnableToLLMAdapter(self.as_runnable())
