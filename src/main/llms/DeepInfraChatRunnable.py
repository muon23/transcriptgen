from typing import Optional

from huggingface_hub import InferenceClient
from langchain.schema.runnable import Runnable
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input


class DeepInfraChatRunnable(Runnable):

    def __init__(self, model_name: str, api_url: str, api_key: str, **kwargs):
        self.model_name = model_name

        self.client = InferenceClient(
            base_url=api_url,
            api_key=api_key,
            **kwargs
        )

    def invoke(self, text: Input, config: Optional[RunnableConfig] = None, **kwargs) -> str:
        # Convert the prompt into a chat completion format
        messages = convert_to_openai_messages(text)
        kwargs["model"] = self.model_name
        completion = self.client.chat_completion(messages=messages, **kwargs)
        return completion.choices[0].message["content"]
