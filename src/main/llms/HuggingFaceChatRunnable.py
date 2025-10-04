from typing import Optional

from huggingface_hub import InferenceClient
from langchain.schema.runnable import Runnable
from langchain_core.messages.utils import convert_to_messages, convert_to_openai_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input


class HuggingFaceChatRunnable(Runnable):
    __MODEL_PROVIDER = {
        # If not defined here, default to "hf-inference"
        "deepseek-ai/DeepSeek-R1": "together",
    }
    # ["hf-inference", 'sambanova', 'together']

    def __init__(self, model_name: str, api_key: str, **kwargs):
        self.model_name = model_name

        self.client = InferenceClient(
            model=self.model_name,
            provider=self.__MODEL_PROVIDER.get(self.model_name, "hf-inference"),
            api_key=api_key,
            **kwargs
        )

    def invoke(self, text: Input, config: Optional[RunnableConfig] = None, **kwargs) -> str:
        """
        Invoke the Hugging Face chat model with the given query.

        Args:
            text (dict): A dictionary containing the "content" field with the user query.
            config (Optional[RunnableConfig]): Optional configuration for the Runnable.

        Returns:
            str: The generated response from the model.
        """
        # base_messages = convert_to_messages(text)

        task = config.get("metadata", {}).get("task", "chat")

        if task == "chat":
            messages = convert_to_openai_messages(text)
            # messages = [{"role": "user", "content": m.content} for m in base_messages]
            completion = self.client.chat_completion(messages=messages)
            return completion.choices[0].message["content"]

        elif task == "generation":
            base_messages = convert_to_messages(text)
            prompt = "\n\n".join([m.content for m in base_messages])
            completion = self.client.text_generation(prompt=prompt)
            return completion

        # Call the Hugging Face chat completion API
        # completion = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=messages,
        # )
        # completion = self.client.text_generation(
        #     prompt=messages,
        # )
        # completion = self.client.chat_completion(messages=messages)
        # completion = self.client.text_generation(prompt=messages)
        #
        # # Extract and return the response content
        # # return completion.choices[0].message["content"]
        # return completion

