from typing import Union, Any, Optional

from langchain.agents import AgentExecutor  # This is needed!  Or pydantic will complain
from langchain_core.language_models import BaseLanguageModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

__xxx = AgentExecutor  # So the import above won't get automatically removed


class RunnableToLLMAdapter(BaseLanguageModel):
    runnable: Runnable = Field(...)

    def __init__(self, runnable: Runnable, **kwargs):
        super().__init__(runnable=runnable, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "runnable_adapter"

    def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        prompt_texts = [prompt.to_string() for prompt in prompts]
        generations = [{"text": self.runnable.invoke(prompt)} for prompt in prompt_texts]
        return LLMResult(generations=generations)

    async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        import asyncio
        prompt_texts = [prompt.to_string() for prompt in prompts]
        generations = await asyncio.gather(*[asyncio.to_thread(self.runnable.invoke, prompt) for prompt in prompt_texts])
        return LLMResult(generations=[{"text": gen} for gen in generations])

    def predict(self, text, *, stop=None, **kwargs):
        return self.runnable.invoke(text, **kwargs)

    async def apredict(self, text, *, stop=None, **kwargs):
        import asyncio
        return await asyncio.to_thread(self.runnable.invoke, text)

    def predict_messages(self, messages, *, stop=None, **kwargs):
        input_text = "\n".join([message.content for message in messages])
        response = self.runnable.invoke(input_text)
        return BaseMessage(content=response)

    async def apredict_messages(self, messages, *, stop=None, **kwargs):
        import asyncio
        input_text = "\n".join([message.content for message in messages])
        response = await asyncio.to_thread(self.runnable.invoke, input_text)
        return BaseMessage(content=response)

    def invoke(self, query, config=None, **kwargs):
        return self.runnable.invoke(query, **kwargs)

    def with_structured_output(self, schema: Union[dict, type], **kwargs: Any) -> \
            Runnable[LanguageModelInput, Union[dict, BaseModel]]:
        """
        Enforce structured output by wrapping the LLM's output in a schema.
        """
        class StructuredOutputRunnable(Runnable):
            def __init__(self, runnable: Runnable, schema: Union[dict, type]):
                self.runnable = runnable
                self.schema = schema

            def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
                # Get the raw output from the LLM
                raw_output = self.runnable.invoke(input, config=config, **kwargs)

                # If the schema is a Pydantic model, validate and parse the output
                if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
                    return self.schema.parse_raw(raw_output)

                # If the schema is a dictionary, assume JSON and validate manually
                elif isinstance(self.schema, dict):
                    import json
                    parsed_output = json.loads(raw_output)
                    for key, value_type in self.schema.items():
                        if key not in parsed_output or not isinstance(parsed_output[key], value_type):
                            raise ValueError(f"Invalid output: {parsed_output}")
                    return parsed_output

                else:
                    raise ValueError("Schema must be a Pydantic model or a dictionary.")

        # Return a new Runnable that enforces the schema
        return StructuredOutputRunnable(self.runnable, schema)
