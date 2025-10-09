import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence, Any, List, Dict

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, Runnable, RunnableConfig, RunnableLambda
from transformers import AutoTokenizer


class Llm(ABC):
    """
    Abstract class for accessing LLM models.

    This class provides a standardized interface for interacting with different LLM
    providers (like OpenAI or Gemini) using the LangChain framework, handling
    prompt preprocessing, token counting, and configuration management.
    Concrete subclasses must implement provider-specific logic.
    """

    class Role(Enum):
        """Enumeration for standard conversational roles."""
        SYSTEM = 0
        HUMAN = 1
        AI = 2

    @dataclass
    class Response:
        """Standardized structure for LLM responses across all provider subclasses."""
        text: str = None
        image_url: str = None
        tool_calls: list[dict] = field(default_factory=list)
        citations: list[dict] = field(default_factory=list)
        thought: str = None
        metadata: Any = None
        raw: Any = None

    def __init__(self, llm: Runnable, role_names: dict = None):
        """
        Initializes the LLM wrapper.

        Args:
            llm: The underlying LangChain Runnable object for the specific LLM.
            role_names: Optional dictionary to map internal Role enums to custom
                        string names (e.g., HUMAN -> "user", AI -> "assistant").
        """
        self.llm = llm
        # Default role names used for converting history into LangChain format
        self.role_names = {
            self.Role.SYSTEM: "system",
            self.Role.HUMAN: "user",
            self.Role.AI: "assistant",
        }
        if role_names:
            # Overwrite defaults if custom roles are provided
            self.role_names.update(role_names)

        # Use GPT-2 tokenizer as a robust, universal approximation for token counting
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def invoke(self, prompt: Sequence[tuple[Role | str, str] | str] | str, **kwargs) -> Response:
        """
        Processes the prompt, executes the LLM chain, and returns the cleaned response.

        Args:
            prompt: The input prompt. Can be a simple string or a sequence of
                    (Role/role_name, content) tuples representing chat history.
            **kwargs: Additional arguments passed to the chain, including
                      'arguments' (for prompt template values) and 'task' (for metadata).

        Returns:
            A dictionary containing the cleaned response from the LLM.
        """
        # Format the prompt into a LangChain ChatPromptTemplate object
        prompt = self.preprocess_prompt(prompt)

        # Prompt template parameters for filling holes in the prompt
        arguments = kwargs.get("arguments", {})

        # Create a sequential chain: Prompt -> LLM -> Response Cleanup
        chain = RunnableSequence(prompt | self.llm | self.clean_up_response)

        # Task, e.g., chat or completion.
        # Some LLM models need to distinguish chat or completion.
        # This is a way for the derived class to pass in its purpose.
        task = kwargs.get("task", self.get_default_task())
        config = RunnableConfig(metadata={"task": task})

        # Execute the chain
        response: Llm.Response = chain.invoke(input=arguments, config=config, **kwargs)

        return response

    def preprocess_prompt(self, prompt: Sequence[tuple[Role | str, str] | str] | str) -> ChatPromptTemplate:
        """
        Converts various input prompt formats into a standardized ChatPromptTemplate.
        """
        # If the prompt is a simple string, wrap it as a single message
        if isinstance(prompt, str):
            # This handles single instruction prompts
            human_role_name = self.role_names[self.Role.HUMAN]
            return ChatPromptTemplate(messages=[(human_role_name, prompt)])
        else:
            # Reformat the sequence of (Role, content) tuples into LangChain messages
            messages = []
            for msg in prompt:
                role_name = self.role_names[msg[0]] if isinstance(msg[0], self.Role) else msg[0]
                messages.append((role_name, msg[1]))
            return ChatPromptTemplate(messages=messages)

    @abstractmethod
    def clean_up_response(self, response: Any) -> Response:
        """
        Abstract method to clean up and standardize the LLM's raw response.

        Concrete implementation must extract text, cost metadata, etc., and return a dict.
        """
        pass

    def get_num_tokens(self, text: str) -> int:
        """
        Estimates the number of tokens in a given text using the GPT-2 tokenizer.
        """
        return len(self.tokenizer.encode(text))

    @abstractmethod
    def get_max_tokens(self) -> int:
        """
        Abstract method to return the maximum context window size for the underlying model.
        """
        pass

    def get_default_task(self) -> str:
        """Returns the default task type for metadata."""
        return "chat"

    @classmethod
    @abstractmethod
    def get_supported_models(cls) -> List[str]:
        """Abstract method to return a list of model names supported by the subclass."""
        pass

    #
    # LangChain adapters (Allowing the Llm object to be used seamlessly in LangChain chains)
    #
    @abstractmethod
    def as_runnable(self) -> Runnable:
        """Returns the LLM instance as a LangChain Runnable."""
        pass

    @abstractmethod
    def as_language_model(self) -> BaseLanguageModel:
        """Returns the LLM instance as a LangChain BaseLanguageModel."""
        pass

    #
    # Helper functions
    #
    @classmethod
    def _alias2model(cls, models: Dict[str, dict]) -> Dict[str, str]:
        """Helper to create a mapping from model aliases to canonical model names."""
        a2m = dict()
        for model, properties in models.items():
            aliases = properties.get("aliases", [])
            for alias in aliases:
                a2m[alias] = model
        return a2m

    @classmethod
    def _model_token_limit(cls, models: Dict[str, dict], default: int) -> Dict[str, int]:
        """Helper to extract token limits from model configuration data."""
        limits = dict()
        for model, properties in models.items():
            limits[model] = properties.get("token_limit", default)
        return limits

    @staticmethod
    def _safe_json_loads(x):
        try:
            return json.loads(x) if isinstance(x, str) else x
        except Exception:
            return None

    @staticmethod
    def _extract_sources_from_observation(obs) -> List[str]:
        urls = []
        data = Llm._safe_json_loads(obs)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    url = item.get("url")
                    if url:
                        urls.append(url)
        elif isinstance(data, dict):
            url = data.get("url")
            if url:
                urls.append(url)
        return urls

    @classmethod
    def _make_agent_runnable(cls, llm, tools, system_prompt: str = "You are helpful.") -> Runnable:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)

        # Return intermediate steps so we can summarize tools/sources
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            return_intermediate_steps=True,
        )

        def _invoke(x):
            result: Dict[str, Any] = executor.invoke({"input": x, "chat_history": []})
            output_text = result.get("output", "")

            # Put in more information for troubleshooting
            steps = result.get("intermediate_steps", [])  # list of (AgentAction, observation)
            tools_used: List[str] = []
            sources: List[str] = []

            for action, observation in steps:
                # action.tool and action.tool_input exist for tool-calling agents
                tool_name = getattr(action, "tool", None)
                if tool_name:
                    tools_used.append(tool_name)

                # try to extract URLs from your WebSearch tool’s observation (list[dict])
                sources.extend(Llm._extract_sources_from_observation(observation))

            # dedupe but keep order
            def _dedupe(seq): return list(dict.fromkeys(seq))
            tools_used = _dedupe(tools_used)
            sources = _dedupe(sources)

            # Optional compact trace (be careful not to bloat tokens)
            compact_trace = []
            for action, observation in steps:
                tool_name = getattr(action, "tool", None)
                tool_args = getattr(action, "tool_input", None)
                compact_trace.append({
                    "tool": tool_name,
                    "args_preview": str(tool_args)[:200] if tool_args is not None else None,
                    "obs_preview": (observation[:200] if isinstance(observation, str) else None)
                })

            meta = {
                "agent_executor": "tool_calling",
                "model": getattr(llm, "model_name", getattr(llm, "model", None)),
                "num_steps": len(steps),
                "tools_used": tools_used,
                "sources": sources,
            }

            return AIMessage(
                id=str(uuid.uuid4()),
                content=output_text,
                response_metadata=meta,
                # keep the big stuff out of `additional_kwargs` if you worry about token bloat
                additional_kwargs={"trace": compact_trace} if compact_trace else {},
            )

        return RunnableLambda(_invoke)


