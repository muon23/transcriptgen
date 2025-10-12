from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from langchain.tools import tool


class KvQuery(ABC):
    """
    Abstract base class for SQL query a database.
    Subclasses should implement `get()`, `get_range()` and `get_collection_descriptions()`.
    """

    def __init__(self, name: str = "kev_value_lookup", description: Optional[str] = None):
        self.name = name
        self.description = description or "Looks up a NoSQL database"

    @dataclass
    class FieldDescription:
        name: str
        description: str
        type: str

    @dataclass
    class CollectionDescription:
        name: str
        description: str
        key: str
        key_description: str
        value: List[KvQuery.FieldDescription]

    @abstractmethod
    def get_collection_descriptions(self) -> List[CollectionDescription]:
        pass

    @abstractmethod
    def get(self, key: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_range(self, key_from: str, key_to: str) -> List[Dict[str, Any]]:
        pass

    def get_tool(self):
        """
        Wraps the get() function as a LangChain tool so it can be bound via `.bind_tools()`.
        """

        @tool(self.name, description=self.description + " with a key")
        def _get_tool(key: str) -> List[Dict[str, Any]]:
            return self.get(key)

        return _get_tool

    def get_range_tool(self):
        """
        Wraps the get_range() function as a LangChain tool so it can be bound via `.bind_tools()`.
        """

        @tool(self.name + "_range", description=self.description + " with a range of keys")
        def _get_range_tool(key_from: str, key_to: str) -> List[Dict[str, Any]]:
            return self.get_range(key_from, key_to)

        return _get_range_tool
