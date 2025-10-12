from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from langchain.tools import tool


class SqlQuery(ABC):
    """
    Abstract base class for SQL query a database.
    Subclasses should implement `query()` and get_table_description().
    """
    def __init__(self, name: str = "sql_query", description: Optional[str] = None):
        self.name = name
        self.description = description or "Perform a SQL query to a database."

    @dataclass
    class ColumnDescription:
        name: str
        description: str
        type: str

    @dataclass
    class TableDescription:
        name: str
        description: str
        columns: List[SqlQuery.ColumnDescription]

    @abstractmethod
    def get_table_descriptions(self) -> List[TableDescription]:
        pass

    @abstractmethod
    def query(self, sql: str) -> List[Dict[str, Any]]:
        pass

    def as_tool(self):
        """
        Wraps this query as a LangChain tool so it can be bound via `.bind_tools()`.
        """
        @tool(self.name, description=self.description)
        def _query_tool(sql: str) -> List[Dict[str, Any]]:
            return self.query(sql)

        return _query_tool

