# WebSearch.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.tools import tool


class WebSearch(ABC):
    """
    Abstract base class for web search integrations.
    Subclasses should implement `search()` to return structured results.
    """

    def __init__(self, name: str = "web_search", description: Optional[str] = None):
        self.name = name
        self.description = description or "Perform a web search and return summarized results."

    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the search query and return a list of results.
        Each result should contain 'title', 'url', and 'snippet'.
        """
        pass

    def as_tool(self):
        """
        Wraps this search as a LangChain tool so it can be bound via `.bind_tools()`.
        """

        @tool(self.name, description=self.description)
        def _search_tool(query: str) -> List[Dict[str, Any]]:
            return self.search(query)

        return _search_tool
