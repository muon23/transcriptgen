from llms.websearch.WebSearch import WebSearch
from llms.websearch.DuckDuckGoWebSearch import DuckDuckGoWebSearch


def of(provider: str = "auto") -> WebSearch:
    if provider.lower() in {"ddgs", "duckduckgo"}:
        return DuckDuckGoWebSearch()

    raise NotImplementedError(f"Web search provider {provider} not supported.")
