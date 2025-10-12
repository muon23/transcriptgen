import logging
from typing import List, Dict, Any
from ddgs import DDGS
from llms.websearch.WebSearch import WebSearch


class DuckDuckGoWebSearch(WebSearch):
    """
    DDG search with optional expansion: fetch & extract page content + top passages.
    """

    def __init__(self):
        super().__init__(
            name="duckduckgo_search",
            description="Search the web using DuckDuckGo and optionally expand results with extracted page content."
        )

    # noinspection PyMethodMayBeStatic
    def search(
            self,
            query: str,
            max_results: int = 5,
            expand: bool = True,  # << turn on extraction
            top_passages: int = 5,
            max_chars_per_doc: int = 12000,  # cap raw text
    ) -> List[Dict[str, Any]]:
        logging.info(f"DDG search: {query!r}")
        out: List[Dict[str, Any]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title")
                url = r.get("href")
                snippet = r.get("body")
                item: Dict[str, Any] = {"title": title, "url": url, "snippet": snippet}

                if expand and url:
                    try:
                        if url.lower().endswith(".pdf"):
                            extracted = WebSearch._extract_pdf(url)
                        else:
                            extracted = WebSearch._extract_html(url)
                        full_text = (extracted.get("text") or "")[:max_chars_per_doc]
                        item["content"] = full_text
                        if extracted.get("title") and not title:
                            item["title"] = extracted["title"]
                        if extracted.get("published"):
                            item["published"] = extracted["published"]

                        # lightweight passage selection biased toward the query
                        if full_text:
                            item["passages"] = WebSearch._top_passages(full_text, query, top_k=top_passages)
                    except Exception as e:
                        logging.warning(f"Failed to expand {url}: {e}")

                out.append(item)
        return out
