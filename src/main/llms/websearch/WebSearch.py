# WebSearch.py
import io
import json
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from typing import Optional

import httpx  # pip install httpx trafilatura pypdf duckduckgo-search
import trafilatura
from langchain.tools import tool
from pypdf import PdfReader


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

    #
    # Helper functions
    #
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        # naive sentence splitter that works "ok" without extra deps
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _top_passages(text: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # ultra-light query scoring: counts of query tokens in each sentence,
        # then returns 2-sentence windows around the best matches.
        q_terms = [t.lower() for t in re.findall(r"\w+", query)]
        if not q_terms:
            q_terms = [query.lower()]
        sents = WebSearch._split_sentences(text)
        if not sents:
            return []

        scores = []
        for i, s in enumerate(sents):
            tok = [t.lower() for t in re.findall(r"\w+", s)]
            score = sum(tok.count(qt) for qt in q_terms)
            scores.append((score, i))

        scores.sort(reverse=True, key=lambda x: x[0])
        keep = []
        used_idx = set()
        for sc, i in scores:
            if sc <= 0:
                break
            # take a small window around the sentence
            window = sents[max(0, i - 1): min(len(sents), i + 2)]
            txt = " ".join(window)
            # prevent near-duplicates by index overlap
            if any(j in used_idx for j in range(max(0, i - 1), min(len(sents), i + 2))):
                continue
            used_idx.update(range(max(0, i - 1), min(len(sents), i + 2)))
            keep.append({"text": txt, "score": sc})
            if len(keep) >= top_k:
                break
        return keep

    @staticmethod
    def _extract_html(url: str, timeout: float = 15.0) -> Dict[str, Any]:
        """
        Fetch and extract main text from an HTML page.
        Uses httpx for network timeouts, trafilatura for readability extraction.
        Returns {'text': str, 'title': str|None, 'published': str|None}
        """
        # Make a single request with sane timeouts + redirects
        # (tweak connect/read if you like: httpx.Timeout(connect=5, read=timeout))
        headers = {"User-Agent": "Mozilla/5.0 (compatible; transcriptgen/1.0)"}
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            r = client.get(url)
            r.raise_for_status()
            html = r.text  # decoded string

        # Try JSON output first to capture metadata, then fall back to plain text
        data_json = trafilatura.extract(
            html,
            include_images=False,
            include_links=False,
            include_comments=False,
            with_metadata=True,
            output_format="json",
        )
        if not data_json:
            # fallback: plain text only
            plain = trafilatura.extract(html) or ""
            return {"text": plain, "title": None, "published": None}

        try:
            data = json.loads(data_json)
        except Exception:
            return {"text": "", "title": None, "published": None}

        return {
            "text": data.get("text") or "",
            "title": data.get("title"),
            "published": data.get("date"),
        }

    @staticmethod
    def _extract_pdf(url: str, timeout: float = 20.0, max_pages: int = 6) -> Dict[str, Any]:
        """
        Fetch and extract text from the first few pages of a PDF.
        """
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()
            bio = io.BytesIO(r.content)
            reader = PdfReader(bio)
            pages = []
            for i, page in enumerate(reader.pages[:max_pages]):
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    continue
            return {"text": "\n".join(pages)}

