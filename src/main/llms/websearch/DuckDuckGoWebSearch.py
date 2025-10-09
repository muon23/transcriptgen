# llms/websearch/DuckDuckGoWebSearch.py
import io
import json
import logging
import re
from typing import List, Dict, Any

import httpx  # pip install httpx trafilatura pypdf duckduckgo-search
import trafilatura
from ddgs import DDGS
from pypdf import PdfReader

from llms.websearch.WebSearch import WebSearch


def _split_sentences(text: str) -> List[str]:
    # naive sentence splitter that works "ok" without extra deps
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _top_passages(text: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    # ultra-light query scoring: counts of query tokens in each sentence,
    # then returns 2-sentence windows around the best matches.
    q_terms = [t.lower() for t in re.findall(r"\w+", query)]
    if not q_terms:
        q_terms = [query.lower()]
    sents = _split_sentences(text)
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
                            extracted = _extract_pdf(url)
                        else:
                            extracted = _extract_html(url)
                        full_text = (extracted.get("text") or "")[:max_chars_per_doc]
                        item["content"] = full_text
                        if extracted.get("title") and not title:
                            item["title"] = extracted["title"]
                        if extracted.get("published"):
                            item["published"] = extracted["published"]

                        # lightweight passage selection biased toward the query
                        if full_text:
                            item["passages"] = _top_passages(full_text, query, top_k=top_passages)
                    except Exception as e:
                        logging.warning(f"Failed to expand {url}: {e}")

                out.append(item)
        return out
