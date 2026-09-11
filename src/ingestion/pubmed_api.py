"""
Pulls abstracts from PubMed via NCBI E-utilities (esearch + efetch).
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
Free, but set PUBMED_EMAIL (and optionally PUBMED_API_KEY) in .env
to raise your rate limit and follow NCBI's usage policy.
"""
import xml.etree.ElementTree as ET

import requests

from src.config import settings

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_ESEARCH_MAX_RETMAX = 500


def _identity_params() -> dict:
    """NCBI-recommended identification params — optional, but raises rate limits."""
    params = {}
    if settings.pubmed_email:
        params["email"] = settings.pubmed_email
    if settings.pubmed_api_key:
        params["api_key"] = settings.pubmed_api_key
    return params


def search_pubmed(query: str, max_results: int = 10) -> list[str]:
    """Return a list of PubMed IDs matching `query`."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": min(max_results, _ESEARCH_MAX_RETMAX),
        "retmode": "json",
        **_identity_params(),
    }
    response = requests.get(ESEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pubmed_ids: list[str]) -> list[dict]:
    """
    Fetch abstract text + metadata for a list of PubMed IDs.
    Returns a list of {pubmed_id, title, abstract} dicts.
    """
    if not pubmed_ids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pubmed_ids),
        "rettype": "abstract",
        "retmode": "xml",
        **_identity_params(),
    }
    response = requests.get(EFETCH_URL, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    articles = []
    for article_elem in root.findall(".//PubmedArticle"):
        articles.append(
            {
                "pubmed_id": article_elem.findtext(".//PMID", default=""),
                "title": article_elem.findtext(".//ArticleTitle", default=""),
                "abstract": _join_abstract_sections(article_elem),
            }
        )
    return articles


def _join_abstract_sections(article_elem: ET.Element) -> str:
    """
    Join every AbstractText element into one string.

    Structured abstracts (Background / Methods / Results / Conclusions)
    have multiple <AbstractText Label="..."> elements rather than one
    flat block — grabbing only the first would silently drop most of
    the abstract. Each section is prefixed with its label when present.
    """
    sections = []
    for elem in article_elem.findall(".//Abstract/AbstractText"):
        label = elem.get("Label")
        text = "".join(elem.itertext()).strip()
        if not text:
            continue
        sections.append(f"{label}: {text}" if label else text)
    return "\n".join(sections)