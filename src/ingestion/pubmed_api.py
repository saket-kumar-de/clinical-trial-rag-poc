"""
Pulls abstracts from PubMed via NCBI E-utilities (esearch + efetch).
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
Free, but set PUBMED_EMAIL (and optionally PUBMED_API_KEY) in .env
to raise your rate limit and follow NCBI's usage policy.
"""
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query: str, max_results: int = 10) -> list[str]:
    """
    Return a list of PubMed IDs matching `query`.
    TODO: call esearch, parse the <IdList> from the XML response.
    """
    raise NotImplementedError


def fetch_abstracts(pubmed_ids: list[str]) -> list[dict]:
    """
    Fetch abstract text + metadata for a list of PubMed IDs.
    TODO: call efetch with rettype=abstract, parse XML into
    {pubmed_id, title, abstract} dicts.
    """
    raise NotImplementedError
