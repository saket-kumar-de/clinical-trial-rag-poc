"""
Pulls trial records from the ClinicalTrials.gov API v2 (no key required).
Docs: https://clinicaltrials.gov/data-api/api
"""
import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_trials(condition: str, max_results: int = 30) -> list[dict]:
    """
    Fetch up to `max_results` trial records for a given condition.

    TODO:
      - Build the query params (query.cond, pageSize)
      - Handle pagination (nextPageToken) if max_results > page size
      - Return the raw JSON records; parsing happens in extraction/
    """
    raise NotImplementedError
