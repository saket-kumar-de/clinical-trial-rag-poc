"""
Pulls trial records from the ClinicalTrials.gov API v2 (no key required).
Docs: https://clinicaltrials.gov/data-api/api
"""
import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_trials(condition: str, max_results: int = 30) -> list[dict]:
    """
    Fetch up to `max_results` trial records for a given condition.

    Follows nextPageToken until either max_results is reached or the
    API reports no more pages. Returns raw study JSON records
    (each with a protocolSection); parsing happens in extraction/.
    """
    studies: list[dict] = []
    page_token: str | None = None

    while len(studies) < max_results:
        params = {
            "query.cond": condition,
            "pageSize": min(max_results - len(studies), 1000),
            "format": "json",
        }
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        studies.extend(data.get("studies", []))
        page_token = data.get("nextPageToken")

        if not page_token:
            break

    return studies[:max_results]