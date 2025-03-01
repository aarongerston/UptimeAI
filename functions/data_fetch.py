"""
data_fetch.py

Contains functionality to query the IODA API for metadata
and raw signal data based on user parameters.
"""

import requests
from config import API_URL, logger


def get_available_entities() -> dict:
    """
    Fetch metadata on available entities from the IODA API.
    Returns JSON data containing entity codes, names, and types.
    """
    endpoint = "entities/query"
    url = API_URL + endpoint
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", [])
        logger.info("Successfully fetched entity metadata from IODA API.")
        return data
    except requests.RequestException as e:
        logger.error(f"Error fetching entities from IODA API: {e}")
        return {}


def fetch_signal_data(
    entity_type: str, entity_code: str, from_date: str, to_date: str, datasource: str
) -> list:
    """
    Fetch raw signal data for a given entity type and code from the IODA API.

    :param entity_type: 'continent', 'country', or 'region'
    :param entity_code: Comma-separated region codes or country codes
    :param from_date: POSIX timestamp (as string)
    :param to_date: POSIX timestamp (as string)
    :param datasource: e.g., "bgp", "merit-nt", "ping-slash24"
    :return: List of dicts containing time-series data.
    """
    endpoint = f"signals/raw/{entity_type}/{entity_code}"
    params = {"from": from_date, "until": to_date, "datasource": datasource}
    url = API_URL + endpoint
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json().get("data", [])
        logger.info("Data successfully fetched from IODA API.")
        return data
    except requests.RequestException as e:
        logger.error(f"Error fetching data from IODA API: {e}")
        return []


def fetch_outage_data(
        entity_code: str,
        from_date: str,
        to_date: str,
        entity_type: str = "region",
) -> dict:

    endpoint = f"outages/events/"
    params = {
        "from": from_date,
        "until": to_date,
        "entityCode": entity_code,
        "entityType": entity_type,
        "includeAlerts": "true",
        "overall": "true",
    }
    url = API_URL + endpoint
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json().get("data", [])
        logger.info("Data successfully fetched from IODA API.")
        return data
    except requests.RequestException as e:
        logger.error(f"Error fetching data from IODA API: {e}")
        return {}

