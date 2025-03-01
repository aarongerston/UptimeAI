"""
config.py

Global configuration constants for the ML pipeline.
"""
import logging

# Setup basic logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

logger = logging.getLogger(__name__)

# IODA API Endpoint
API_URL = "https://api.ioda.inetintel.cc.gatech.edu/v2/"

# Example default parameters for fetching data
DEFAULT_FROM_DATE = "1704067200"  # POSIX timestamp, e.g., 2024-12-31
DEFAULT_TO_DATE = "1704239999"    # POSIX timestamp, e.g., 2025-01-02
DEFAULT_DATASOURCE = ""           # e.g., "bgp", "merit-nt", etc.
DEFAULT_COUNTRY = "Spain"
