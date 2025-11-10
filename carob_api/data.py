import requests
from pydantic import BaseModel
from .config import BASE_URL


class DataRequest(BaseModel):
    country: str
    crop: str
    token: str


def dataCarob(country: str, crop: str, token: str):
    """Fetch data for a given country and crop."""
    url = f"{BASE_URL}/{country}/{crop}/0/100"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()
