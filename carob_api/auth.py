import requests
from pydantic import BaseModel
from .config import AUTH_URL, AUDIENCE


class AuthConfig(BaseModel):
    client_id: str
    client_secret: str


def getToken(client_id: str, client_secret: str) -> str:
    """Request an Auth0 token for the Qvantum API."""
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "audience": AUDIENCE,
        "grant_type": "client_credentials"
    }

    resp = requests.post(AUTH_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data.get("access_token")
