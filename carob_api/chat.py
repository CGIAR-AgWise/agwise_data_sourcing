import requests
from pydantic import BaseModel
from .config import BASE_URL


class ChatRequest(BaseModel):
    question: str
    token: str


def chatCarob(question: str, token: str):
    """Query Carob Chat endpoint."""
    url = f"{BASE_URL}/all/0/100"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {"text": question}
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()
