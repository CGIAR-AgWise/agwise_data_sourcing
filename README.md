# carob_api

A minimal Python wrapper around the Qvantum Carob API.  
Provides three simple functions:

- `getToken(client_id, client_secret)` — fetch OAuth2 client-credentials token.
- `dataCarob(country, crop, token)` — GET datapool results for a country & crop.
- `chatCarob(question, token)` — POST a question to the datapool chat endpoint.

## Requirements

- Python 3.9+
- `requests`
- `pydantic` (used for optional argument validation)
