"""OAuth2 token management for Freesound CLI downloads.

The /sounds/{id}/download/ endpoint requires OAuth2 (Bearer token).
Token auth (API key) only works for read-only endpoints like search and analysis.

Credentials (client_id / client_secret) must be passed explicitly by the
caller; this module never reads them from environment variables. Tokens are
cached in .freesound_tokens.json and refreshed automatically. The interactive
authorization code step is delegated to a ``code_provider`` callable so that
callers can wire it to a GUI dialog or a CLI ``input()`` prompt.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Callable, Optional

import requests

_AUTH_URL = "https://freesound.org/apiv2/oauth2/authorize/"
_TOKEN_URL = "https://freesound.org/apiv2/oauth2/access_token/"
_REDIRECT_URI = "https://freesound.org/home/app_permissions/permission_granted/"
_TOKEN_CACHE = Path(".freesound_tokens.json")

CodeProvider = Callable[[str], str]


def _load_cache() -> Optional[dict]:
    try:
        return json.loads(_TOKEN_CACHE.read_text()) if _TOKEN_CACHE.exists() else None
    except Exception:
        return None


def _save_cache(data: dict) -> None:
    _TOKEN_CACHE.write_text(json.dumps(data, indent=2))


def clear_cache() -> None:
    """Delete the cached token file (call when credentials change)."""
    _TOKEN_CACHE.unlink(missing_ok=True)


def authorization_url(client_id: str) -> str:
    return _AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": _REDIRECT_URI,
    })


def _post_token(client_id: str, client_secret: str, **payload) -> dict:
    resp = requests.post(
        _TOKEN_URL,
        data={"client_id": client_id, "client_secret": client_secret, **payload},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    data["expires_at"] = time.time() + data.get("expires_in", 86400)
    return data


def exchange_code(client_id: str, client_secret: str, code: str) -> dict:
    """Exchange an authorization code for tokens and cache them."""
    data = _post_token(
        client_id, client_secret,
        grant_type="authorization_code",
        code=code.strip(),
        redirect_uri=_REDIRECT_URI,
    )
    _save_cache(data)
    return data


def _run_authorization_flow(
    client_id: str,
    client_secret: str,
    code_provider: Optional[CodeProvider],
) -> dict:
    """Open the browser, ask ``code_provider`` for the code, exchange and cache."""
    if code_provider is None:
        raise RuntimeError(
            "Freesound OAuth2 authorization required but no code_provider given. "
            "Authorize from the UI before starting a download."
        )

    auth_url = authorization_url(client_id)
    webbrowser.open(auth_url)
    code = code_provider(auth_url)
    if not code or not code.strip():
        raise ValueError("No authorization code provided.")
    return exchange_code(client_id, client_secret, code)


def get_access_token(
    client_id: str,
    client_secret: str,
    code_provider: Optional[CodeProvider] = None,
) -> str:
    """Return a valid access token, refreshing or re-authorizing as needed."""
    cached = _load_cache()

    if cached:
        if time.time() < cached.get("expires_at", 0) - 60:
            return cached["access_token"]
        if cached.get("refresh_token"):
            try:
                data = _post_token(
                    client_id, client_secret,
                    grant_type="refresh_token",
                    refresh_token=cached["refresh_token"],
                )
                _save_cache(data)
                return data["access_token"]
            except Exception:
                pass

    return _run_authorization_flow(client_id, client_secret, code_provider)["access_token"]


def has_valid_cached_token() -> bool:
    cached = _load_cache()
    if not cached:
        return False
    return time.time() < cached.get("expires_at", 0) - 60
