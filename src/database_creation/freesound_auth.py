"""OAuth2 token management for Freesound CLI downloads.

The /sounds/{id}/download/ endpoint requires OAuth2 (Bearer token).
Token auth (API key) only works for read-only endpoints like search and analysis.

One-time setup
--------------
1. Go to https://freesound.org/apiv2/apply/ and open your app settings.
2. Set the Redirect URI to:
       https://freesound.org/home/app_permissions/permission_granted/
3. Add to your .env:
       FREESOUND_CLIENT_SECRET=<your client secret>
   (FREESOUND_CLIENT_ID defaults to FREESOUND_API_KEY if not set separately)

After the first authorization, tokens are cached in .freesound_tokens.json
and refreshed automatically. Re-authorization is only needed when the refresh
token expires (typically after a long period of inactivity).
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Optional

import requests

_AUTH_URL = "https://freesound.org/apiv2/oauth2/authorize/"
_TOKEN_URL = "https://freesound.org/apiv2/oauth2/access_token/"
_REDIRECT_URI = "https://freesound.org/home/app_permissions/permission_granted/"
_TOKEN_CACHE = Path(".freesound_tokens.json")


def _load_cache() -> Optional[dict]:
    try:
        return json.loads(_TOKEN_CACHE.read_text()) if _TOKEN_CACHE.exists() else None
    except Exception:
        return None


def _save_cache(data: dict) -> None:
    _TOKEN_CACHE.write_text(json.dumps(data, indent=2))


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


def _run_authorization_flow(client_id: str, client_secret: str) -> dict:
    """Open browser for authorization, ask user to paste the code, return token data."""
    auth_url = _AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": _REDIRECT_URI,
    })

    print("\n" + "=" * 68)
    print("Freesound OAuth2 — one-time authorization needed for downloads")
    print("=" * 68)
    print("1. Opening browser (or copy the URL manually):")
    print(f"   {auth_url}")
    print("2. Log in to Freesound and authorize the app.")
    print("3. The page will show an authorization code — paste it below.")
    print("=" * 68)
    webbrowser.open(auth_url)

    code = input("Authorization code: ").strip()
    if not code:
        raise ValueError("No authorization code provided.")

    data = _post_token(
        client_id, client_secret,
        grant_type="authorization_code",
        code=code,
        redirect_uri=_REDIRECT_URI,
    )
    _save_cache(data)
    print("[OK] Authorized. Tokens cached in .freesound_tokens.json\n")
    return data


def get_access_token(client_id: str, client_secret: str) -> str:
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
            except Exception as exc:
                print(f"Token refresh failed ({exc}), re-authorizing...")

    return _run_authorization_flow(client_id, client_secret)["access_token"]


def load_oauth2_credentials() -> Optional[tuple[str, str]]:
    """Return (client_id, client_secret) from env, or None if not configured."""
    client_id = (os.getenv("FREESOUND_CLIENT_ID") or os.getenv("FREESOUND_API_KEY", "")).strip()
    client_secret = os.getenv("FREESOUND_CLIENT_SECRET", "").strip()
    if client_id and client_secret:
        return client_id, client_secret
    return None
