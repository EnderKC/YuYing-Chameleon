"""Media type helpers for remote/local image handling."""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Optional

import httpx


def _url_path(url: str) -> str:
    try:
        return urllib.parse.urlparse((url or "").strip()).path or ""
    except Exception:
        return ""


def looks_like_gif_ref(value: Optional[str]) -> bool:
    v = (value or "").strip().lower()
    if not v:
        return False
    if v.startswith("data:"):
        return v.startswith("data:image/gif")
    path = _url_path(v) if v.startswith(("http://", "https://")) else v
    path = path.split("?", 1)[0].split("#", 1)[0]
    return path.endswith(".gif")


def looks_like_gif_path(path: Optional[str]) -> bool:
    try:
        return str(Path(str(path)).suffix).lower() == ".gif"
    except Exception:
        return False


async def is_remote_gif(url: str, *, timeout_seconds: float = 1.5) -> bool:
    """Best-effort GIF detection for remote URLs.

    - Fast path: .gif suffix / data:image/gif
    - Otherwise: HEAD Content-Type; fallback to small range GET and check GIF magic bytes.
    """

    u = (url or "").strip()
    if not u:
        return False
    if looks_like_gif_ref(u):
        return True
    if not u.startswith(("http://", "https://")):
        return False

    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        # 1) HEAD Content-Type
        try:
            resp = await client.head(u)
            ctype = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if ctype == "image/gif":
                return True
            if ctype.startswith("image/") and ctype != "":
                return False
        except Exception:
            pass

        # 2) Range GET, check magic bytes
        try:
            headers = {"Range": "bytes=0-5"}
            async with client.stream("GET", u, headers=headers) as resp:
                data = b""
                async for chunk in resp.aiter_bytes():
                    data += chunk
                    if len(data) >= 6:
                        break
            return data.startswith((b"GIF87a", b"GIF89a"))
        except Exception:
            return False

