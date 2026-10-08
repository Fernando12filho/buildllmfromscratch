"""Shared HTTP helper for all scrapers.

Standard library only (urllib), so downloading works before installing anything.
Being polite matters: these are public government servers, so we wait between
requests and back off when the server says it's busy.
"""

import json
import random
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "llmbr-research-scraper/0.1 (educational LLM project)"

# python.org builds of Python on macOS ship without root certificates, which makes
# every HTTPS request fail with CERTIFICATE_VERIFY_FAILED. Use certifi's CA bundle
# when available; otherwise fall back to the system default.
try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()


def get_json(url: str, params: dict | None = None, *, retries: int = 5,
             delay: float = 0.3, timeout: float = 30.0) -> dict:
    """GET a URL and parse JSON, retrying on network errors, 429 and 5xx.

    `delay` is slept after every successful request (rate limit).
    Retries use exponential backoff: ~1s, 2s, 4s, 8s, ...
    """
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
                data = json.load(resp)
            time.sleep(delay)
            return data
        except urllib.error.HTTPError as e:
            # 4xx other than 429 won't fix itself by retrying.
            if e.code != 429 and e.code < 500:
                raise
            err = e
        except urllib.error.URLError as e:
            # Certificate problems are configuration, not flakiness — fail fast.
            if isinstance(e.reason, ssl.SSLCertVerificationError):
                raise RuntimeError("SSL certificate check failed: run "
                                   "`pip install certifi`") from e
            err = e
        except (TimeoutError, json.JSONDecodeError) as e:
            err = e
        wait = 2 ** attempt + random.random()
        print(f"  ! {err} — retry {attempt + 1}/{retries} in {wait:.1f}s")
        time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")
