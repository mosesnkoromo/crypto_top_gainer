"""
src/alerts/whatsapp.py — Whapi.cloud
HTTP 200 with {"sent": true} = success.
"""

import time
import requests
from config import WhatsAppConfig
from src.utils.logger import get_logger

log = get_logger(__name__)


class WhatsAppSender:

    def __init__(self, cfg: WhatsAppConfig):
        self._cfg = cfg
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        self._number = cfg.number.strip().lstrip("+")
        log.info("WhatsApp sender ready — number: %s", self._number)

    def send(self, message: str) -> bool:
        for attempt in range(1, self._cfg.retry_attempts + 1):
            ok, retry = self._try_send(message, attempt)
            if ok:
                return True
            if not retry or attempt == self._cfg.retry_attempts:
                break
            delay = self._cfg.retry_delay_seconds * (2 ** (attempt - 1))
            log.warning("Retrying in %ds (attempt %d/%d)…", delay, attempt, self._cfg.retry_attempts)
            time.sleep(delay)
        log.error("All %d attempts failed: %s…", self._cfg.retry_attempts, message[:60])
        return False

    def _try_send(self, message: str, attempt: int) -> tuple[bool, bool]:
        try:
            resp = self._session.post(
                "https://gate.whapi.cloud/messages/text",
                json={"to": self._number, "body": message},
                timeout=self._cfg.request_timeout,
            )
            data = resp.json() if resp.text.strip().startswith("{") else {}
            log.info("Whapi [HTTP %d]: sent=%s", resp.status_code, data.get("sent"))

            if resp.status_code in (200, 201) and data.get("sent") is True:
                log.info("✅ Delivered (attempt %d): %s…", attempt, message[:60])
                return True, False

            if resp.status_code == 429:
                log.warning("Rate-limited (429)")
                return False, True
            if resp.status_code in (401, 403):
                log.error("Auth error %d — check WHAPI_TOKEN", resp.status_code)
                return False, False

            log.warning("Whapi HTTP %d: %s", resp.status_code, resp.text[:200])
            return False, True

        except requests.Timeout:
            log.warning("Whapi timed out (attempt %d)", attempt)
            return False, True
        except Exception as exc:
            log.warning("Whapi error (attempt %d): %s", attempt, exc)
            return False, True
