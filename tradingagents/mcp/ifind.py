from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

from tradingagents.dataflows.symbol_utils import normalize_a_share_symbol


class IFinDConfigError(RuntimeError):
    pass


class IFinDRequestError(RuntimeError):
    pass


def _normalize_codes(codes: str) -> str:
    normalized = []
    for raw in str(codes).split(","):
        value = raw.strip()
        if not value:
            continue
        try:
            normalized.append(normalize_a_share_symbol(value))
        except Exception:  # noqa: BLE001
            # iFinD also supports non-A-share instruments; keep their native code.
            normalized.append(value.upper())
    if not normalized:
        raise ValueError("codes不能为空")
    return ",".join(normalized)


def _split_indicator_params(indicators: str, params: str = "") -> list[dict[str, Any]]:
    indicator_list = [item.strip() for item in str(indicators).split(";") if item.strip()]
    if not indicator_list:
        raise ValueError("indicators不能为空")
    param_groups = str(params or "").split(";")
    result = []
    for idx, indicator in enumerate(indicator_list):
        group = param_groups[idx] if idx < len(param_groups) else ""
        values = [item.strip() for item in group.split(",")] if group != "" else []
        item: dict[str, Any] = {"indicator": indicator}
        if values:
            item["indiparams"] = values
        result.append(item)
    return result


@dataclass
class _TokenCache:
    value: str = ""
    expires_at: float = 0.0


class IFinDHTTPClient:
    """Minimal iFinD HTTP API client suitable for wrapping as MCP tools.

    Credentials never enter LLM prompts. The long-lived refresh token is read only from
    process configuration, exchanged for a short-lived access token, and then attached
    to HTTP headers server-side.
    """

    def __init__(
        self,
        *,
        refresh_token: str,
        base_url: str = "https://quantapi.51ifind.com/api/v1",
        timeout: float = 20.0,
        verify_ssl: bool = True,
        language: str = "cn",
    ):
        if not refresh_token:
            raise IFinDConfigError("未配置TRADINGAGENTS_IFIND_REFRESH_TOKEN")
        self.refresh_token = refresh_token.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.verify_ssl = bool(verify_ssl)
        self.language = language
        self.session = requests.Session()
        self._token = _TokenCache()
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "IFinDHTTPClient":
        return cls(
            refresh_token=str(config.get("ifind_refresh_token", "") or ""),
            base_url=str(config.get("ifind_base_url", "https://quantapi.51ifind.com/api/v1")),
            timeout=float(config.get("ifind_timeout", 20.0)),
            verify_ssl=bool(config.get("ifind_verify_ssl", True)),
            language=str(config.get("ifind_language", "cn")),
        )

    def _access_token(self, *, force: bool = False) -> str:
        now = time.time()
        if not force and self._token.value and now < self._token.expires_at:
            return self._token.value
        with self._lock:
            now = time.time()
            if not force and self._token.value and now < self._token.expires_at:
                return self._token.value
            response = self.session.post(
                f"{self.base_url}/get_access_token",
                headers={"Content-Type": "application/json", "refresh_token": self.refresh_token},
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            try:
                response.raise_for_status()
                payload = response.json()
                token = str(payload.get("data", {}).get("access_token", "") or "")
            except Exception as exc:  # noqa: BLE001
                raise IFinDRequestError(f"获取iFinD access_token失败: HTTP {response.status_code}") from exc
            if not token:
                raise IFinDRequestError(f"获取iFinD access_token失败: {payload}")
            self._token = _TokenCache(token, now + 6 * 24 * 3600)
            return token

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        for attempt in range(2):
            token = self._access_token(force=attempt > 0)
            response = self.session.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "access_token": token,
                    "ifindlang": self.language,
                },
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            if response.status_code in {401, 403} and attempt == 0:
                continue
            try:
                response.raise_for_status()
                data = response.json()
            except Exception as exc:  # noqa: BLE001
                body = response.text[:500]
                raise IFinDRequestError(f"iFinD请求失败: HTTP {response.status_code}: {body}") from exc
            errorcode = data.get("errorcode")
            if errorcode not in (None, 0, "0"):
                raise IFinDRequestError(f"iFinD errorcode={errorcode}: {data.get('errmsg', '')}")
            return data
        raise IFinDRequestError("iFinD鉴权失败")

    def snapshot(self, *, codes: str, indicators: str = "latest", start_time: str = "", end_time: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {"codes": _normalize_codes(codes), "indicators": indicators}
        if start_time:
            payload["starttime"] = start_time
        if end_time:
            payload["endtime"] = end_time
        return self._post("snap_shot", payload)

    def basic_data(self, *, codes: str, indicators: str, params: str = "") -> dict[str, Any]:
        return self._post(
            "basic_data_service",
            {"codes": _normalize_codes(codes), "indipara": _split_indicator_params(indicators, params)},
        )

    def date_sequence(
        self,
        *,
        codes: str,
        indicators: str,
        params: str,
        start_date: str,
        end_date: str,
        fill: str = "Blank",
        days: str = "Tradedays",
        interval: str = "D",
    ) -> dict[str, Any]:
        return self._post(
            "date_sequence",
            {
                "codes": _normalize_codes(codes),
                "startdate": start_date.replace("-", ""),
                "enddate": end_date.replace("-", ""),
                "functionpara": {"Fill": fill, "Days": days, "Interval": interval},
                "indipara": _split_indicator_params(indicators, params),
            },
        )


def compact_ifind_json(data: dict[str, Any], *, max_chars: int = 12000) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text if len(text) <= max_chars else text[:max_chars] + "...[iFinD结果已截断]"
