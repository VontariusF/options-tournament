"""Featherless chat transport for the tournament specialist roster.

OpenAI-compatible ``/v1/chat/completions``. This module only moves messages:
Cloudflare User-Agent, concurrency cap, 429 downshift, thinking-token strip,
and a soft daily call budget. Risk math stays in Python.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_BASE_URL = "https://api.featherless.ai/v1"
DEFAULT_USER_AGENT = "options-tournament/1.0 (+https://featherless.ai)"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)


class LLMError(RuntimeError):
    """Transport or decode failure. Never carries a risk number."""


class LLMBudgetExceeded(LLMError):
    """Soft daily call budget is spent for this UTC day."""


def strip_reasoning(text: str) -> str:
    s = _THINK_RE.sub("", text or "").strip()
    return _FENCE_RE.sub("", s).strip()


def extract_json_object(content: str) -> dict:
    s = strip_reasoning(content)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        return {"_error": f"no JSON object in reply ({s[:80]!r})"}
    blob = s[i:j + 1]
    for attempt in (blob, blob.replace("\n", " ")):
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        return parsed if isinstance(parsed, dict) else {"_error": "JSON root is not an object"}
    return {"_error": f"bad JSON ({blob[:80]!r})"}


@dataclass
class ChatResponse:
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    model: str = ""
    raw: Optional[dict] = None

    def json(self) -> dict:
        for tc in self.tool_calls:
            args = (tc.get("function") or {}).get("arguments")
            if isinstance(args, dict):
                return args
            if isinstance(args, str) and args.strip():
                try:
                    return json.loads(args)
                except json.JSONDecodeError:
                    parsed = extract_json_object(args)
                    if "_error" not in parsed:
                        return parsed
        return extract_json_object(self.content)


class _DailyBudget:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = ""
        self._n = 0

    def check_and_increment(self, limit: int) -> None:
        if limit <= 0:
            return
        today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if today != self._day:
                self._day, self._n = today, 0
            if self._n >= limit:
                raise LLMBudgetExceeded(f"daily LLM call budget {limit} spent for {today}")
            self._n += 1


_BUDGET = _DailyBudget()


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except (TypeError, ValueError):
        return default


class FeatherlessClient:
    """OpenAI-compatible chat client. ``_post`` is the network seam for tests."""

    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 user_agent: Optional[str] = None, fallback_model: Optional[str] = None,
                 max_concurrency: Optional[int] = None, max_retries: int = 4,
                 daily_call_budget: Optional[int] = None, timeout: float = 90.0,
                 sleep=time.sleep) -> None:
        self.base_url = (base_url or os.environ.get("FEATHERLESS_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.user_agent = user_agent or os.environ.get("FEATHERLESS_USER_AGENT") or DEFAULT_USER_AGENT
        self.fallback_model = (fallback_model or os.environ.get("FEATHERLESS_FALLBACK_MODEL")
                               or "Qwen/Qwen3-4B-Instruct-2507")
        conc = max_concurrency if max_concurrency is not None else _env_int("FEATHERLESS_MAX_CONCURRENCY", 2)
        self._sem = threading.Semaphore(max(1, int(conc)))
        self._max_retries = max_retries
        self._budget_limit = (daily_call_budget if daily_call_budget is not None
                              else _env_int("LLM_DAILY_CALL_BUDGET", 400))
        self.timeout = timeout
        self._sleep = sleep
        self._api_key = api_key

    def _key(self) -> str:
        if self._api_key:
            return self._api_key
        try:
            from pma_common.secrets import get_secret
            k = get_secret("FEATHERLESS_API_KEY")
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"could not resolve FEATHERLESS_API_KEY: {e}")
        if not k:
            raise LLMError("FEATHERLESS_API_KEY not set (env or app_secrets)")
        self._api_key = k
        return k

    def _post(self, path: str, body: dict) -> tuple[int, dict]:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method="POST",
            headers={"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json",
                     "Accept": "application/json", "User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                payload = json.loads(e.read().decode())
            except Exception:  # noqa: BLE001
                payload = {"error": "http error"}
            return e.code, payload
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"{type(e).__name__}: {str(e)[:120]}")

    def chat(self, messages: List[Dict[str, Any]], *, model: str,
             tools: Optional[list] = None, tool_choice: Optional[Any] = None,
             response_format: Optional[dict] = None, temperature: float = 0.0,
             max_tokens: int = 700) -> ChatResponse:
        self._sem.acquire()
        try:
            return self._chat_with_retries(messages, model=model, tools=tools,
                                           tool_choice=tool_choice, response_format=response_format,
                                           temperature=temperature, max_tokens=max_tokens)
        finally:
            self._sem.release()

    def _chat_with_retries(self, messages, *, model, tools, tool_choice, response_format,
                           temperature, max_tokens) -> ChatResponse:
        if self._budget_limit:
            _BUDGET.check_and_increment(self._budget_limit)
        body: Dict[str, Any] = {"model": model, "messages": messages,
                                "temperature": float(temperature), "max_tokens": int(max_tokens)}
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        tried_fallback = False
        last: tuple[int, dict] = (0, {})
        for attempt in range(self._max_retries + 1):
            status, payload = self._post("/chat/completions", body)
            if status == 200:
                return self._normalize(payload, body["model"])
            last = (status, payload)
            if status == 429 or 500 <= status < 600:
                if attempt < self._max_retries:
                    self._sleep(min(8.0, 0.6 * (2 ** attempt)))
                    if status == 429 and not tried_fallback and body["model"] != self.fallback_model:
                        body["model"] = self.fallback_model
                        tried_fallback = True
                    continue
            break
        raise LLMError(f"chat failed (status={last[0]}): {str(last[1])[:120]}")

    @staticmethod
    def _normalize(payload: dict, model: str) -> ChatResponse:
        choice = (payload.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = strip_reasoning(msg.get("content") or "")
        tool_calls = msg.get("tool_calls") or []
        return ChatResponse(content=content, tool_calls=tool_calls,
                            finish_reason=str(choice.get("finish_reason") or ""),
                            model=str(payload.get("model") or model), raw=payload)


def default_client() -> FeatherlessClient:
    return FeatherlessClient()
