"""Small Qwen-assisted experiment helper.

The API is used only to propose candidate configurations. Prediction remains
local and every proposed config is validated before it can be selected.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .qwen_client import QwenClient


def compact_json(value: Any, max_chars: int = 6000) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 20)] + "...<truncated>"


def extract_json_payload(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    starts: List[Tuple[int, str]] = [(text.find("{"), "}"), (text.find("["), "]")]
    for start, closer in sorted((item for item in starts if item[0] >= 0), key=lambda item: item[0]):
        end = text.rfind(closer)
        if end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def ask_qwen_json(
    messages: List[Dict[str, str]],
    env_path: str | Path = ".env",
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1800,
) -> Tuple[Any, Dict[str, Any]]:
    client = QwenClient(env_path=env_path, model=model)
    report: Dict[str, Any] = {
        "enabled": True,
        "available": client.available,
        "model": client.model,
        "base_url": client.base_url,
        "status": "not_called",
        "raw_response_chars": 0,
    }
    if not client.available:
        report["status"] = "missing_api_key"
        return None, report

    content = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    report["raw_response_chars"] = len(content)
    if not content:
        report["status"] = "empty_response_or_error"
        return None, report

    parsed = extract_json_payload(content)
    if parsed is None:
        report["status"] = "parse_failed"
        return None, report
    report["status"] = "ok"
    return parsed, report
