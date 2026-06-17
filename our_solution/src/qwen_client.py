"""Optional Qwen API helper.

The prediction pipeline does not require an LLM. This module is only for
optional agent narration or future strategy generation, and it accepts either a
standard KEY=value .env file or a single-line raw API key file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


def load_env_file(path: str | Path = ".env") -> Dict[str, str]:
    path = Path(path)
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "=" in text:
            key, value = text.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        elif text.startswith("sk-") or text.startswith("dashscope"):
            values["QWEN_API_KEY"] = text
    return values


class QwenClient:
    def __init__(
        self,
        env_path: str | Path = ".env",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        env_values = load_env_file(env_path)
        self.api_key = (
            env_values.get("QWEN_API_KEY")
            or env_values.get("DASHSCOPE_API_KEY")
            or env_values.get("LLM_API_KEY")
            or os.environ.get("QWEN_API_KEY", "")
            or os.environ.get("DASHSCOPE_API_KEY", "")
            or os.environ.get("LLM_API_KEY", "")
        )
        self.model = (
            model
            or env_values.get("QWEN_MODEL")
            or os.environ.get("QWEN_MODEL")
            or DEFAULT_MODEL
        )
        self.base_url = (
            base_url
            or env_values.get("QWEN_BASE_URL")
            or env_values.get("LLM_BASE_URL")
            or os.environ.get("QWEN_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 1024) -> str:
        if not self.available:
            return ""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
        except Exception:
            return ""

