"""Explicit model routing and auditable JSON completions for the paper workflow."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass(frozen=True)
class Models:
    # Paper 1: local Llama for stages 1/2 and a three-model stage-3 ensemble.
    process: str = "llama3.1:70b"
    themes: str = "llama3.1:70b"
    edges: tuple[str, str, str] = ("llama3.1:70b", "qwen2.5:72b", "gpt-4o-mini")
    # Graph2Counsel uses GPT-4o for profile generation and evaluation.
    profile: str = "gpt-4o"
    judge: str = "gpt-4o"
    dialogue: str = "mistral-nemo"
    mode: str = "paper"

    @classmethod
    def from_env(cls, mode: str = "paper") -> Models:
        """Select the paper models or an explicitly labeled local adaptation."""
        if mode == "local":
            llama = "llama3.1:70b"
            qwen = "qwen2.5:72b"
            edge = os.getenv("LOCAL_EDGE_MODEL_ID") or "gpt-oss:20b"
            profile = os.getenv("LOCAL_PROFILE_MODEL_ID") or "gemma3:27b"
            for model in (edge, profile):
                if model.endswith(("-cloud", ":cloud")):
                    raise ValueError(f"Cloud model {model!r} is not allowed in local mode")
            return cls(process=llama, themes=llama, edges=(llama, qwen, edge),
                       profile=profile, judge=profile, mode=mode)
        if mode != "paper":
            raise ValueError(f"Unknown graph model mode: {mode}")
        llama = os.getenv("PAPER_LLAMA_MODEL_ID", "llama3.1:70b")
        qwen = os.getenv("PAPER_QWEN_MODEL_ID", "qwen2.5:72b")
        return cls(process=llama, themes=llama, edges=(llama, qwen, "gpt-4o-mini"))

    def required(self) -> set[str]:
        return {self.process, self.themes, *self.edges, self.profile, self.judge, self.dialogue}


class CompletionClient:
    def __init__(self, models: Models, trace_path: Path):
        self.models = models
        self.trace_path = trace_path
        self._clients: dict[str, OpenAI] = {}

    def _client(self, model: str) -> OpenAI:
        kind = self._provider(model)
        if kind not in self._clients:
            if kind == "openai":
                key = os.getenv("OPENAI_API_KEY")
                if not key:
                    raise RuntimeError("OPENAI_API_KEY is required for the paper's GPT models")
                self._clients[kind] = OpenAI(api_key=key, timeout=180, max_retries=2)
            elif kind == "graph_endpoint":
                self._clients[kind] = OpenAI(
                    base_url=os.environ["GRAPH_MODEL_BASE_URL"].rstrip("/"),
                    api_key=os.getenv("GRAPH_MODEL_API_KEY", "unused"),
                    timeout=600, max_retries=1,
                )
            else:
                url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
                self._clients[kind] = OpenAI(
                    base_url=f"{url}/v1", api_key="ollama", timeout=600, max_retries=1
                )
        return self._clients[kind]

    def _provider(self, model: str) -> str:
        if self.models.mode == "local":
            return "ollama"
        if model.startswith(("gpt-", "o1", "o3", "o4")):
            return "openai"
        if model != self.models.dialogue and os.getenv("GRAPH_MODEL_BASE_URL"):
            return "graph_endpoint"
        return "ollama"

    def preflight(self) -> None:
        missing = []
        for model in sorted(self.models.required()):
            try:
                if self._provider(model) in {"ollama", "graph_endpoint"}:
                    names = {x.id for x in self._client(model).with_options(
                        timeout=12, max_retries=0).models.list().data}
                    if model not in names and f"{model}:latest" not in names:
                        missing.append(f"{self._provider(model)} model {model} "
                                       f"(available: {', '.join(sorted(names))})")
                else:
                    self._client(model).with_options(timeout=12, max_retries=0).models.retrieve(model)
            except Exception as exc:  # noqa: BLE001 - report every failed model in one preflight
                missing.append(f"{model}: {type(exc).__name__}: {exc}")
        if missing:
            raise RuntimeError("Model preflight failed:\n- " + "\n- ".join(missing))

    def complete(self, model: str, system: str, user: str, *, temperature: float = 0,
                 max_tokens: int = 3000, json_output: bool = True) -> object:
        for attempt in range(3):
            response = self._client(model).chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"model": model, "temperature": temperature,
                                         "system": system, "user": user, "response": content,
                                         "response_model": response.model,
                                         "response_id": response.id,
                                         "system_fingerprint": response.system_fingerprint,
                                         "usage": response.usage.model_dump() if response.usage else None,
                                         "attempt": attempt + 1}, ensure_ascii=False) + "\n")
            if not json_output:
                return content
            try:
                stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
                return json.loads(stripped)
            except json.JSONDecodeError:
                user += "\nReturn valid JSON only, with no markdown fence."
        raise ValueError(f"{model} did not produce valid JSON after three attempts; see {self.trace_path}")
