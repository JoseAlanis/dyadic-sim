"""
dyadic_sim/generation/agents/local_agent.py

Ollama wrapper for local models (Llama, Mistral, Gemma, etc.)
Ollama exposes an OpenAI-compatible API at localhost:11434,
so this uses the openai client pointed at the local endpoint.

Start Ollama before running: `ollama serve`
Pull a model before using: `ollama pull llama3.1`
"""

import http.client
import json
import os
import urllib.request

from openai import APIConnectionError, OpenAI

from dyadic_sim.generation.agents.base_agent import (
    AgentResponse,
    BaseAgent,
    Message,
    observed,
)


class LocalAgent(BaseAgent):
    """Agent backed by a locally running Ollama model."""

    def __init__(self, model: str, role: str, base_url: str | None = None):
        super().__init__(model, role)
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        # Ollama's OpenAI-compatible endpoint.
        #
        # The timeout is load-bearing, not defensive boilerplate. Left unset,
        # the OpenAI client waits 600s and retries, so a hung Ollama request
        # (for example after the model is evicted from VRAM mid-batch) stalls a
        # batch silently for half an hour. A turn normally takes well under a
        # minute, so 180s is ample slack; failing fast and loudly is worth more
        # than tolerating one slow request.
        self._client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="ollama",  # Ollama ignores the key but the client requires one
            timeout=float(os.getenv("OLLAMA_TIMEOUT", "180")),
            max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "2")),
        )

    def complete(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AgentResponse:
        formatted = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            formatted.append({"role": msg.role, "content": msg.content})

        request = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature}
        response = self._client.chat.completions.create(messages=formatted, **request)

        return AgentResponse(
            content=response.choices[0].message.content,
            model=self.model,
            input_tokens=getattr(response.usage, "prompt_tokens", None),
            output_tokens=getattr(response.usage, "completion_tokens", None),
            provider="ollama",
            finish_reason=response.choices[0].finish_reason,
            request=request,
            observed=self._observed(getattr(response, "model", None)),
        )

    def _observed(self, response_model: str | None) -> dict:
        """Digest, quantization and context length of the loaded model.

        Read from Ollama's /api/ps after the completion, when the model is
        loaded, so the context length is the one actually in effect rather than
        an assumed default. Anything the server does not confirm stays None.
        """
        name = self.model if ":" in self.model else f"{self.model}:latest"
        facts: dict = {}
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/ps", timeout=5) as reply:
                payload = json.load(reply)
            # The reply's shape is the server's, so every level is checked:
            # anything unexpected is "not confirmed", never an exception.
            models = payload.get("models") if isinstance(payload, dict) else None
            for entry in models if isinstance(models, list) else []:
                if isinstance(entry, dict) and name in (entry.get("name"), entry.get("model")):
                    details = entry.get("details")
                    facts = {
                        "digest": entry.get("digest"),
                        "quantization": (details.get("quantization_level")
                                         if isinstance(details, dict) else None),
                        "context_length": entry.get("context_length"),
                    }
                    break
        # The completion has already returned; diagnostics must never cost it.
        # An unreachable or non-Ollama server (OSError, HTTPException) or an
        # unreadable reply (ValueError) means "not confirmed".
        except (OSError, http.client.HTTPException, ValueError):
            facts = {}
        digest, quantization = facts.get("digest"), facts.get("quantization")
        context_length = facts.get("context_length")
        return observed(
            response_model=response_model if isinstance(response_model, str) else None,
            digest=digest if isinstance(digest, str) else None,
            quantization=quantization if isinstance(quantization, str) else None,
            context_length=context_length if type(context_length) is int else None,
        )

    def is_available(self) -> bool:
        try:
            # List available models (lightweight availability check)
            models = self._client.models.list()
            return any(self.model in m.id for m in models.data)
        except APIConnectionError:
            return False
        # Backstop for a health check: specific transport errors are caught
        # above, and anything else still means 'not available'.
        except Exception:  # noqa: BLE001
            return False
