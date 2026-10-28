"""
dyadic_sim/generation/agents/local_agent.py

Ollama wrapper for local models (Llama, Mistral, Gemma, etc.)
Ollama exposes an OpenAI-compatible API at localhost:11434,
so this uses the openai client pointed at the local endpoint.

Start Ollama before running: `ollama serve`
Pull a model before using: `ollama pull llama3.1`
"""

import os

from openai import APIConnectionError, OpenAI

from dyadic_sim.generation.agents.base_agent import AgentResponse, BaseAgent, Message


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

        response = self._client.chat.completions.create(
            model=self.model,
            messages=formatted,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return AgentResponse(
            content=response.choices[0].message.content,
            model=self.model,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            provider="ollama",
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
