"""
dyadic_sim/generation/agents/openai_agent.py

OpenAI wrapper (GPT-4o, o1, etc.)
Requires OPENAI_API_KEY in environment / .env
"""

import os

from openai import APIConnectionError, APIStatusError, OpenAI

from dyadic_sim.generation.agents.base_agent import AgentResponse, BaseAgent, Message


class OpenAIAgent(BaseAgent):
    """Agent backed by OpenAI's API."""

    def __init__(self, model: str, role: str):
        super().__init__(model, role)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OSError(
                "OPENAI_API_KEY not set. "
                "Add it to your .env file or environment."
            )
        self._client = OpenAI(api_key=api_key)

    def complete(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AgentResponse:
        formatted = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            if msg.role in ("user", "assistant"):
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
            provider="openai",
        )

    def is_available(self) -> bool:
        try:
            self._client.models.retrieve(self.model)
            return True
        except (APIConnectionError, APIStatusError):
            return False
        # Backstop for a health check: specific transport errors are caught
        # above, and anything else still means 'not available'.
        except Exception:  # noqa: BLE001
            return False
