"""
dyadic_sim/generation/agents/claude_agent.py

Anthropic Claude wrapper.
Requires ANTHROPIC_API_KEY in environment / .env
"""

import os

import anthropic
from anthropic import APIConnectionError, APIStatusError

from dyadic_sim.generation.agents.base_agent import AgentResponse, BaseAgent, Message


class ClaudeAgent(BaseAgent):
    """Agent backed by Anthropic's Claude API."""

    def __init__(self, model: str, role: str):
        super().__init__(model, role)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise OSError(
                "ANTHROPIC_API_KEY not set. "
                "Add it to your .env file or environment."
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AgentResponse:
        # Anthropic's API takes system separately from messages
        # and requires user/assistant alternation
        formatted = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            # Anthropic does not accept "system" in the messages list
            if msg.role in ("user", "assistant")
        ]

        response = self._client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=formatted,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return AgentResponse(
            content=response.content[0].text,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            provider="anthropic",
        )

    def is_available(self) -> bool:
        try:
            # Minimal test call
            self._client.messages.create(
                model=self.model,
                system="ping",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True
        except (APIConnectionError, APIStatusError):
            return False
        # Backstop for a health check: specific transport errors are caught
        # above, and anything else still means 'not available'.
        except Exception:  # noqa: BLE001
            return False
