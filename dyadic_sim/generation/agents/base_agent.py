"""
dyadic_sim/generation/agents/base_agent.py

Abstract base class for all LLM agent wrappers.
Every provider (Claude, OpenAI, Ollama) implements this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    """A single message in a conversation."""
    role: str          # "system", "user", or "assistant"
    content: str


@dataclass
class AgentResponse:
    """The response from an agent turn."""
    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    provider: str | None = None
    # Diagnostics, saved with the turn. None means the wrapper did not report
    # it; a record keeps that as unknown and never fills in a provider default.
    finish_reason: str | None = None
    # The settings the wrapper sent, exactly as sent; messages excluded.
    request: dict | None = None
    # What the provider confirmed about the model that answered (see observed()).
    observed: dict | None = None


def observed(
    response_model: str | None = None,
    digest: str | None = None,
    quantization: str | None = None,
    context_length: int | None = None,
) -> dict:
    """Facts about the serving model, each None unless the provider confirmed it.

    The keys are always present, so a record says "unknown" explicitly rather
    than by omission.
    """
    return {
        "response_model": response_model,
        "digest": digest,
        "quantization": quantization,
        "context_length": context_length,
    }


class BaseAgent(ABC):
    """
    Abstract agent. Every provider subclass must implement `complete`.

    The agent is stateless: it receives the full conversation context
    on each call and returns a response. State is managed externally
    by the simulation layer and injected into the context each turn.
    """

    def __init__(self, model: str, role: str):
        """
        Args:
            model:  The model identifier (e.g. "llama3.1", "gpt-4o")
            role:   The dyadic role this agent plays ("therapist" or "patient")
        """
        self.model = model
        self.role = role

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AgentResponse:
        """
        Send a completion request to the underlying model.

        Args:
            system_prompt:  The role + prior context, built by TurnManager
            messages:       Conversation history (user/assistant alternating)
            max_tokens:     Maximum tokens to generate
            temperature:    Sampling temperature

        Returns:
            AgentResponse with content and usage metadata
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the model / provider is reachable."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, role={self.role})"
