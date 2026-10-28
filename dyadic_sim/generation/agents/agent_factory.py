"""
dyadic_sim/generation/agents/agent_factory.py

Builds the correct agent subclass from a model name and the models.yaml registry.
Provider is inferred from which section of models.yaml the model appears in.
"""

import yaml

from dyadic_sim.core import paths
from dyadic_sim.generation.agents.base_agent import BaseAgent
from dyadic_sim.generation.agents.claude_agent import ClaudeAgent
from dyadic_sim.generation.agents.local_agent import LocalAgent
from dyadic_sim.generation.agents.openai_agent import OpenAIAgent


def _load_registry(config_path: str | None = None) -> dict:
    with open(config_path or paths.config_path("models.yaml")) as f:
        return yaml.safe_load(f)


def _infer_provider(model: str, registry: dict) -> str:
    """Infer which provider a model name belongs to."""
    providers = registry.get("providers", {})
    for provider_name, provider_cfg in providers.items():
        if model in provider_cfg.get("models", []):
            return provider_name
    raise ValueError(
        f"Model '{model}' not found in config/models.yaml. "
        f"Add it to the appropriate provider's model list."
    )


def build_agent(model: str, role: str, config_path: str | None = None) -> BaseAgent:
    """
    Build and return the correct agent for a given model name and role.

    Args:
        model:        Model identifier, e.g. "llama3.1", "claude-sonnet-4-6"
        role:         Dyadic role: "therapist" or "patient"
        config_path:  Path to models.yaml

    Returns:
        Configured BaseAgent subclass instance

    Example:
        therapist = build_agent("llama3.1", "therapist")
        patient   = build_agent("mistral-nemo", "patient")
    """
    registry = _load_registry(config_path)
    provider = _infer_provider(model, registry)

    if provider == "ollama":
        base_url = registry["providers"]["ollama"].get("base_url")
        return LocalAgent(model=model, role=role, base_url=base_url)

    elif provider == "anthropic":
        return ClaudeAgent(model=model, role=role)

    elif provider == "openai":
        return OpenAIAgent(model=model, role=role)

    else:
        raise ValueError(f"Unknown provider '{provider}' for model '{model}'.")
