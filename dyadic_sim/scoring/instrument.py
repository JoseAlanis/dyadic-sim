"""The rating instrument: the topic taxonomy a rater scores against.

The scoring pipeline (parse -> filter for relevance -> rate -> aggregate) is
fixed. Everything that is specific to *which* clinical instrument is being
approximated lives in an `Instrument`: its topics, how a turn is recognised as
being about a topic, the score range, the language the rater expects, and how
the topics line up with the PHQ-style anchors written into patient priors.

Swapping instruments therefore means defining another `Instrument` and pointing
`ScoringConfig.instrument` at it, not editing the pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, eq=False)
class Instrument:
    """A topic taxonomy plus the metadata the pipeline needs to apply it."""

    key: str
    """Short registry name, e.g. ``"madrs"``."""

    name: str
    """Display name used in outputs and warnings, e.g. ``"MADRS"``."""

    topics: tuple[str, ...]
    """Topic identifiers, in the order the rater expects them."""

    topic_descriptions: Mapping[str, str]
    """English gloss per topic, used for semantic relevance matching."""

    relevance_patterns: Mapping[str, str]
    """English regex per topic, applied to the original (untranslated) turn."""

    score_range: tuple[float, float]
    """Inclusive severity range the rater emits, e.g. ``(0.0, 6.0)``."""

    language: str
    """Language the rater expects its input in; drives the translation step."""

    prior_topic_map: Mapping[str, str | None]
    """PHQ prior symptom key -> topic, or None where the instrument has no match."""

    prior_mapping_quality: Mapping[str, str]
    """How well each PHQ symptom maps onto its topic: close/partial/approximate/none."""

    topic_to_prior_symptom: Mapping[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "topic_to_prior_symptom",
            {
                topic: symptom
                for symptom, topic in self.prior_topic_map.items()
                if topic is not None
            },
        )
        self._validate()

    def _validate(self) -> None:
        missing_descriptions = [t for t in self.topics if t not in self.topic_descriptions]
        if missing_descriptions:
            raise ValueError(
                f"{self.name}: topics without a description: {missing_descriptions}"
            )
        missing_patterns = [t for t in self.topics if t not in self.relevance_patterns]
        if missing_patterns:
            raise ValueError(
                f"{self.name}: topics without a relevance pattern: {missing_patterns}"
            )
        unknown = sorted(
            {t for t in self.prior_topic_map.values() if t is not None} - set(self.topics)
        )
        if unknown:
            raise ValueError(f"{self.name}: prior map points at unknown topics: {unknown}")
        if set(self.prior_topic_map) != set(self.prior_mapping_quality):
            raise ValueError(
                f"{self.name}: prior_topic_map and prior_mapping_quality cover "
                "different symptoms"
            )
        low, high = self.score_range
        if not low < high:
            raise ValueError(f"{self.name}: score_range must be increasing, got {self.score_range}")

    def clip(self, value: float) -> float:
        """Constrain a raw rater output to the instrument's score range."""
        low, high = self.score_range
        return max(low, min(high, value))

    def needs_translation_from(self, source_language: str) -> bool:
        return source_language != self.language

    def describe(self) -> dict:
        """Instrument provenance recorded alongside every scored session."""
        return {
            "key": self.key,
            "name": self.name,
            "topics": list(self.topics),
            "score_range": list(self.score_range),
            "language": self.language,
        }
