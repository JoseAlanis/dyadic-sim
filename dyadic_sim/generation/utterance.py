"""Reduce a generated turn to spoken language only.

Local models narrate. Turns come back with stage directions such as
"(pauses, looking down)" or "*I lean forward slightly*", with a leaked role
label ("Patient:"), or wrapped whole in quotation marks. How much a model does
this differs between models, so left in place it confounds cross-model
comparison, and gesture words feed measures that are meant to read speech.

The priors ask for clean speech, but models comply unreliably, so this also runs
on every recorded turn. `dyadic_sim/generation/dyad.py` keeps the original in `*_text_raw`
whenever cleaning changed anything, so nothing is lost.

Policy: under the speech-only contract every bracketed span is treated as
narration and removed. The raw text is retained in `*_text_raw`, so a bracketed
span that was meant as speech can still be recovered.
"""

from __future__ import annotations

import re

# Bracketed narration: (...) and [...] spans.
_BRACKETED = re.compile(r"[(\[][^)\]]{0,300}[)\]]")

# Markdown-ish emphasis used as narration: **...** and *...* / _..._ spans.
_EMPHASIS = re.compile(r"\*\*[^*]{1,300}?\*\*|\*[^*\n]{1,300}?\*|(?<!\w)_[^_\n]{1,300}?_(?!\w)")

# A role label the model echoed instead of just speaking.
_ROLE_LABEL = re.compile(
    r"^\s*[*_\"']*\s*(?:you|your\s+response|your\s+reply|therapist|patient|assistant|me)\s*:\s*[*_\"']*\s*",
    re.IGNORECASE,
)

# Reasoning-model scratchpads. Some models (Qwen3-class, for example) emit
# them, and the tags would otherwise land in the transcript verbatim. Structural backstop only; the reliable
# control is the provider's own switch (ollama `think: false`), not a prompt.
_REASONING_BLOCK = re.compile(
    r"<(think|thinking|reasoning|scratchpad)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_STRAY_REASONING_TAG = re.compile(
    r"</?(think|thinking|reasoning|scratchpad)\b[^>]*>", re.IGNORECASE
)

# The whole turn delivered as a quotation, a habit that differs strongly between
# models. Stripped only when the quotes wrap the entire utterance, so quoted
# speech *inside* a turn survives.
_WRAPPING_QUOTES = re.compile(r'^\s*["“](?P<body>.*)["”]\s*$', re.DOTALL)

# A turn delivered as several quoted fragments and nothing else:
#   "I'm scared of dogs."  "I just want a normal life."
# Rare, but the same artifact, and the single-wrapper rule below deliberately
# skips it, because its body contains
# quotes, a guard that exists to protect split dialogue
# ('"Yes," he said, "fine."').
# Requiring the whole turn to be quoted fragments separates the two: real
# split dialogue has unquoted narration between the parts.
_ALL_QUOTED_SEGMENTS = re.compile(r'^\s*(?:["“][^"“”]+["”]\s*){2,}$')
_QUOTED_SEGMENT = re.compile(r'["“]([^"“”]+)["”]')

# Whitespace left behind once spans are removed.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_REPEATED_SPACE = re.compile(r"[ \t]{2,}")
_BLANK_LINES = re.compile(r"\n{3,}")


# Version of the *generation contract*: what the agents are told to produce and
# what is done to their output. Distinct from `metadata_schema_version`, which
# versions the metadata field layout. Recorded on every session, so analyses can
# filter on it; a record without the field is version 1. A session can only be
# continued by the version that started it (`check_continuable` in
# dyadic_sim/generation/session.py). Bump whenever a change makes new sessions incomparable
# with old ones.
#
# Version 6:
#   - the system prompt is the prior alone: the patient case (plus the injected
#     symptom block, if any) or the therapist base plus orientation variant;
#     both priors carry SPEECH_ONLY_INSTRUCTION and the therapist's also
#     STAY_WITH_WHAT_IS_SAID;
#   - the therapist opens with dyadic_sim/generation/opening.py's FIRST_SESSION_OPENING,
#     saved with the session as `initial_patient_prompt`;
#   - both agents receive the conversation in the order it was spoken, starting
#     from the saved opening (dyadic_sim/generation/turn_manager.py);
#   - every recorded turn passes through clean_utterance(); the original text
#     is kept in `*_text_raw` when cleaning changed it.
#
# Version 7, as version 6 except:
#   - the therapist runs a first-appointment intake: an intake role and stance
#     (therapist base.yaml, cbt variant without treatment techniques) and, per
#     reply, one step of base.yaml's `intake_agenda`, so its system prompt
#     changes with the turn;
#   - the patient prompt has a frame without a presumed problem, symptom facts
#     with a disclosure rule (config/priors/patient/symptom_facts.yaml),
#     symptoms the batch declares absent, a rule against inventing problems, and
#     plain-speech instructions.
#
# A prompt's identity is (GENERATION_VERSION, PROMPT_VARIANT). Every change to
# prompt text gets a new, immutable PROMPT_VARIANT ("7a", "7b", ...), never
# reused; a change to the generation mechanics gets a new GENERATION_VERSION.
# 7a: first intake candidate (development only). 7b: each reply acknowledges
# and then does its step, which takes precedence over general guidance; one
# fixed sleep question at reply 4; spoken summary at 9, no question at 10;
# compact baseline case with optional psychological sections; symptom rules
# scoped to the symptoms they name. 7c (candidate): reply 1 grounded in what was
# said (a greeting gets "what would you like to talk about"); reply 4's question
# without quotation marks; reply 9 summarises what was disclosed, health
# difficulties and qualifications included; the patient corrects premises that
# don't fit its profile.
# Both are recorded on every session and both must match to continue one.
GENERATION_VERSION = 7
PROMPT_VARIANT = "7c"

# Shared by both priors so the therapist and the patient are held to the same
# rule. Phrased concretely, with examples, because an abstract instruction not
# to describe one's actions is not followed reliably.
SPEECH_ONLY_INSTRUCTION = (
    "Write only the words spoken aloud, as in a verbatim transcript. "
    "No stage directions, no descriptions of gestures, posture, tone, or "
    "expression, and no narration of what you are doing or feeling. "
    "Do not use parentheses, brackets, or asterisks for actions. "
    "For example, never write \"(pauses)\" or \"*leans forward*\". "
    "Do not prefix your reply with a name or role label, and do not wrap your "
    "reply in quotation marks. Write the words plainly, as they were said."
)


# Therapist-only. A suggestive therapist can implant the thing being measured:
# if it proposes a symptom, a history or a feeling the patient never brought,
# the patient may take it up, and the run records symptom expression that the
# manipulation did not produce. Each clause closes one way of doing that:
#
#   - stating unstated facts (a guessed history invites agreement);
#   - raising intensity ("a bit tired" asked back as "most exhausted");
#   - supplying emotion words the patient has not used (relabelling an
#     annoyance as frustration, then as feeling trapped, walks the patient into
#     the vocabulary the detectors count);
#   - treating "nothing is wrong" as a puzzle rather than as an answer;
#   - talking about itself: base.yaml rules out self-disclosure in the
#     abstract, and the concrete instruction is here because the abstract one
#     is not followed reliably.
#
# Since version 7 the intake agenda raises the areas of life itself, so the
# rule allows neutral questions about them and forbids only suggesting a
# problem; "follow what they bring rather than introducing topics" would
# contradict the agenda.
#
# Phrased as clinical practice rather than methodology, so it reads as part of
# the therapist's stance rather than an experimental control.
STAY_WITH_WHAT_IS_SAID = (
    "Work only from what this person has actually told you. Do not state as "
    "fact anything they have not said, and do not fill in details of their "
    "history or their feelings on their behalf. If you have a hunch, hold it, "
    "or ask about it openly instead of asserting it. You may raise each area of "
    "today's agenda with a brief, neutral question; never suggest that they "
    "have a problem they have not described. "
    "Match the strength of what they said rather than raising it. If they call "
    "something a bit difficult, it is a bit difficult, not devastating; reflect "
    "it back at their intensity, in their words where you can. "
    "Do not supply emotion words they have not used: if they say annoyed, work "
    "with annoyed, not frustrated, hurt, or trapped; if they describe a thing, "
    "do not turn it into a feeling for them. "
    "If they tell you nothing is wrong, or that an area is fine, take that as "
    "their answer, not as a puzzle: accept it, do not press, and go on. "
    "Do not talk about your own life or work. If they ask about you, answer in "
    "a few words at most and return to them; the hour is theirs."
)


def _strip_wrapping_quotes(text: str) -> str:
    """Remove quotes that enclose the whole turn, keeping quotes used inside it."""
    if _ALL_QUOTED_SEGMENTS.match(text):
        return " ".join(m.strip() for m in _QUOTED_SEGMENT.findall(text)).strip()

    match = _WRAPPING_QUOTES.match(text)
    if match:
        body = match.group("body")
        # Only unwrap when the body isn't itself two quoted fragments, e.g.
        # '"yes," he said, "fine."', where the outer pair is not a wrapper.
        if '"' not in body and "“" not in body and "”" not in body:
            return body.strip()
    # Truncation at max_tokens can leave an opening quote with no partner. A
    # lone leading quote can't be legitimate quoted speech.
    if text[:1] in '"“' and (text.count('"') + text.count("“") + text.count("”")) == 1:
        return text[1:].strip()
    return text


def clean_utterance(text: str) -> str:
    """Strip narration, emphasis spans, leaked role labels, and wrapping quotes."""
    if not text:
        return ""

    cleaned = _REASONING_BLOCK.sub(" ", text)
    cleaned = _STRAY_REASONING_TAG.sub(" ", cleaned)
    cleaned = _BRACKETED.sub(" ", cleaned)
    cleaned = _EMPHASIS.sub(" ", cleaned)

    # The label can sit at the start of any line the model produced.
    cleaned = "\n".join(_ROLE_LABEL.sub("", line) for line in cleaned.split("\n"))

    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = _REPEATED_SPACE.sub(" ", cleaned)
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    cleaned = _strip_wrapping_quotes(cleaned.strip())

    # A turn can be nothing but a stage direction ("(pauses)"). Dropping it
    # would break the alternation, so keep the original and let the analysis
    # layer exclude it.
    if not cleaned:
        return text.strip()
    return cleaned
