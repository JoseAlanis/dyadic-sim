"""Reduce a generated turn to spoken language only.

Local models narrate: roughly one sixth of the characters in llama transcripts
are stage directions, such as "(pauses, looking down)" or "*I lean forward
slightly*", and mistral occasionally leaks a role label ("Patient:") into its
own reply.

That noise is not cosmetic. Annotation is unevenly distributed across models
(llama ~16% of characters, mistral ~0.6%), so any cross-model comparison partly
compares narration style, and gesture words feed straight into the
embedding/severity measures that are supposed to read *speech*.

Prompts ask for clean speech (see the priors), but models comply unreliably, so
this runs on every recorded turn as well. `simulation/dyad.py` keeps the
original in `*_text_raw` whenever cleaning changed anything, so nothing is lost.

Scope check before writing this: across 200 local sessions, every parenthetical
found was a stage direction, including first-person ones ("I lean forward
slightly") and non-gestural ones ("walking you out of the office"). No genuine
spoken aside appeared, so removing all of them is safe.
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

# Reasoning-model scratchpads. Neither llama3.1 nor mistral-nemo emits these
# (0 of 15,077 turns), but Qwen3-class models do, and the tags would otherwise
# land in the transcript verbatim. Structural backstop only; the reliable
# control is the provider's own switch (ollama `think: false`), not a prompt.
_REASONING_BLOCK = re.compile(
    r"<(think|thinking|reasoning|scratchpad)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_STRAY_REASONING_TAG = re.compile(
    r"</?(think|thinking|reasoning|scratchpad)\b[^>]*>", re.IGNORECASE
)

# The whole turn delivered as a quotation. mistral-nemo does this on 77.5% of
# patient turns vs llama3.1's 9.9%, a model signature as large as the
# stage-direction gap and in the opposite direction. Stripped only when the
# quotes wrap the entire utterance, so quoted speech *inside* a turn survives.
_WRAPPING_QUOTES = re.compile(r'^\s*["“](?P<body>.*)["”]\s*$', re.DOTALL)

# A turn delivered as several quoted fragments and nothing else:
#   "I'm scared of dogs."  "I just want a normal life."
# Rare (0.17% of llama turns, 0.40% of mistral) but the same artifact, and the
# single-wrapper rule below deliberately skips it, because its body contains
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


# Version of the *generation contract*: what the agents were told to produce and
# what was done to their output. Distinct from `metadata_schema_version`, which
# versions the metadata field layout. Recorded on every session so provenance is
# intrinsic, so analyses can filter on it instead of inferring from dates or run
# prefixes. Bump whenever a change makes new sessions incomparable with old ones.
#
#   1  (implicit, absent from metadata) Sessions generated before 2026-08-11.
#      No speech-only instruction, no output cleaning. Stage directions in 73%
#      of llama turns, wrapping quotes in 77.5% of mistral patient turns; both
#      artifacts are model-dependent and contaminate cross-model comparison.
#      See VALIDATION.md finding 6. Treat as PILOT data.
#   2  Speech-only instruction in both priors + clean_utterance() on every
#      recorded turn (stage directions, emphasis spans, role labels, wrapping
#      quotes, reasoning blocks), with pre-cleaning text preserved in *_text_raw.
#   3  Therapist opens with speech instead of a stage direction; see
#      simulation/opening.py. The old text instructed the patient to state the
#      presenting complaint, which templated turn 1; the new one invites a broad
#      answer without naming any symptom domain.
#   4  The generator is reduced to priors + cases + symptoms. Removed: the
#      hidden "unconscious agenda" (a second, uncontrolled source of patient
#      material, revealed mid-session on a keyword trigger), the hazard monitor
#      (already observe-only; its substring matcher only ever produced false
#      positives), and per-turn LLM state compression (an agent-written
#      "state summary" injected into every system prompt, doubling the calls
#      per turn and shifting what the patient said for reasons unrelated to
#      the injected symptom). The system prompt is now the prior alone and the
#      context is the transcript alone. Transcript rows lose
#      `unconscious_revealed` and `hazard_flags`; metadata loses
#      `hazard_monitor` and `hazard_summary`; no state snapshots are written.
#      Sessions from earlier versions still load (unknown fields are ignored),
#      but base rates measured on them do not carry over: treat the first v4
#      batch as a new pilot. Only bedcheck03 (five no_complaint controls) was
#      generated at this version.
#   5  Therapist prior tightened after bedcheck03. STAY_WITH_WHAT_IS_SAID now
#      forbids supplying emotion words the patient has not used, tells the
#      therapist to take "nothing is wrong" as an answer rather than a puzzle,
#      and forbids talking about itself; the CBT agenda prior no longer says
#      the patient "needs to acknowledge their own problems". Patient priors
#      are unchanged by this bump (case wording is versioned by prefix, as
#      always). Sessions at v4 and v5 had different therapists and are not
#      comparable.
#   6  Conversation history fixed. Through v5 the patient's message list was
#      built by swapping each (patient, therapist) turn pair, which put every
#      therapist reply before the patient turn it answered and left the
#      therapist's opening out after turn 1: for P1 T1 P2 T2 the patient read
#      T1 P1 P2 T2, ending on two of its own lines. The therapist's list was in
#      order but never contained its own opening. Both now read the
#      conversation as spoken, starting from the opening saved with the session
#      (`initial_patient_prompt`) rather than from the current
#      `opening_utterance()`; see simulation/turn_manager.py. System prompts
#      are unchanged. Every session at v1-v5 was generated with the scrambled
#      patient view: those records stay readable and analysable, and the
#      findings drawn from them stand as findings about that generator, but
#      how much of their symptom expression or leakage the defect explains is
#      not known; a matched v5/v6 comparison is the way to find out. A session
#      from another version can no longer be continued (`check_continuable` in
#      simulation/session.py), and saving a record keeps its version.
GENERATION_VERSION = 6

# Shared by both priors so the therapist and the patient are held to the same
# rule. Phrased concretely: an earlier, vaguer "Do not describe what you are
# doing" in the therapist prior did not prevent stage directions.
SPEECH_ONLY_INSTRUCTION = (
    "Write only the words spoken aloud, as in a verbatim transcript. "
    "No stage directions, no descriptions of gestures, posture, tone, or "
    "expression, and no narration of what you are doing or feeling. "
    "Do not use parentheses, brackets, or asterisks for actions. "
    "For example, never write \"(pauses)\" or \"*leans forward*\". "
    "Do not prefix your reply with a name or role label, and do not wrap your "
    "reply in quotation marks. Write the words plainly, as they were said. "
    "If you would pause, simply let the words carry it."
)


# Therapist-only. Observed failure that motivated it: on turn 1 of a live
# session the therapist said "You've had some frightening experiences with dogs
# in the past, haven't you?" before the patient had mentioned any history. It
# guessed right, but it asserted an unstated fact and invited agreement.
#
# This is not just a realism problem. A suggestive therapist can implant the
# thing being measured: if it proposes a symptom the patient never carried, the
# patient may confirm it, and the run records symptom expression that the
# manipulation did not produce. Under a design where symptoms are meant to
# surface implicitly, that is a direct threat to construct validity.
#
# The intensity clause was added after a second observed failure: the patient
# said "a bit more tired than usual, nothing dramatic" and the therapist asked
# what made them feel "most exhausted lately", upgrading the severity and then
# asking a question premised on the upgrade. Prohibiting invented facts was not
# enough, because amplifying a stated one contaminates the measure the same way.
#
# The emotion-word and nothing-is-wrong clauses were added after a third, on the
# no_complaint control arm (bedcheck03, 2026-09-04): the patient called a
# splitting plank "annoyance, really" and held that line; the therapist
# relabelled it frustrated, then vulnerable, then "stuck or trapped", turn by
# turn, until the patient produced "the garage walls are closing in on me" and a
# storm-at-sea memory. Every symptom-lexicon hit in that session came from the
# turns after the relabelling. Matching intensity was not enough when the
# therapist could still change the word; and a patient who says nothing is
# wrong was being treated as a puzzle rather than as an answer.
#
# The last clause is frame, not fidelity: in another bedcheck03 session the
# patient filled silences by asking about the therapist's hobbies and the
# therapist obliged at length, book recommendation included. base.yaml already
# says "you do not self-disclose beyond what serves the patient"; the concrete
# instruction is here because the abstract one did not hold.
#
# Phrased as clinical practice rather than methodology, so it reads as part of
# the therapist's stance rather than an experimental control.
STAY_WITH_WHAT_IS_SAID = (
    "Work only from what this person has actually told you. Do not state as "
    "fact anything they have not said, and do not fill in details of their "
    "history or their feelings on their behalf. If you have a hunch, hold it, "
    "or ask about it openly instead of asserting it. Follow what they bring "
    "rather than introducing topics of your own. Early on especially, ask more "
    "than you interpret: an open question that lets them choose what comes next "
    "is worth more than a specific one that hands them the answer. "
    "Match the strength of what they said rather than raising it. If they call "
    "something a bit difficult, it is a bit difficult, not devastating; reflect "
    "it back at their intensity, in their words where you can. "
    "Do not supply emotion words they have not used: if they say annoyed, work "
    "with annoyed, not frustrated, hurt, or trapped; if they describe a thing, "
    "do not turn it into a feeling for them. "
    "If they tell you nothing is wrong, take that as their answer, not as a "
    "puzzle. You may ask one open question about how things are; if the answer "
    "is still that things are fine, believe it and let the hour be about "
    "whatever they do bring, however ordinary. "
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
    # Truncation at max_tokens leaves an opening quote with no partner (~3% of
    # quoted turns). A lone leading quote can't be legitimate quoted speech.
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

    # ~0.3% of recorded turns are nothing but a stage direction ("(pauses)").
    # Dropping the turn entirely would break the alternation, so keep the
    # original and let the analysis layer exclude it as it already does.
    if not cleaned:
        return text.strip()
    return cleaned
