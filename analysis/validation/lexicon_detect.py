"""Naive symptom detection: did the patient actually say it?

The honest floor for the question "would a reader find this symptom in the
transcript". Deliberately transparent, no embeddings and no trained model, so
every firing can be traced to the exact words that caused it.

A plain keyword search is not good enough, and the corpus shows why. Sampling
sentence-level hits from the existing lexicon:

  matched "down"      in "(looking down, fidgeting with hands)"    stage direction
  matched "enjoy"     in "learn how to enjoy the journey"          an aspiration
  matched "nightmare" in "stuck in this never-ending nightmare"    a metaphor
  matched "dying"     in "I could be dying on the inside"          a figure of speech

A quarter of sentence-level hits sit in a sentence containing a negation.
Vocabulary presence answers "was this word used", not "does this person have
this symptom", and the two diverge most where the base rate is already high - a
broad lexicon like `depressed_mood` fires in most sessions where nothing at all
was injected, at which point the column carries no information.

Three rules narrow that gap while staying naive:

1. **Negation scoping.** A cue ("not", "never", "don't") suppresses matches
   later in the same clause. Clause-level rather than sentence-level, so
   "I can't sleep, but I still enjoy my hobbies" negates sleep and not enjoy.
   This removes false firings from control and injected sessions at the same
   rate, so it leaves the lift over control roughly unchanged while making each
   individual firing trustworthy. Kept for auditability, not for sensitivity.
2. **Recurrence is measured but NOT gated.** `min_turns` defaults to 1, i.e.
   off. The intuition (metaphors are one-off, real complaints recur) does not
   survive contact with a 10-turn session, where a patient may say the thing
   once and never return to it. Requiring a second mention discards a genuine
   single complaint - "And sleep? That's been almost impossible lately" - faster
   than it discards a metaphor, and the lift over control falls as the gate
   tightens. The per-turn counts and evidence are still returned, so recurrence
   can be inspected without filtering on it.
3. **Patient speech only**, with one exception: if the therapist raised the
   topic and the patient answered referentially ("yes", "sometimes"), that
   counts, because the patient did confirm it. The confirmation is recorded
   with its own source so it can be excluded or inspected separately.

What it still cannot do is understand meaning. Metaphor and aspiration that
survive the recurrence filter will pass. That is the line where "naive" stops,
and it is the argument for reading this alongside a trained rater rather than
instead of one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from analysis.validation.symptom_lexicon import SOMATIC_AMBIGUOUS

# Cues that suppress what follows them in the same clause.
_NEGATION = re.compile(
    r"\b(?:not|no|never|none|nothing|neither|nor|without|"
    r"don'?t|doesn'?t|didn'?t|isn'?t|aren'?t|wasn'?t|weren'?t|"
    r"can'?t|cannot|couldn'?t|won'?t|wouldn'?t|shouldn'?t|haven'?t|hasn'?t|hadn'?t|"
    r"hardly|barely|scarcely|rarely|seldom)\b",
    re.IGNORECASE,
)

# "not only" and "no doubt" are not denials; they would wrongly suppress.
_PSEUDO_NEGATION = re.compile(
    r"\b(?:not only|not just|no doubt|no wonder|not to mention)\b", re.IGNORECASE
)

# Clause boundaries. A negation does not reach past one of these.
_CLAUSE_SPLIT = re.compile(
    r"[,;:]|\b(?:but|however|although|though|except|whereas|yet|still|"
    r"while|and then|because|since)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# A reply that only makes sense against the question just asked. Same shape as
# the patterns in symptom_scoring, kept here so the naive path has no dependency
# on the scoring pipeline's thresholds.
# A reply that only makes sense against the question just asked, SPLIT BY
# POLARITY. The single undifferentiated class was a polarity bug: "Have you had
# trouble sleeping?" / "No, never." was recorded as a positive confirmation,
# because any referential answer counted once the therapist's turn matched the
# pattern. This was not hypothetical: it is the most common shape of reply a
# reflective therapist elicits, so the bug inflated exactly the cells where
# the therapist asked most.
#
# Order matters. Negative is tested first, because "not really" and "no" would
# otherwise be reachable by looser patterns.
_NEGATIVE_ANSWER = re.compile(
    r"^\s*(?:no|nope|nah|not really|not at all|not particularly|never|rarely|"
    r"hardly|scarcely|seldom|none|neither)\b",
    re.IGNORECASE,
)

# Hedged or non-committal: the patient has not endorsed the symptom, but has not
# denied it either. Recorded as rejected rather than counted, because "maybe" is
# not evidence.
_UNCERTAIN_ANSWER = re.compile(
    r"^\s*(?:maybe|perhaps|possibly|i don'?t know|i'?m not sure|not sure|"
    r"hard to say|difficult to say|it depends|sometimes i wonder|"
    # Lean affirmative but are not unambiguous; retained as rejected evidence.
    r"kind of|sort of|i guess|i suppose)\b",
    re.IGNORECASE,
)

# Endorsements, and ONLY unambiguous ones. "Sometimes" and "a bit" stay, because
# on a PHQ-9 frequency scale they are endorsements ("several days") rather than
# hedges.
#
# Deliberately absent: "i guess", "i suppose", "it is", "i do", "i have",
# "i am", "i'm". These turn therapist vocabulary into patient detections,
# because this mechanism infers WHICH symptom was confirmed from what the
# therapist just asked. Each of these was an observed failure:
#
#   "I guess I haven't really noticed any..."  -> counted as confirming
#                                                 depressed mood; it is a denial
#   "I'm sorry to hear that things have..."    -> counted as a confirmation; it
#                                                 is role-confused therapist
#                                                 speech, not a patient reply
#   "I guess I just feel guilty..."            -> counted as confirming low
#                                                 energy, a symptom the patient
#                                                 never mentioned
#
# "kind of" and "sort of" moved to `_UNCERTAIN_ANSWER`: they lean affirmative
# but are not unambiguous, and uncertain replies are retained as rejected
# evidence rather than discarded, so nothing is lost by classifying them there.
_AFFIRMATIVE_ANSWER = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|mhm+|mm+|uh[- ]huh|definitely|absolutely|"
    r"certainly|exactly|constantly|always|often|a lot|all the time|most days|"
    r"pretty much|sometimes|a bit|a little|that'?s right|that'?s it)\b",
    re.IGNORECASE,
)

# An endorsement only counts as a *referential* confirmation when the reply is
# short enough to be one. This is not a heuristic patch on top of the token list
# - it is what the mechanism is for. Confirmation exists to catch replies that
# carry no symptom vocabulary of their own ("Yes." / "Sometimes."). A long reply
# that does contain symptom words is already caught by direct detection, so
# nothing is lost by refusing to guess at the topic of a paragraph.
_MAX_CONFIRMATION_WORDS = 10


def classify_reply(patient_text: str) -> str:
    """"affirmative" | "negative" | "uncertain" | "not_referential".

    Kept as four distinct classes, not collapsed to a boolean, so the evidence
    trail records *why* a reply was rejected. A denial counted as a confirmation
    reverses polarity, which is the worst failure available here.
    """
    t = patient_text.strip()
    if _NEGATIVE_ANSWER.match(t):
        return "negative"
    if _UNCERTAIN_ANSWER.match(t):
        return "uncertain"
    if _AFFIRMATIVE_ANSWER.match(t):
        if len(t.split()) > _MAX_CONFIRMATION_WORDS:
            # Opens like an endorsement but continues into content we would be
            # attributing to the therapist's topic rather than the patient's.
            return "not_referential"
        return "affirmative"
    return "not_referential"


@dataclass
class Hit:
    """One firing, with everything needed to audit it."""

    turn: int
    matched: str
    evidence: str
    source: str  # "patient" | "patient_confirms_therapist"


@dataclass
class Detection:
    symptom: str
    hits: list[Hit] = field(default_factory=list)
    negated_hits: list[Hit] = field(default_factory=list)

    @property
    def turns(self) -> set[int]:
        return {h.turn for h in self.hits}

    def detected(self, min_turns: int = 1) -> bool:
        """Whether the symptom counts as present.

        `min_turns` defaults to 1 (no recurrence gate) for the reason given in
        the module docstring: gating on recurrence costs more true positives
        than it removes false ones at this session length.
        """
        return len(self.turns) >= min_turns


def _clause_of(sentence: str, position: int) -> str:
    """The clause containing `position`, so negation cannot leak across one."""
    start = 0
    for boundary in _CLAUSE_SPLIT.finditer(sentence):
        if boundary.end() <= position:
            start = boundary.end()
        else:
            break
    return sentence[start:position]


# Someone else's symptom. Found by running the 360 a-priori hard negatives
# against the lexicon: `third_party` was the worst-performing category at 28%.
# The cause is structural rather than a missing word. `_DECREMENT_FRAME` in
# symptom_lexicon.py encodes POLARITY ("no", "hasn't", "lost") but nothing about
# OWNERSHIP, so "my brother hasn't enjoyed anything since his divorce" fires
# `lack_of_pleasure` exactly as "I haven't enjoyed anything" does, whatever the
# term list contains.
#
# Handled here rather than in the patterns because it is the same shape as
# negation: a clause-level property of the context, not a property of the term.
_THIRD_PARTY_SUBJECT = re.compile(
    r"\b(?:"
    r"my (?:mum|mother|dad|father|brother|sister|son|daughter|wife|husband|partner|"
    r"friend|colleague|boss|flatmate|neighbour|neighbor|uncle|aunt|gran|grandad|"
    r"grandmother|grandfather|kids?|child|children|family|doctor|therapist)"
    r"|his|her|hers|their|theirs|he|she|they"
    r"|mum|mother|dad|father|everyone else|other people|people like"
    r")\b",
    re.IGNORECASE,
)

# Bare first person only. "my" is excluded deliberately: it appears inside every
# "my brother" style third-party phrase, so counting it as first person would
# defeat the check it is meant to support.
_FIRST_PERSON = re.compile(r"\b(?:i|i'?m|i'?ve|i'?d|i'?ll|me|myself)\b", re.IGNORECASE)

# "told me", "says to me": here "me" is the LISTENER, not the person with the
# symptom. Without this, "my colleague told me everything feels grey to her"
# counts as first-person self-report because of that "me". Reclaiming
# constructions ("people keep telling me I'm slow") are unaffected: they are
# carried by the later "I", not by the "me".
_ME_AS_LISTENER = re.compile(
    r"\b(?:told|tells?|telling|says?|said|asked?|asking|mentioned|warned)\s+(?:to\s+)?me\b",
    re.IGNORECASE,
)


def has_third_party_subject(sentence: str, position: int) -> bool:
    """Is the match at `position` describing someone other than the speaker?

    True when the clause names a third party and contains no bare first-person
    pronoun. "My mum says I don't enjoy things" keeps firing, because the "I"
    shows the symptom is still being claimed by the speaker.
    """
    clause = _ME_AS_LISTENER.sub(" ", _clause_of(sentence, position))
    return bool(_THIRD_PARTY_SUBJECT.search(clause)) and not _FIRST_PERSON.search(clause)


def is_negated(sentence: str, position: int) -> bool:
    """Is a match at `position` inside the scope of a negation cue?"""
    clause = _clause_of(sentence, position)
    if _PSEUDO_NEGATION.search(clause):
        clause = _PSEUDO_NEGATION.sub(" ", clause)
    return bool(_NEGATION.search(clause))


# Autonomic arousal: breathlessness, cardiac and muscular activation. A mood
# metaphor sitting in one of these sentences is describing a panic body, not a
# mood. Scoped to the SENTENCE rather than the clause, because the disambiguator
# is usually a trailing participial phrase - "..., making it hard to breathe" -
# which `_clause_of` would cut away.
_PANIC_CONTEXT = re.compile(
    r"\b(?:"
    r"breathe|breathing|breath|gasp\w*|suffocat\w*|hyperventilat\w*|"
    r"heart (?:is |was )?(?:racing|race|races|pounding|pound|hammering|thumping)|"
    r"pounding|racing|palpitations|"
    r"tighten\w*|churn\w*|sweat\w*|shak(?:e|ing|y)|trembl\w*|"
    r"panic\w*|dizzy|nausea\w*|sick to my stomach"
    r")\b",
    re.IGNORECASE,
)

# The metaphor being withdrawn rather than reported: "the weight is lifting",
# "a weight off my shoulders". Relief reads as the symptom without this.
_RELIEF_CONTEXT = re.compile(
    r"\b(?:lift(?:ing|ed|s)?|off my|less heavy|not as heavy|gone now)\b",
    re.IGNORECASE,
)

# Bound to a trigger or to contemplating one, rather than reported as a standing
# state: "...when I see one", "the thought of...", "when I think about...".
_TRIGGER_BOUND = re.compile(
    r"\b(?:when I (?:see|think|imagine)|the thought of|thinking about|"
    r"if I (?:see|had to)|whenever (?:one|a dog|I see))\b",
    re.IGNORECASE,
)


#: How far before the metaphor a modifier may sit and still be part of it.
#: "there's this heavy weight on my chest" fires on `heavy` alone, several
#: characters before "weight", so keying the guard on the matched STRING missed
#: it - the adjective and the metaphor are one construction.
_MODIFIER_REACH = 20


def is_somatic_not_mood(sentence: str, position: int, matched: str) -> bool:
    """Is this shared depression/panic metaphor describing a panic body?

    Applies ONLY where the match is part of a `SOMATIC_AMBIGUOUS` construction -
    the handful of metaphors depression and panic genuinely share. The rest of
    the lexicon is untouched, because for those terms the word already carries
    its own sense.

    Scoped by POSITION rather than by the matched text, because the same
    construction can fire on its adjective: "there's this heavy weight on my
    chest, making it hard to breathe" matches `heavy` (a `depressed_mood`
    symptom term) some way to the left of the metaphor proper.

    Sentence-scoped, not clause-scoped: the disambiguating cue is usually a
    trailing participial phrase ("..., making it hard to breathe") that
    `_clause_of` would cut away.
    """
    spans = [m.span() for m in SOMATIC_AMBIGUOUS.finditer(sentence)]
    if not spans:
        return False
    end = position + len(matched)
    in_construction = SOMATIC_AMBIGUOUS.search(matched) or any(
        start - _MODIFIER_REACH <= position and end <= stop for start, stop in spans
    )
    if not in_construction:
        return False
    return bool(
        _PANIC_CONTEXT.search(sentence)
        or _RELIEF_CONTEXT.search(sentence)
        or _TRIGGER_BOUND.search(sentence)
    )


def _scan_text(text: str, compiled: dict, turn: int, results: dict, source: str) -> set[str]:
    """Record every non-negated match in `text`. Returns the symptoms that fired."""
    fired: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(text):
        for symptom, pattern in compiled.items():
            # EVERY match, not just the first. `search()` stopped at the leftmost
            # match, so a rejected one hid any valid assertion after it:
            # "I don't have nightmares, but I lie awake for hours" scored as
            # not-detected, because the negated "nightmares" ended the scan. The
            # third-party guard made this worse by adding a second way for a
            # first match to be rejected.
            for match in pattern.finditer(sentence):
                hit = Hit(
                    turn=turn,
                    matched=match.group(0),
                    evidence=sentence.strip()[:200],
                    source=source,
                )
                # All three suppressions land in `negated_hits`, the bucket for
                # "matched but rejected", so the evidence CSV can show why
                # something did not count.
                if (
                    is_negated(sentence, match.start())
                    or has_third_party_subject(sentence, match.start())
                    or is_somatic_not_mood(sentence, match.start(), match.group(0))
                ):
                    results[symptom].negated_hits.append(hit)
                else:
                    results[symptom].hits.append(hit)
                    fired.add(symptom)
    return fired


def detect_therapist(pairs, patterns: dict[str, str]) -> dict[str, Detection]:
    """Does the THERAPIST use this symptom's vocabulary?

    Same lexicon, same negation scoping, other speaker. A different question
    from patient expression: whether the injected symptom propagates into the
    clinician's speech.

    Caveat this cannot resolve on its own: a therapist saying "sleep" may be
    raising the topic or reflecting back what the patient just said. This counts
    both. Reading it against the `no_symptoms` control is what separates uptake
    from a therapist who asks about sleep in every session regardless.
    """
    compiled = {s: re.compile(p, re.IGNORECASE) for s, p in patterns.items()}
    results = {s: Detection(symptom=s) for s in patterns}
    for pair in pairs:
        if not pair.valid:
            continue
        _scan_text(pair.therapist_text or "", compiled, pair.turn_index, results, "therapist")
    return results


def detect_session(
    pairs,
    patterns: dict[str, str],
    *,
    count_confirmations: bool = True,
) -> dict[str, Detection]:
    """Scan one session's turn pairs for every symptom in `patterns`.

    `pairs` are `symptom_scoring.transcript_parser.TurnPair` objects, which
    already pair each patient turn with the therapist turn that preceded it and
    flag echo artifacts. Invalid pairs are skipped.
    """
    compiled = {s: re.compile(p, re.IGNORECASE) for s, p in patterns.items()}
    results = {s: Detection(symptom=s) for s in patterns}

    for pair in pairs:
        if not pair.valid:
            continue
        patient = pair.patient_text or ""
        therapist = pair.therapist_text or ""
        reply = classify_reply(patient)

        fired_symptoms = _scan_text(patient, compiled, pair.turn_index, results, "patient")

        for symptom, pattern in compiled.items():
            # The patient never used the vocabulary, but the therapist raised the
            # topic and the patient answered referentially. ONLY an affirmative
            # reply counts. A negative or hedged reply is recorded as rejected
            # evidence, so it stays auditable instead of silently reversing
            # polarity, which is what the old undifferentiated class did.
            if symptom in fired_symptoms or not count_confirmations:
                continue
            if reply == "not_referential":
                continue
            question = pattern.search(therapist)
            if not question or is_negated(therapist, question.start()):
                continue
            hit = Hit(
                turn=pair.turn_index,
                matched=question.group(0),
                evidence=f"[{reply}] T: {therapist.strip()[:100]} | P: {patient.strip()[:70]}",
                source="patient_confirms_therapist",
            )
            if reply == "affirmative":
                results[symptom].hits.append(hit)
            else:
                results[symptom].negated_hits.append(hit)
    return results
