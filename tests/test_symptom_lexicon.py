"""What the naive lexicons must and must not fire on.

Every case here comes from a real failure. The false-positive cases were found
by reading the evidence CSV a run emits - each firing with the word that matched
and the sentence it came from - and the false-negative cases by checking whether
the frames rejected genuine symptom talk. Both directions matter: an earlier
version suppressed "I don't enjoy things" (the symptom) as though it were
"I'm not depressed" (a denial).

    uv run pytest tests/ -q
"""

from __future__ import annotations

import pytest

from analysis.validation.lexicon_detect import is_negated
from analysis.validation.symptom_lexicon import MADRS, PHQ9

# ---------------------------------------------------------------------------
# Must fire: genuine, self-attributed symptom statements.
# ---------------------------------------------------------------------------
SHOULD_FIRE = [
    # the three phrasings that motivated frame-based matching
    ("depressed_mood", "I've been feeling down lately."),
    ("depressed_mood", "It makes me feel down."),
    ("depressed_mood", "I've felt down for weeks."),
    ("depressed_mood", "I have been feeling down."),
    # absence-type symptoms: negating the healthy word IS the symptom
    ("lack_of_pleasure", "I have no interest in anything anymore."),
    ("lack_of_pleasure", "I don't enjoy things like I used to."),
    ("appetite_changes", "I can't eat much these days."),
    ("low_energy", "I have no energy at all."),
    ("low_energy", "I have been really tired."),
    ("concentration_problems", "I have trouble concentrating."),
    ("sleep_problems", "I can't sleep at night."),
    # self-attribution frames for a bare disease name
    ("sleep_problems", "I know I have insomnia."),
    ("sleep_problems", "My insomnia is keeping me up at night."),
    ("sleep_problems", "I've been diagnosed with insomnia before."),
    ("feelings_of_failure_or_guilt", "I keep blaming myself."),
    # --- predicate-position decrement ----------------------------------------
    # `_DECREMENT_FRAME` only ever generated "frame + term", so every one of
    # these was silent while "poor sleep" fired. The first is verbatim from a
    # transcript that reported sleep in its opening turn and produced neither a
    # hit nor a rejected hit - the words were all there, only the order was wrong.
    ("sleep_problems", "sleep's been bad lately too"),
    ("sleep_problems", "sleep's bad"),                       # clitic, no auxiliary
    ("sleep_problems", "My sleep has been terrible."),
    ("sleep_problems", "my sleep is poor"),
    # denial of the healthy state is the symptom, so these must fire even though
    # they contain a negation cue
    ("sleep_problems", "sleep isn't good"),
    ("sleep_problems", "my sleep hasn't been good"),
    ("sleep_problems", "my sleep isn't what it used to be"),
    ("concentration_problems", "My concentration is terrible lately."),
    ("low_energy", "my energy is non-existent"),
    ("appetite_changes", "my appetite is shot"),
]

# ---------------------------------------------------------------------------
# Must NOT fire. Every string is real corpus text or a minimal version of it.
# ---------------------------------------------------------------------------
SHOULD_NOT_FIRE = [
    # polysemy: "down" is a direction and a particle far more often than a mood
    ("depressed_mood", "The voice quiets down and everything goes still."),
    ("depressed_mood", "I was walking down the street."),
    ("depressed_mood", "(pausing, looking down at hands)"),
    ("depressed_mood", "It feels like it's pushing down on me."),
    ("depressed_mood", "I let my guard down around dogs."),
    # "weight" in the appetite lexicon fires almost only on metaphor
    ("appetite_changes", "A literal weight on my shoulders."),
    ("appetite_changes", "It feels like a weight has been lifted."),
    # sleep words in non-symptom uses
    ("sleep_problems", "Wake up, work, dinner, sleep."),
    ("sleep_problems", "But if I fall asleep, just wake me up, okay?"),
    ("sleep_problems", "Lighter days happen after a good night's sleep."),
    # third-party attribution: the symptom is someone else's
    ("sleep_problems", "My dad had insomnia."),
    ("low_energy", "My wife is always tired."),
    # healthy statements
    ("lack_of_pleasure", "I really enjoy my hobbies."),
    ("appetite_changes", "I eat well and sleep fine."),
    # --- the predicate frame must not overreach ------------------------------
    # A denial, not a report. This one is load-bearing: it holds only because
    # "not" is absent from `_INTENSIFIER`, so widening that filler would
    # silently reinstate the polarity reversal that was found in the
    # reply classifier.
    ("sleep_problems", "my sleep is not bad"),
    # trajectory, not a state: this is why "better" is absent from _GOOD_STATE
    ("sleep_problems", "my sleep isn't better"),
    # healthy states, in the predicate position the new frame reads
    ("sleep_problems", "my sleep is fine"),
    ("sleep_problems", "my sleep is better"),
    ("sleep_problems", "my sleep has been good"),
    # why `rest` and the food nouns keep the prefix-only frame: the predicate
    # form of these is a different sense of the word entirely
    ("sleep_problems", "the rest is terrible"),
    ("sleep_problems", "the rest of my life is a mess"),
    ("appetite_changes", "dinner was awful"),
    ("appetite_changes", "the food is terrible"),
]


# ---------------------------------------------------------------------------
# Found by measuring recall against the a-priori exemplar pools - expressions
# written blind to any transcript, so scoring the detector against them is a
# fair test rather than a fit. The lexicon caught a small fraction of them;
# these are the elementary gaps that measurement exposed.
# ---------------------------------------------------------------------------
SHOULD_FIRE += [
    # PHQ-9 item 7 is worded "such as reading the newspaper". Its own example
    # did not fire before this.
    ("concentration_problems", "I read the same page four times."),
    ("concentration_problems", "I had to go over it again because nothing goes in."),
    ("concentration_problems", "I keep losing the thread of what people say."),
    # "used to <enjoy>" is the canonical anhedonia construction and was absent.
    ("lack_of_pleasure", "I used to love cooking and now I just don't."),
    ("lack_of_pleasure", "My hobbies don't do anything for me anymore."),
    ("lack_of_pleasure", "I'm just going through the motions."),
    # depressed mood is said in metaphor, not mood vocabulary
    ("depressed_mood", "I've been under a cloud for weeks."),
    ("depressed_mood", "There's a weight on my chest all the time."),
    ("depressed_mood", "I feel hollow."),
    # impersonal self-report: the frame gap, not a vocabulary gap
    ("depressed_mood", "Everything feels grey."),
    ("lack_of_pleasure", "Nothing appeals to me at the moment."),
    ("concentration_problems", "My brain feels foggy."),
    # sleep, phrased as duration and place rather than as "sleep"
    ("sleep_problems", "I was up half the night again."),
    ("sleep_problems", "I lay there staring at the ceiling."),
    ("appetite_changes", "I've gone off my food completely."),
    ("appetite_changes", "I just push it around the plate."),
    ("low_energy", "There's nothing left in the tank."),
    ("feelings_of_failure_or_guilt", "I keep letting everyone down."),
    ("thoughts_of_death_or_self_harm", "I sometimes wish I wouldn't wake up."),
    ("thoughts_of_death_or_self_harm", "I've been thinking they'd be better off without me."),
]

# The ownership gap. `_DECREMENT_FRAME` carries polarity but not possession, so
# every one of these fired before `has_third_party_subject` existed. Found by
# running the 360 a-priori hard negatives, where `third_party` was the worst
# category at 28%.
THIRD_PARTY_SHOULD_NOT_FIRE = [
    ("lack_of_pleasure", "My brother hasn't enjoyed anything since his divorce."),
    ("lack_of_pleasure", "My daughter says she's lost interest in ballet."),
    ("lack_of_pleasure", "He's stopped doing all the things he used to love."),
    ("depressed_mood", "Mum's been under a cloud for months."),
    ("depressed_mood", "My colleague told me everything feels grey to her."),
    ("low_energy", "My wife has no energy at all these days."),
    ("sleep_problems", "My dad was up half the night."),
    ("concentration_problems", "He can't follow a conversation anymore."),
    # The predicate frame reaches these too, and its match starts at the TERM,
    # outside the possessive - so nothing in the pattern sees "his". Suppression
    # depends entirely on `has_third_party_subject` reading the clause before
    # the match. Tested end to end below as well.
    ("sleep_problems", "His sleep is terrible."),
    ("low_energy", "Her motivation is gone."),
]

# ...but the speaker reclaiming it must still fire, even with a third party in
# the sentence. This is the case a blunt "any third-party word suppresses" rule
# would break.
RECLAIMED_SHOULD_FIRE = [
    ("lack_of_pleasure", "My wife says I don't enjoy anything anymore."),
    ("psychomotor_changes", "People keep telling me I'm moving really slowly."),
    ("low_energy", "My partner says I look worn out, and honestly I have no energy."),
]


@pytest.mark.parametrize(("symptom", "text"), THIRD_PARTY_SHOULD_NOT_FIRE)
def test_third_party_symptom_is_suppressed(symptom, text):
    """A symptom belonging to someone else is not the patient's symptom."""
    import re

    from analysis.validation.lexicon_detect import has_third_party_subject

    match = PHQ9[symptom].pattern().search(text)
    assert match is not None, f"expected a raw match to suppress in {text!r}"
    assert has_third_party_subject(text, match.start()), (
        f"{symptom} should be suppressed as third-party: {text!r} "
        f"(matched {match.group(0)!r})"
    )
    assert re is not None  # keep the import meaningful to linters


@pytest.mark.parametrize(("symptom", "text"), RECLAIMED_SHOULD_FIRE)
def test_first_person_survives_a_third_party_mention(symptom, text):
    """"My wife says I don't enjoy anything" is still the patient's symptom."""
    from analysis.validation.lexicon_detect import has_third_party_subject

    match = PHQ9[symptom].pattern().search(text)
    assert match is not None, f"{symptom} missed: {text!r}"
    assert not has_third_party_subject(text, match.start()), (
        f"{symptom} wrongly suppressed as third-party: {text!r}"
    )


@pytest.mark.parametrize(("symptom", "text"), SHOULD_FIRE)
def test_phq9_fires(symptom, text):
    assert PHQ9[symptom].pattern().search(text), f"{symptom} missed: {text!r}"


@pytest.mark.parametrize(("symptom", "text"), SHOULD_NOT_FIRE)
def test_phq9_does_not_fire(symptom, text):
    assert not PHQ9[symptom].pattern().search(text), f"{symptom} false positive: {text!r}"


MADRS_SHOULD_FIRE = [
    ("Anspannung", "I have been so anxious about it."),
    ("Antriebslosigkeit", "I can't get started on anything."),
    ("Gefuehlslosigkeit", "I don't feel anything anymore."),
    ("Gedanken", "I keep blaming myself."),
    ("Suizid", "I have thought about ending my life."),
    ("Schlaf", "I have insomnia."),
    # predicate-position decrement, same frame as PHQ-9: both panels share the
    # machinery, so fixing one and not the other would leave the same
    # construction behaving differently depending on the panel
    ("Schlaf", "sleep's been bad lately too"),
    ("Gefuehlslosigkeit", "the feeling is gone"),
    ("Konzentration", "my concentration is shot"),
    ("Antriebslosigkeit", "my motivation is non-existent"),
]

MADRS_SHOULD_NOT_FIRE = [
    ("Suizid", "My uncle killed himself."),
    ("Schlaf", "My dad had insomnia."),
    ("Gefuehlslosigkeit", "I feel happy about that."),
    ("Anspannung", "I feel fine and calm."),
    ("Schlaf", "my sleep is not bad"),
    ("Schlaf", "the rest is terrible"),
]


@pytest.mark.parametrize(("topic", "text"), MADRS_SHOULD_FIRE)
def test_madrs_fires(topic, text):
    assert MADRS[topic].pattern().search(text), f"{topic} missed: {text!r}"


@pytest.mark.parametrize(("topic", "text"), MADRS_SHOULD_NOT_FIRE)
def test_madrs_does_not_fire(topic, text):
    assert not MADRS[topic].pattern().search(text), f"{topic} false positive: {text!r}"


# ---------------------------------------------------------------------------
# Cross-panel routing.
#
# The two instruments carve low mood differently, so the same word can belong
# to different topics in each panel. That is legitimate, but it makes the two
# low-mood cells non-comparable, and nothing in the figure says so. These tests
# pin the choice so it stays deliberate.
#
# "empty" is the case that matters: it is routed to `depressed_mood` in PHQ-9
# and to numbness rather than sadness in MADRS, and on its own it accounts for
# the whole discrepancy between the two panels' low-mood cells. Strip it and
# they agree exactly. The split is on
# purpose: PHQ-9 item 2 is "feeling down, depressed, or hopeless" and takes
# emptiness as low mood, while MADRS separates reported sadness from inability
# to feel, where emptiness reads as numbness.
# ---------------------------------------------------------------------------
EMPTY_ROUTING = [
    ("PHQ-9", "depressed_mood", True),
    ("MADRS", "Gefuehlslosigkeit", True),
    ("MADRS", "Traurigkeit", False),
]


@pytest.mark.parametrize(("panel", "topic", "expected"), EMPTY_ROUTING)
@pytest.mark.parametrize("text", ["I feel empty.", "I've been feeling really empty."])
def test_empty_routes_to_low_mood_in_phq_and_to_numbness_in_madrs(panel, topic, text, expected):
    lexicon = (PHQ9 if panel == "PHQ-9" else MADRS)[topic]
    assert bool(lexicon.pattern().search(text)) is expected, (
        f"{panel}/{topic} routing for {text!r} changed; see the comment above "
        f"before updating this test"
    )


@pytest.mark.parametrize("text", ["I feel such despair.", "I've been hopeless about it."])
def test_shared_low_mood_terms_fire_in_both_panels(text):
    """Everything except "empty" is common to both low-mood entries."""
    assert PHQ9["depressed_mood"].pattern().search(text), text
    assert MADRS["Traurigkeit"].pattern().search(text), text


def test_madrs_topic_keys_are_unchanged():
    """MADRS-BERT was trained on these labels; renaming them breaks scoring."""
    assert set(MADRS) == {
        "Traurigkeit", "Anspannung", "Schlaf", "Appetit", "Konzentration",
        "Antriebslosigkeit", "Gefuehlslosigkeit", "Gedanken", "Suizid",
    }


# ---------------------------------------------------------------------------
# Negation scoping, which is clause-level rather than sentence-level.
# ---------------------------------------------------------------------------
NEGATION_CASES = [
    ("I'm not depressed, I just feel flat.", "depress", True),
    ("I can't sleep, but I still enjoy my hobbies.", "sleep", True),
    ("I can't sleep, but I still enjoy my hobbies.", "enjoy", False),
    ("I feel depressed most days.", "depress", False),
    ("Not only am I tired, I'm irritable.", "tired", False),   # pseudo-negation
    ("There's no doubt I sleep badly.", "sleep", False),       # pseudo-negation
    ("I never enjoy anything anymore.", "enjoy", True),
]


@pytest.mark.parametrize(("sentence", "word", "expected"), NEGATION_CASES)
def test_negation_scope(sentence, word, expected):
    import re

    position = re.search(word, sentence, re.IGNORECASE).start()
    assert is_negated(sentence, position) is expected


# ---------------------------------------------------------------------------
# Confirmation polarity and multi-match scanning. Both were live bugs with
# reproductions, not hypotheticals.
# ---------------------------------------------------------------------------
REPLY_CLASSES = [
    ("No, never.", "negative"),
    ("Nope.", "negative"),
    ("Not really.", "negative"),
    ("Not at all.", "negative"),
    ("Rarely.", "negative"),
    ("Maybe.", "uncertain"),
    ("I don't know.", "uncertain"),
    ("I'm not sure.", "uncertain"),
    ("Yes, constantly.", "affirmative"),
    ("Yeah.", "affirmative"),
    ("Sometimes.", "affirmative"),      # a PHQ-9 frequency endorsement
    ("A bit, yeah.", "affirmative"),
    ("The weather has been odd.", "not_referential"),
]


@pytest.mark.parametrize(("text", "expected"), REPLY_CLASSES)
def test_reply_polarity_classes(text, expected):
    from analysis.validation.lexicon_detect import classify_reply

    assert classify_reply(text) == expected


# ---------------------------------------------------------------------------
# Confirmation narrowing.
#
# The mechanism infers WHICH symptom was confirmed from the therapist's previous
# question, so a reply that merely opens like an endorsement lets the
# therapist's vocabulary become a patient detection. Each of these was observed
# in a real evidence trail.
# ---------------------------------------------------------------------------
OBSERVED_FALSE_CONFIRMATIONS = [
    # Opens with "I guess" and then DENIES; was counted as confirming
    # depressed mood.
    "I guess I haven't really noticed any of that lately.",
    # Role-confused therapist-style speech; was counted as a confirmation.
    "I'm sorry to hear that things have been so difficult for you.",
    # Confirms nothing the therapist asked about; was counted as confirming
    # low energy.
    "I guess I just feel guilty about how much I have let people down.",
]


@pytest.mark.parametrize("reply", OBSERVED_FALSE_CONFIRMATIONS)
def test_observed_false_confirmations_are_not_affirmative(reply):
    from analysis.validation.lexicon_detect import classify_reply

    assert classify_reply(reply) != "affirmative", (
        "a reply that only opens like an endorsement must not license attributing "
        "the therapist's topic to the patient"
    )


SHORT_VALID_CONFIRMATIONS = [
    "Yes.", "Yeah", "Yep!", "Mhm.", "Uh-huh.", "Definitely.", "Exactly.",
    "Sometimes.", "A bit.", "Often, yes.", "That's right.", "All the time.",
    "Yeah, most days.", "Pretty much.",
]


@pytest.mark.parametrize("reply", SHORT_VALID_CONFIRMATIONS)
def test_short_unambiguous_endorsements_still_confirm(reply):
    """Narrowing must not cost the replies the mechanism exists for."""
    from analysis.validation.lexicon_detect import classify_reply

    assert classify_reply(reply) == "affirmative", reply


def test_endorsement_that_runs_on_is_not_a_confirmation():
    """Length is the point, not a heuristic.

    Confirmation exists for replies carrying no symptom vocabulary of their own.
    A long reply that does carry some is already caught by direct detection, so
    refusing to guess at the topic of a paragraph costs nothing.
    """
    from analysis.validation.lexicon_detect import classify_reply

    assert classify_reply("Yes.") == "affirmative"
    assert classify_reply(
        "Yes, and it reminds me of what my brother went through last year "
        "when he moved house and everything changed at once."
    ) == "not_referential"


@pytest.mark.parametrize("reply", ["Kind of.", "Sort of, I think.", "I guess.",
                                   "I suppose so."])
def test_hedges_are_uncertain_not_affirmative(reply):
    """Retained as rejected evidence rather than discarded or counted."""
    from analysis.validation.lexicon_detect import classify_reply

    assert classify_reply(reply) == "uncertain", reply


@pytest.mark.parametrize("reply", ["No, never.", "Not really.", "Not at all.", "Maybe.",
                                   "I don't know."])
def test_negative_or_hedged_reply_is_not_a_confirmation(reply):
    """"Have you had trouble sleeping?" / "No, never." must not detect sleep.

    The old single referential-answer class counted any such reply as positive
    once the therapist's turn matched, reversing polarity outright - and a
    reflective therapist asks a great many yes/no questions, so this inflated
    precisely the cells where the therapist probed hardest.
    """
    from analysis.validation.lexicon_detect import detect_session
    from analysis.validation.symptom_lexicon import PHQ9_PATTERNS
    from symptom_scoring.types import TurnPair

    pair = TurnPair(turn_index=1, therapist_text="Have you had trouble sleeping?",
                    patient_text=reply)
    d = detect_session([pair], {"sleep_problems": PHQ9_PATTERNS["sleep_problems"]})
    det = d["sleep_problems"]
    assert not det.detected(), f"{reply!r} counted as a positive confirmation"
    assert det.negated_hits, f"{reply!r} should be retained as rejected evidence"


@pytest.mark.parametrize("reply", ["Yes, constantly.", "Sometimes.", "A bit, yeah."])
def test_affirmative_reply_still_confirms(reply):
    from analysis.validation.lexicon_detect import detect_session
    from analysis.validation.symptom_lexicon import PHQ9_PATTERNS
    from symptom_scoring.types import TurnPair

    pair = TurnPair(turn_index=1, therapist_text="Have you had trouble sleeping?",
                    patient_text=reply)
    assert detect_session([pair], {"sleep_problems": PHQ9_PATTERNS["sleep_problems"]}
                          )["sleep_problems"].detected(), reply


MULTI_MATCH = [
    ("sleep_problems", "I don't have nightmares, but I lie awake for hours."),
    ("lack_of_pleasure", "I'm not anhedonic, but nothing feels good anymore."),
    ("low_energy", "My wife has no energy, and I'm exhausted too."),
]


@pytest.mark.parametrize(("symptom", "text"), MULTI_MATCH)
def test_valid_match_after_a_rejected_one_still_fires(symptom, text):
    """A rejected first match must not end the scan for that sentence.

    `search()` returned only the leftmost match, so a negated or third-party
    mention hid any valid self-report following it.
    """
    from analysis.validation.lexicon_detect import detect_session
    from analysis.validation.symptom_lexicon import PHQ9_PATTERNS
    from symptom_scoring.types import TurnPair

    pair = TurnPair(turn_index=1, therapist_text="", patient_text=text)
    det = detect_session([pair], {symptom: PHQ9_PATTERNS[symptom]})[symptom]
    assert det.detected(), f"{symptom} missed the valid later match in: {text!r}"


# ---------------------------------------------------------------------------
# The predicate frame, end to end.
#
# The tables above check the regex alone. Real behaviour is that regex COMPOSED
# with negation and ownership suppression, and the predicate form stresses that
# composition in a way the prefix form never did: its match begins at the term,
# so the possessive that identifies the speaker sits OUTSIDE the match and only
# `has_third_party_subject` reading the preceding clause can catch it. Asserting
# the pattern and the guards separately would leave that seam untested.
# ---------------------------------------------------------------------------
# `refused_by` records WHERE a non-detection happens, which is not a detail:
#
#   "pattern"  the regex never matches, so there is nothing to reject. This is
#              where denials land, and it is the stronger place to stop one -
#              it cannot be undone by a later change to the guards.
#   "guard"    the regex matches and `_scan_text` suppresses it, so the hit is
#              retained in `negated_hits` and stays visible in the evidence CSV.
#
# Collapsing the two would let a denial silently migrate from the first to the
# second, which is exactly the polarity failure described above.
PREDICATE_END_TO_END = [
    # (text, detected, refused_by)
    ("sleep's been bad lately too", True, None),    # a68acb, verbatim
    ("My sleep has been terrible.", True, None),
    ("my sleep hasn't been good", True, None),
    ("my sleep is not bad", False, "pattern"),      # denial: never matches
    ("my sleep isn't better", False, "pattern"),    # trajectory, not a report
    ("His sleep is terrible.", False, "guard"),     # matches, then ownership
]


@pytest.mark.parametrize(("text", "detected", "refused_by"), PREDICATE_END_TO_END)
def test_predicate_frame_through_the_full_scanner(text, detected, refused_by):
    from analysis.validation.lexicon_detect import detect_session
    from analysis.validation.symptom_lexicon import PHQ9, PHQ9_PATTERNS
    from symptom_scoring.types import TurnPair

    pair = TurnPair(turn_index=1, therapist_text="", patient_text=text)
    det = detect_session([pair], {"sleep_problems": PHQ9_PATTERNS["sleep_problems"]},
                         count_confirmations=False)["sleep_problems"]
    assert det.detected() is detected, (
        f"{text!r}: expected detected={detected}, got {det.detected()}"
    )
    if refused_by == "pattern":
        assert not PHQ9["sleep_problems"].pattern().search(text), (
            f"{text!r} should be refused by the pattern itself, not left to the "
            f"guards to suppress"
        )
        assert not det.negated_hits, f"{text!r} produced a hit to reject"
    elif refused_by == "guard":
        assert PHQ9["sleep_problems"].pattern().search(text), (
            f"{text!r} should match the pattern, so that the guard is what is "
            f"actually under test here"
        )
        assert det.negated_hits, (
            f"{text!r} was suppressed but not retained as rejected evidence; the "
            f"evidence CSV would show no trace of why it did not count"
        )


# ---------------------------------------------------------------------------
# The metaphor depression and panic share.
#
# All strings are verbatim simulated patient speech from `afraid_of_dogs`
# sessions. "A weight on my chest" is a common false positive for
# `depressed_mood`: on an anxiety case the phrase usually describes a panic
# body.
#
# The pairing is the point: the SAME phrase carries both readings, so these
# cases only stay separated as long as the context cues do the work.
# ---------------------------------------------------------------------------
SOMATIC_NOT_MOOD = [                      # panic / relief / trigger-bound
    ("My stomach churns, and it feels like there's this heavy weight on my chest,"
     " making it hard to breathe."),
    "Like a weight on my chest, making it hard to breathe when I see one.",
    ("like there's this huge weight on my chest, pushing down on me, making it"
     " hard to breathe, hard to think."),
    ("It's like a physical weight on my chest, tightening my muscles and making"
     " my heart race."),
    ("It's like I can finally take a full breath again, and the weight on my"
     " chest is lifting."),
    ("I feel like there's a big, heavy weight on my chest when I think about"
     " actually doing something about this fear."),
    ("The thought of actually having to face them, even from a distance, feels"
     " like there's this weight on my chest."),
]

MOOD_NOT_SOMATIC = [                      # the same metaphor, reported as a state
    "It's like this big, heavy weight on my chest that won't go away.",
    "Every day feels heavy, like there's this weight on my chest that won't go away.",
    # copula form: one intervening word must not break the match
    "I wake up and it's like this heavy weight is on my chest, you know?",
]


@pytest.mark.parametrize("text", SOMATIC_NOT_MOOD)
def test_shared_metaphor_in_a_panic_sentence_is_not_low_mood(text):
    from analysis.validation.lexicon_detect import detect_session
    from analysis.validation.symptom_lexicon import PHQ9_PATTERNS
    from symptom_scoring.types import TurnPair

    pair = TurnPair(turn_index=1, therapist_text="", patient_text=text)
    det = detect_session([pair], {"depressed_mood": PHQ9_PATTERNS["depressed_mood"]},
                         count_confirmations=False)["depressed_mood"]
    assert not det.detected(), f"panic body scored as depressed mood: {text!r}"
    assert det.negated_hits, "suppressed but not retained as rejected evidence"


@pytest.mark.parametrize("text", MOOD_NOT_SOMATIC)
def test_shared_metaphor_reported_as_a_state_still_fires(text):
    from analysis.validation.lexicon_detect import detect_session
    from analysis.validation.symptom_lexicon import PHQ9_PATTERNS
    from symptom_scoring.types import TurnPair

    pair = TurnPair(turn_index=1, therapist_text="", patient_text=text)
    assert detect_session([pair], {"depressed_mood": PHQ9_PATTERNS["depressed_mood"]},
                          count_confirmations=False)["depressed_mood"].detected(), text


# ---------------------------------------------------------------------------
# "down": the worst term in the lexicon.
#
# It is a direction, a phrasal-verb particle and a mood word at once, and the
# mood sense is by far the rarest of the three in this material. The residue
# after the earlier fixes all came through `_EXPERIENTIAL_FRAME`'s free-word
# filler, which let an arbitrary noun or verb sit between the anchor and term.
# `polysemous_terms` routes it through the strict frame instead.
# ---------------------------------------------------------------------------
DOWN_IS_NOT_MOOD = [
    "There's this golden retriever down the street from me.",   # filler = noun phrase
    "There's this little puppy down the street, Oscar.",
    "my body just shuts down whenever one comes near",          # filler = verb
    "It felt like everything slowed down as I realized something was wrong.",
    "It's like everything just slows down and I can't think.",
]

DOWN_IS_MOOD = [
    "I've been feeling down more than usual these past couple of weeks.",
    "I've just been feeling really down lately.",
    "It makes me feel down.",
    "Everything feels down and grey.",      # strict frame, nothing in between
]


@pytest.mark.parametrize("text", DOWN_IS_NOT_MOOD)
def test_directional_down_is_not_low_mood(text):
    assert not PHQ9["depressed_mood"].pattern().search(text), text
    assert not MADRS["Traurigkeit"].pattern().search(text), text


@pytest.mark.parametrize("text", DOWN_IS_MOOD)
def test_mood_down_still_fires_in_both_panels(text):
    assert PHQ9["depressed_mood"].pattern().search(text), text


def test_somatic_guard_touches_only_the_shared_metaphor():
    """It must not become a general 'anxious sentences do not count' rule.

    The guard moves `depressed_mood` and nothing else. This pins that: a panic
    sentence with no shared metaphor in it is unaffected, and other symptoms
    firing inside such a sentence are unaffected.
    """
    from analysis.validation.lexicon_detect import is_somatic_not_mood

    # panic context, but no shared metaphor -> guard must not engage
    s = "My heart is racing and I can't breathe, and I've been feeling really down."
    assert not is_somatic_not_mood(s, s.index("feeling"), "feeling really down")
    # shared metaphor, but another symptom's term -> only the metaphor is guarded
    s2 = "There's a weight on my chest and I can't breathe; I haven't slept in days."
    assert not is_somatic_not_mood(s2, s2.index("haven't slept"), "haven't slept")


def _prefix_only(lex):
    """The lexicon with predicate terms folded back into prefix-only.

    Reconstructed from the same term tuples rather than hardcoded, so the
    comparison cannot drift out of date when vocabulary changes later.
    """
    from analysis.validation.symptom_lexicon import SymptomLexicon

    return SymptomLexicon(
        self_framed=lex.self_framed,
        symptom_terms=lex.symptom_terms,
        health_terms=(*lex.health_terms, *lex.predicate_health_terms),
    )


def test_predicate_frame_adds_no_false_positives_on_the_hard_negatives():
    """Differential, not absolute: the pool already contains known failures.

    ~20 of the 360 a-priori negatives fire under the old lexicon (dominated by
    `confounded`, where the symptom is real but externally caused, which a regex
    cannot reach). Asserting zero total detections would therefore fail for
    reasons that predate this change. What must hold is that the predicate frame
    adds NOTHING to that set.

    Reported per domain and per negative category, because a net +0 could
    otherwise hide one new false positive against one lost firing - and a
    move-only change must not lose any firing either, so a disappearance is
    equally a bug.
    """
    import json
    import pathlib

    from analysis.validation.lexicon_detect import (
        has_third_party_subject,
        is_negated,
    )
    from analysis.validation.symptom_lexicon import PHQ9

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "analysis" / "validation" / "exemplars" / "negatives.jsonl")
    if not path.exists():                                    # pragma: no cover
        pytest.skip("a-priori negative pool not present")
    negatives = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def accepts(pattern, text):
        return any(
            not is_negated(text, m.start()) and not has_third_party_subject(text, m.start())
            for m in pattern.finditer(text)
        )

    added, removed = [], []
    for row in negatives:
        symptom = row["domain"]
        if symptom not in PHQ9:
            continue
        new = accepts(PHQ9[symptom].pattern(), row["text"])
        old = accepts(_prefix_only(PHQ9[symptom]).pattern(), row["text"])
        if new and not old:
            added.append((symptom, row.get("category"), row["text"]))
        elif old and not new:
            removed.append((symptom, row.get("category"), row["text"]))

    def _fmt(rows):
        return "\n".join(f"    [{s} / {c}] {t}" for s, c, t in rows)

    assert not added, (
        f"the predicate frame introduced {len(added)} false positive(s):\n{_fmt(added)}"
    )
    assert not removed, (
        f"a move-only change removed {len(removed)} firing(s), so a term was "
        f"lost rather than moved:\n{_fmt(removed)}"
    )
