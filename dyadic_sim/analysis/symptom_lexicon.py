"""The naive symptom lexicons, in one place, as word lists.

**This is the file to edit.** Previously the PHQ-9 lexicon lived in
`symptom_manifest.py` (since deleted) and the MADRS one inside
`dyadic_sim/scoring/instruments/`,
both as raw regex. Here they are plain word lists; the regex is generated, so
adding a term means adding a word.

Three categories per symptom, because the same word means different things
depending on how it is used:

``self_framed``
    Phrases that already carry both their polarity and their owner, so they can
    fire as written: "kill myself", "no energy", "can't sleep". Still suppressed
    by an outer negation ("I would never kill myself").

``symptom_terms``
    Single words naming the symptom: "insomnia", "depressed", "worthless".
    Fire only inside a **self-attribution frame**, so "I have insomnia", "my
    insomnia", and "I've been diagnosed with insomnia" all count while "my dad
    had insomnia" does not. Suppressed when negated.

``health_terms``
    Words naming the *healthy* state: "interest", "appetite", "energy",
    "concentrate", "sleep". Fire only inside a **decrement frame**, because for
    these the symptom is the absence: "no interest", "lost interest", "trouble
    concentrating", "can't sleep". The frame must PRECEDE the term.

``predicate_health_terms``
    Health terms that are also safe on the other side of a copula, so they get
    the decrement frame in both positions: "poor sleep" AND "my sleep is poor",
    "my sleep hasn't been good". Named for that operational property rather than
    for grammar - `concentrat\\w*`, `focus\\w*` and `interest\\w*` all admit
    verbal and inflected forms, so "is it a noun" is the wrong admission test.

    The admission test is whether the predicate form stays clean. It does not
    for `rest` ("the rest is terrible" is not a sleep complaint) or for the food
    nouns ("dinner was awful" is a restaurant, not appetite loss), so those stay
    in ``health_terms``.

Why the split exists. Most PHQ-9 items are losses, so negating the healthy word
*is* the symptom. A single lexicon with a blanket negation filter gets this
exactly backwards - these are the symptom:

    "I don't enjoy things"          "I can't eat much"

and this is a denial:

    "I'm not depressed"

Suppressing all three identically is why a blanket filter changes nothing.

Why frames rather than bare words. A bare token fires overwhelmingly on the
wrong sense, and the wrong senses are the common ones:

    "down"    "I was walking down the street"
              "(pausing, looking down at hands)"
              "it feels like it's pushing down on me"
              "I let my guard down around dogs"
              ... against "I've been feeling down lately", which is rare

    "weight"  "a weight on my shoulders" - a metaphor for burden, not appetite

The more phrase-shaped an entry is, the cleaner it stays: "no appetite" cannot
mean anything else, while "appetite" alone means whatever surrounds it.

Frames must be **adjacent**, not merely nearby. A proximity window is not enough
- with 25 characters of slack, "feels like it's pushing down on me" and "let my
guard down" both still fire, because a mood word and a frame happen to share a
sentence. Only intensifiers may sit between the frame and the term.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words allowed between a frame and its term. Anything else breaks adjacency.
_INTENSIFIER = (
    r"(?:so|really|very|pretty|quite|a bit|a little|kinda|kind of|sort of|"
    r"rather|too|extremely|particularly|somewhat|always|constantly|often|"
    r"sometimes|still|just|already|generally|mostly|totally|completely|"
    # auxiliaries and copulas: "I have BEEN so anxious", "I am FEELING down"
    r"been|being|feeling|felt|feel|getting|got|gotten)"
)

# "the symptom belongs to me": first-person ownership or predication.
_SELF_FRAME = (
    r"(?:"
    r"i(?:'m| am| ve| have| had| was| feel| felt| get| got| been|'ve been| keep)"
    r"|i(?:'m| am|'ve| have| had| was)? ?(?:been )?(?:diagnosed with|struggling with|dealing with|suffering from)"
    r"|i know i(?:'m| am| have)"
    r"|i think i(?:'m| am| have)"
    r"|my"
    r"|makes me(?: feel)?"
    r"|leaves me(?: feeling)?"
    r"|feeling"
    r"|felt"
    r")"
)

# "the experience belongs to me, but the subject is not I": impersonal
# self-report. `_SELF_FRAME` requires a first-person subject, so "everything feels
# flat", "nothing goes in" and "the days blur together" could not fire no matter
# what vocabulary was added. This is a frame gap, not a missing word.
_EXPERIENTIAL_FRAME = (
    r"(?:"
    r"(?:everything|nothing|anything|things|it all|the days|the nights|food|sleep)"
    r"|(?:my|the) (?:head|mind|brain|body|hands|legs|mood|sleep|appetite|concentration)"
    r"|there'?s(?: (?:a|this|always))?"
    r")"
    r"(?:\s+\w+){0,2}?\s*"
    r"(?:feels?|felt|seems?|is|are|was|were|gets?|got|goes|go|went|just)?"
)

# The same frame with the free-word filler removed, for terms too polysemous to
# survive it. The filler earns its place for most terms ("my mood HAS BEEN
# really low"), but it lets an arbitrary noun or verb sit between the anchor and
# the term, and for a word like "down" that is the whole ballgame:
#
#   "There's this golden retriever down the street"   filler = a noun phrase
#   "my body just shuts down"                         filler = a verb
#   "everything slowed down"                          filler = a verb
#
# All three fired `depressed_mood`. Every false positive of this shape came
# through the filler; the true positives never need it, using either
# `_SELF_FRAME` ("I've been feeling down") or this frame with nothing in between
# ("Everything feels down"). Which makes the fix the module's own stated rule,
# applied to one more term: only intensifiers may sit between a frame and its
# term.
_EXPERIENTIAL_FRAME_STRICT = (
    r"(?:"
    r"(?:everything|nothing|anything|things|it all|the days|the nights|food|sleep)"
    r"|(?:my|the) (?:head|mind|brain|body|hands|legs|mood|sleep|appetite|concentration)"
    r"|there'?s(?: (?:a|this|always))?"
    r")\s*"
    r"(?:feels?|felt|seems?|is|are|was|were|gets?|got|goes|go|went|just)?"
)

# "the healthy thing is reduced or gone": the symptom for loss-type items.
_DECREMENT_FRAME = (
    r"(?:"
    r"no|not|don'?t|doesn'?t|didn'?t|can'?t|cannot|couldn'?t|won'?t|haven'?t|hasn'?t"
    r"|hardly|barely|scarcely|rarely|seldom|never"
    r"|little|less|low|poor|bad|worse|lack of|lacking"
    r"|lost|losing|lose|gone off|stopped"
    r"|trouble|difficulty|hard to|struggle to|struggling to|problems? with"
    r"|too tired to|unable to"
    r")"
)

# The same decrement, on the other side of a copula. `_DECREMENT_FRAME` only
# ever generates "frame + term", so "poor sleep" fired and "my sleep is poor"
# could not - same words, opposite order, opposite outcome. Found by hand-reading
# the sessions a sleep detector had stayed silent on. One of them opens:
#
#   "I'm tired of being afraid all the time. And yeah, sleep's been bad lately
#    too. It's like my brain won't turn off, you know?"
#
# That produced neither a hit nor a *rejected* hit - so not a negation or an
# ownership refusal. Every word the lexicon needed was present; only the order
# defeated it.
#
# The same CLASS as the gap `_EXPERIENTIAL_FRAME` closed ("everything feels
# flat" could not fire while the frame demanded a first-person subject). That
# frame already handles predicate position, but it applies to `symptom_terms`
# and these domains carry their vocabulary in `health_terms`, so the machinery
# existed and did not connect.
#
# Longer alternatives precede shorter ones they overlap ("okay|ok",
# "worst|worse", "has not been|has been"), since Python alternation is
# first-match rather than longest-match.
_PREDICATE_COPULA = (
    r"(?:'s|’s|is|are|was|were|has been|have been|'ve been|had been"
    r"|feels?|felt|seems?|seemed|gets?|got|becomes?|became)"
)

_NEGATED_COPULA = (
    r"(?:isn'?t|is not|'s not|aren'?t|are not|wasn'?t|was not|weren'?t"
    r"|hasn'?t been|has not been|haven'?t been|hadn'?t been)"
)

# "the healthy thing is in a bad state". Exclusively negative valence: any word
# here is read as the symptom the moment it predicates a health term.
_BAD_STATE = (
    r"(?:bad|poor|terrible|awful|horrible|dreadful|rubbish|shot|gone"
    r"|non-?existent|patchy|broken|erratic|disrupted|disturbed|shattered"
    r"|wrecked|hopeless|minimal|low|worst|worse|off|impossible"
    r"|a mess|a nightmare|a struggle|a joke|a disaster"
    r"|all over the place|out the window|rough|zero|nil|nothing)"
)

# "the healthy thing is NOT in a good state" - only ever used after
# `_NEGATED_COPULA`, where the denial of health is the symptom.
#
# "better" is deliberately absent. "my sleep isn't better" would otherwise count
# as symptom evidence, and it is an inference about trajectory rather than a
# report of a state. This instrument is the auditable floor, so it declines the
# inference.
_GOOD_STATE = (
    r"(?:good|great|fine|right|normal|okay|ok|there|back|the same"
    r"|what it (?:used to be|was)|up to much|restful)"
)


@dataclass(frozen=True)
class SymptomLexicon:
    """One symptom's terms. Edit the tuples; the regex is derived."""

    self_framed: tuple[str, ...] = ()
    symptom_terms: tuple[str, ...] = ()
    polysemous_terms: tuple[str, ...] = ()
    health_terms: tuple[str, ...] = ()
    predicate_health_terms: tuple[str, ...] = ()
    _compiled: dict = field(default_factory=dict, repr=False, compare=False)

    def pattern(self) -> re.Pattern[str]:
        if "p" not in self._compiled:
            parts = []
            for term in self.self_framed:
                parts.append(rf"\b{term}\b")
            for term in self.symptom_terms:
                # Either "I feel <term>" or "everything feels <term>". The second
                # is not a nicety: measured against the a-priori exemplars, the
                # first-person-only frame was rejecting impersonal self-report
                # ("everything feels flat", "nothing goes in") that a reader
                # would score without hesitation.
                parts.append(rf"\b{_SELF_FRAME}\s+(?:{_INTENSIFIER}\s+)*{term}\b")
                parts.append(rf"\b{_EXPERIENTIAL_FRAME}\s+(?:{_INTENSIFIER}\s+)*{term}\b")
            for term in self.polysemous_terms:
                # Same two frames, but the experiential one may not put a free
                # word between itself and the term. See _EXPERIENTIAL_FRAME_STRICT.
                parts.append(rf"\b{_SELF_FRAME}\s+(?:{_INTENSIFIER}\s+)*{term}\b")
                parts.append(
                    rf"\b{_EXPERIENTIAL_FRAME_STRICT}\s+(?:{_INTENSIFIER}\s+)*{term}\b"
                )
            for term in (*self.health_terms, *self.predicate_health_terms):
                # up to two filler words, so "no real interest in things" counts
                parts.append(rf"\b{_DECREMENT_FRAME}\s+(?:\w+\s+){{0,2}}{term}\b")
            for term in self.predicate_health_terms:
                # `\s*` not `\s+` before the copula, so the clitic in "sleep's
                # been bad" attaches. That contraction is the construction the
                # whole predicate frame was added for.
                #
                # Only intensifiers may sit between the copula and the state
                # word, and this is load-bearing rather than tidiness: "not" is
                # not an intensifier, so "my sleep is not bad" cannot match and
                # the denial is refused by the existing adjacency rule instead
                # of by a new one. Widening this filler would reintroduce the
                # polarity reversal that was found in the reply classifier,
                # where a denial was being scored as a confirmation.
                parts.append(
                    rf"\b{term}\s*{_PREDICATE_COPULA}\s+(?:{_INTENSIFIER}\s+)*{_BAD_STATE}\b"
                )
                parts.append(
                    rf"\b{term}\s*{_NEGATED_COPULA}\s+(?:{_INTENSIFIER}\s+)*{_GOOD_STATE}\b"
                )
            self._compiled["p"] = re.compile("|".join(parts), re.IGNORECASE)
        return self._compiled["p"]


# --------------------------------------------------------------------------
# PHQ-9. Edit these.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Hazard tiers for PHQ-9 item 9. A SUPPLEMENTARY axis, not a tenth item.
#
# The item conflates two clinically distinct things: "thoughts that you would be
# better off dead" (a passive death wish) and "of hurting yourself" (self-harm
# ideation). MADRS `Suizid` spans wider still, from "life is not worth living"
# through to plans. Reporting them as one column makes "it's all too much" and
# "I've thought about ending my life" the same observation, which they are not.
#
# Kept OUT of the `PHQ9` dict deliberately. The nine keys and their order are
# load-bearing (`target_z` indexes by position, and tests/test_prior_vocabulary.py
# pins them against the prompt vocabulary), so this is a second axis applied to
# item 9's hits rather than a change to the instrument.
#
# Tiers follow the standard passive/active distinction. Deliberately NOT split
# further into method, intent and plan: that is C-SSRS territory, the exemplars
# were written non-graphic by design, and a regex should not be asked to grade
# imminence.
HAZARD_TIERS = {
    "passive_death_wish": SymptomLexicon(
        self_framed=(r"better off dead", r"no reason to live", r"not worth living",
                     r"wish I (?:was|were) dead",
                     r"wish(?:ing)? I (?:wouldn'?t|didn'?t|would not) wake up",
                     r"(?:not|didn'?t|don'?t|doesn'?t) want(?:ing)? to (?:be here|be alive|exist|wake up|live)",
                     r"wish(?:ing)? I could (?:disappear|vanish|not be here)",
                     r"(?:everyone|people|they'?d|she'?d|he'?d) (?:would )?be better off without me",
                     r"easier if I (?:wasn'?t|weren'?t|just wasn'?t) (?:here|around)",
                     r"stop existing", r"thinking about (?:death|dying|not being here)",
                     r"whether it'?s worth", r"can'?t go on", r"no point (?:in|to) (?:it|anything|going)"),
    ),
    "active_self_harm": SymptomLexicon(
        self_framed=(r"kill(?:ing)? myself", r"hurt(?:ing)? myself", r"harm(?:ing)? myself",
                     r"self[- ]harm\w*", r"end(?:ing)? my (?:own )?life",
                     r"end(?:ing)? it all", r"take my (?:own )?life",
                     r"suicid\w*", r"cut(?:ting)? myself"),
    ),
}

PHQ9 = {
    "lack_of_pleasure": SymptomLexicon(
        # "used to <enjoy>" is the single most common way anhedonia is said and was
        # entirely absent: `used` appeared in 11 of this domain's 79 missed
        # exemplars. It is a construction, not a word.
        self_framed=(r"anhedoni\w*", r"don'?t care about anything",
                     r"nothing (?:feels|seems) (?:good|fun|enjoyable)",
                     r"used to (?:really )?(?:enjoy|love|like|look forward)",
                     r"(?:doesn'?t|don'?t) do (?:anything|much|it) for me",
                     r"nothing (?:lands|appeals|excites|grabs)",
                     r"going through the motions",
                     r"can'?t get (?:into|excited about)",
                     r"no (?:longer )?look(?:ing)? forward"),
        symptom_terms=(r"numb", r"empty inside", r"flat", r"pointless", r"boring"),
        # `enjoy\w*`, `excit\w*` and `care about` stay prefix-only: they are
        # verbs, so the predicate form would be reaching for a construction they
        # do not take. `fun` and `looking forward` predicate a subject rather
        # than being predicated ("nothing is fun" is already in self_framed).
        health_terms=(r"enjoy\w*", r"fun", r"looking forward", r"excit\w*",
                      r"care about"),
        predicate_health_terms=(r"pleasure", r"interest\w*", r"joy"),
    ),
    "depressed_mood": SymptomLexicon(
        # Explicit mood words are a small corner of this domain: only 6 of 100
        # a-priori exemplars used sad/depressed/miserable/unhappy at all, and
        # across 300 external sessions a self-attributed `depress*` never occurs.
        # The register is metaphor and indirection, so that is what is listed.
        self_framed=(r"low mood", r"hopeless\w*", r"despair\w*",
                     r"under a cloud",
                     # Optional copula: "this heavy weight IS on my chest" was
                     # silent while "this heavy weight on my chest" fired, so a
                     # single intervening word decided it. See SOMATIC_AMBIGUOUS
                     # below - on an anxiety bed this phrase needs a guard, not
                     # just a match.
                     r"weight (?:is |was |'s )?on my (?:chest|shoulders|mind)",
                     r"can'?t see (?:a way|the point|it getting better)",
                     r"not myself", r"stuck in a rut", r"can'?t shake (?:it|this)",
                     r"nothing (?:lifts|helps|shifts) it", r"something (?:feels|is) off"),
        symptom_terms=(r"sad\w*", r"depress\w*", r"miserabl\w*", r"empty", r"blue",
                       r"awful", r"terrible", r"flat", r"grey", r"gray", r"gloomy", r"bleak",
                       r"heavy", r"heaviness", r"hollow", r"numb", r"unhappy", r"low",
                       r"discouraged", r"lost"),
        # "down" is the single worst term in this lexicon. It is a direction, a
        # phrasal-verb particle and a mood word, and the first two are far more
        # common in this material:
        #     "There's this golden retriever down the street"
        #     "my body just shuts down"      "everything slowed down"
        # Routed through the strict frame so a noun or verb cannot sit between
        # the frame and the term; see _EXPERIENTIAL_FRAME_STRICT.
        polysemous_terms=(r"down",),
        health_terms=(),
    ),
    "sleep_problems": SymptomLexicon(
        # NOTE: "insomnia" is deliberately NOT here. A bare disease name fires on
        # "my dad had insomnia"; it belongs in symptom_terms so it needs the
        # self-attribution frame.
        self_framed=(r"can'?t sleep", r"couldn'?t sleep", r"toss(?:ing)? and turn(?:ing)?",
                     r"lying awake", r"lie awake", r"lay awake", r"wide awake", r"nightmares?",
                     r"waking up (?:at|in|every|three|four|five)", r"keep waking",
                     r"up (?:half|most of|at) the night", r"staring at the ceiling",
                     r"barely slept", r"hardly slept", r"never slept",
                     r"sleeping (?:too much|all day|through)", r"in bed (?:all|half|by|until|til)",
                     # NOT "fall asleep" / "get to sleep" / "back to sleep" as bare
                     # phrases: they carry no polarity, so "if I fall asleep, wake
                     # me up" fired. The framed forms ("trouble getting to sleep")
                     # are already reached by health_terms + _DECREMENT_FRAME.
                     r"awake (?:at|until|till) \d", r"awake for hours"),
        symptom_terms=(r"insomnia",),
        # `rest` stays prefix-only. "no rest" is a sleep complaint; "the rest is
        # terrible" is the other sense of the word entirely, and the corpus is
        # full of it ("the rest of my life", "the rest of me"). `slept` is a
        # verb: the decrement reads "barely slept", which self_framed carries.
        health_terms=(r"slept", r"rest"),
        predicate_health_terms=(r"sleep\w*",),
    ),
    "low_energy": SymptomLexicon(
        self_framed=(r"no energy", r"worn out", r"run down", r"wiped out",
                     r"nothing (?:left )?in the tank", r"takes everything (?:I have|out of me)",
                     r"can'?t get (?:going|moving|off the (?:sofa|couch))",
                     r"barely (?:get|got) (?:up|through|going)", r"no get up and go"),
        symptom_terms=(r"tired", r"exhausted", r"fatigued?", r"drained", r"letharg\w*",
                       r"sluggish", r"shattered", r"knackered", r"wrung out", r"spent"),
        predicate_health_terms=(r"energy", r"motivation", r"stamina"),
    ),
    "appetite_changes": SymptomLexicon(
        self_framed=(r"no appetite", r"skip(?:ping|ped)? meals?", r"forget(?:ting)? to eat",
                     r"lost weight", r"losing weight", r"gained weight", r"gaining weight",
                     r"overeat\w*", r"binge\w*", r"gone off (?:my )?food",
                     r"food (?:doesn'?t|does not|has no) (?:appeal|interest|taste)",
                     r"push(?:ing)? (?:it|food|things) (?:a)?round",
                     r"(?:can'?t|couldn'?t) face (?:food|eating|breakfast|dinner|it)",
                     r"force (?:it|myself|food) down", r"remind myself to eat",
                     r"eating (?:my way )?(?:through|all)", r"picking at"),
        symptom_terms=(),
        # The food nouns stay prefix-only, and this is the clearest case for the
        # split: "dinner was awful" is a restaurant review, "the food is
        # terrible" is a canteen. Only the two nouns naming the FACULTY -
        # appetite and hunger - mean the symptom when they are predicated.
        health_terms=(r"eat\w*", r"hungry", r"meals?", r"food", r"breakfast",
                      r"lunch", r"dinner", r"plate"),
        predicate_health_terms=(r"appetite", r"hunger"),
    ),
    "feelings_of_failure_or_guilt": SymptomLexicon(
        self_framed=(r"not good enough", r"let (?:\w+ ){0,2}down", r"blam(?:e|ing) myself", r"my fault",
                     r"letting (?:everyone|people|them|him|her|my \w+) down",
                     r"can'?t do anything right", r"deserve better than me",
                     r"hard on myself", r"disappointed in myself",
                     r"better off without me", r"everyone else (?:manages|copes|can)",
                     r"my own fault", r"should have (?:done|been|known)"),
        symptom_terms=(r"guilt\w*", r"failure", r"failed", r"failing", r"worthless\w*",
                       r"ashamed", r"shame", r"useless", r"a burden", r"rubbish about myself",
                       r"a fraud", r"a mess", r"not enough"),
        health_terms=(),
    ),
    "concentration_problems": SymptomLexicon(
        self_framed=(r"can'?t think (?:straight|clearly)", r"mind (?:wanders|wandering|goes blank)",
                     r"lose track", r"zone out", r"zoning out",
                     # PHQ-9 item 7 is worded "such as reading the newspaper or
                     # watching television". The instrument's own example did not
                     # fire before this line existed.
                     r"same (?:page|paragraph|line|sentence|passage)",
                     r"re-?read(?:ing)?", r"read(?:ing)? (?:it|the same|that) (?:again|over|twice)",
                     r"can'?t follow (?:a|the|what|it|anything)",
                     r"los(?:e|ing) the thread", r"nothing goes in",
                     r"can'?t hold (?:a thought|onto|my attention)",
                     r"takes (?:me )?(?:forever|ages|three times)",
                     r"had to (?:read|go over) it (?:again|twice)"),
        symptom_terms=(r"distracted", r"forgetful", r"foggy", r"fog", r"scattered", r"vague",
                       r"muddled", r"blank"),
        # `remember\w*` stays prefix-only: it is a verb, so "my remembering is
        # bad" is not English anyone speaks. The other three are predicated
        # constantly ("my concentration is shot", "my focus is gone").
        health_terms=(r"remember\w*",),
        predicate_health_terms=(r"concentrat\w*", r"focus\w*", r"attention"),
    ),
    "psychomotor_changes": SymptomLexicon(
        self_framed=(r"can'?t sit still", r"can'?t keep still", r"pacing", r"paced",
                     r"moving slow\w*", r"speaking slow\w*", r"talking slow\w*", r"walking slow\w*",
                     r"slowed (?:down|right down)", r"on edge", r"slow motion",
                     r"wading through", r"like treacle", r"underwater",
                     # PHQ-9 item 8 says "so slowly that other people could have
                     # noticed", so third-party report is the instrument's own frame.
                     r"(?:people|everyone|they|my \w+) (?:keep |kept |all )?(?:tell|telling|told|say|saying|said|ask|asking|asked|notice[ds]?|noticing) (?:me|if)",
                     r"keep fidgeting", r"bouncing my (?:leg|knee|foot)",
                     r"jiggling", r"can'?t stop moving"),
        symptom_terms=(r"restless\w*", r"agitat\w*", r"fidget\w*", r"jittery", r"twitchy",
                       r"slow", r"slower", r"wound up", r"revved up"),
        health_terms=(),
    ),
    "thoughts_of_death_or_self_harm": SymptomLexicon(
        self_framed=(r"kill(?:ing)? myself", r"hurt(?:ing)? myself", r"self[- ]harm\w*", r"better off dead",
                     r"end(?:ing)? my life", r"end(?:ing)? it all", r"no reason to live",
                     r"not want(?:ing)? to (?:be here|live|wake up|be alive|exist)",
                     r"wish I (?:was|were) dead", r"suicid\w*",
                     r"wish(?:ing)? I (?:wouldn'?t|didn'?t|would not) wake up",
                     r"(?:everyone|people|they'?d|things'?d|she'?d|he'?d) (?:would )?be better off without me",
                     r"wish(?:ing)? I could (?:disappear|vanish|not be here)",
                     r"thinking about (?:death|dying|not being here)",
                     r"easier if I (?:wasn'?t|weren'?t|just wasn'?t) (?:here|around)",
                     r"stop existing", r"didn'?t want to be alive",
                     r"whether it'?s worth"),
        symptom_terms=(),
        health_terms=(),
    ),
}


# --------------------------------------------------------------------------
# Metaphors that depression and panic SHARE.
#
# "A weight on my chest" is a good low-mood metaphor in a depressive patient and
# a literal description of chest tightness in a panicking one. The term cannot
# tell which it is; only the surrounding sentence can:
#
#   low mood   "this big, heavy weight on my chest THAT WON'T GO AWAY"
#              "Every day feels heavy, like there's this weight on my chest..."
#              "I wake up and it's like this heavy weight is on my chest"
#   panic      "...weight on my chest, MAKING IT HARD TO BREATHE"
#              "a physical weight on my chest, TIGHTENING MY MUSCLES and
#               MAKING MY HEART RACE"
#              "the weight on my chest IS LIFTING"          <- relief, not mood
#              "...when I THINK ABOUT actually doing something about this fear"
#
# The mood reading pairs the metaphor with PERSISTENCE; the panic reading pairs
# it with BREATHING/CARDIAC ACTIVATION, RELIEF, or an anticipatory trigger. That
# is the discriminator, and it is a property of the context rather than of the
# term - the same shape as negation and ownership, so it is handled the same
# way, in `lexicon_detect`.
#
# This matters most on a case whose presenting problem is anxiety, where the
# panic reading is the common one and an unguarded match makes the detector
# report depression every time the patient describes a racing heart.
SOMATIC_AMBIGUOUS = re.compile(
    r"\bweight (?:is |was |'s )?(?:on|in) my (?:chest|shoulders|mind)\b",
    re.IGNORECASE,
)


def as_patterns(lexicons: dict[str, SymptomLexicon]) -> dict[str, str]:
    """Symptom -> regex string, for detectors that take plain patterns."""
    return {name: lex.pattern().pattern for name, lex in lexicons.items()}


PHQ9_PATTERNS = as_patterns(PHQ9)


# --------------------------------------------------------------------------
# MADRS. Edit these.
#
# Topic keys are the German labels the MADRS-BERT checkpoint was trained on, so
# they must not be renamed. The terms are English because detection runs on the
# transcript before translation, where evidence is still traceable to what the
# patient actually said.
#
# Not a relabelling of PHQ-9. Four topics differ in substance:
#   Anspannung          inner tension and dread. PHQ-9 has no anxiety item.
#   Antriebslosigkeit   lassitude: difficulty starting and finishing things,
#                       which is not the same as feeling tired.
#   Gefuehlslosigkeit   inability to feel, emotional numbness. Overlaps
#                       lack_of_pleasure but is about absent feeling, not
#                       absent enjoyment.
#   Gedanken            pessimistic thoughts: guilt, self-blame, and a bleak
#                       view of the future, which PHQ-9 splits differently.
# --------------------------------------------------------------------------
MADRS = {
    "Traurigkeit": SymptomLexicon(
        self_framed=(r"low mood", r"hopeless\w*", r"despair\w*", r"can'?t see the point"),
        symptom_terms=(r"sad\w*", r"depress\w*", r"miserabl\w*", r"gloomy", r"bleak"),
        polysemous_terms=(r"down",),   # see PHQ-9 depressed_mood
    ),
    "Anspannung": SymptomLexicon(
        self_framed=(r"on edge", r"panic attacks?", r"can'?t relax", r"knot in my stomach",
                     r"heart (?:races|racing|pounding)", r"keyed up"),
        symptom_terms=(r"anxious", r"anxiety", r"tense", r"tension", r"panick\w*", r"uneasy",
                       r"irritabl\w*", r"agitat\w*", r"nervous", r"dread", r"scared", r"afraid"),
    ),
    "Schlaf": SymptomLexicon(
        self_framed=(r"can'?t sleep", r"couldn'?t sleep", r"toss(?:ing)? and turn(?:ing)?",
                     r"lying awake", r"lie awake", r"wide awake", r"up all night",
                     r"nightmares?", r"waking up (?:at|in)"),
        symptom_terms=(r"insomnia",),
        # Register split as in PHQ-9 `sleep_problems`; see the reasoning there.
        health_terms=(r"slept", r"rest"),
        predicate_health_terms=(r"sleep\w*",),
    ),
    "Appetit": SymptomLexicon(
        self_framed=(r"no appetite", r"skip(?:ping|ped)? meals?", r"forget(?:ting)? to eat",
                     r"lost weight", r"losing weight", r"food (?:tastes|has) no"),
        health_terms=(r"eat\w*", r"hungry", r"meals?"),
        predicate_health_terms=(r"appetite", r"hunger"),
    ),
    "Konzentration": SymptomLexicon(
        self_framed=(r"can'?t think (?:straight|clearly)", r"mind (?:wanders|wandering|goes blank)",
                     r"lose track", r"zone out", r"zoning out", r"read the same"),
        symptom_terms=(r"distracted", r"forgetful", r"foggy", r"fog"),
        health_terms=(r"remember\w*",),
        predicate_health_terms=(r"concentrat\w*", r"focus\w*", r"attention"),
    ),
    "Antriebslosigkeit": SymptomLexicon(
        self_framed=(r"can'?t get (?:started|going)", r"takes? everything (?:i have|out of me)",
                     r"no energy", r"drag myself", r"put(?:ting)? (?:it|things) off",
                     r"everything (?:is|feels) an effort"),
        symptom_terms=(r"sluggish", r"letharg\w*", r"unmotivated"),
        # `start\w*` and `get(?:ting)? going` stay prefix-only: both are verbal,
        # and "starting is bad" is not how the complaint is voiced.
        health_terms=(r"get(?:ting)? going", r"start\w*"),
        predicate_health_terms=(r"motivation", r"energy"),
    ),
    "Gefuehlslosigkeit": SymptomLexicon(
        self_framed=(r"anhedoni\w*", r"nothing (?:feels|seems) (?:good|real|anything)",
                     r"don'?t feel anything", r"can'?t feel anything", r"going through the motions",
                     r"behind glass"),
        symptom_terms=(r"numb\w*", r"empty", r"flat", r"detached", r"disconnected"),
        health_terms=(r"enjoy\w*",),
        predicate_health_terms=(r"pleasure", r"interest\w*", r"joy", r"feelings?"),
    ),
    "Gedanken": SymptomLexicon(
        self_framed=(r"not good enough", r"blam(?:e|ing) myself", r"my fault", r"let (?:\w+ ){0,2}down",
                     r"nothing will (?:change|get better)", r"no future"),
        symptom_terms=(r"guilt\w*", r"failure", r"failed", r"failing", r"worthless\w*",
                       r"ashamed", r"shame", r"useless", r"a burden", r"regret\w*", r"pessimis\w*"),
    ),
    "Suizid": SymptomLexicon(
        self_framed=(r"kill(?:ing)? myself", r"hurt(?:ing)? myself", r"self[- ]harm\w*", r"better off dead",
                     r"end(?:ing)? my life", r"end(?:ing)? it all", r"no reason to live", r"suicid\w*",
                     r"not want(?:ing)? to (?:be here|live|wake up)", r"wish I (?:was|were) dead"),
    ),
}

MADRS_PATTERNS = as_patterns(MADRS)
