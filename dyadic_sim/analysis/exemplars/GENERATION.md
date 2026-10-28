# PHQ-9 expression exemplars: how these pools were made

Provenance record for the exemplar pools in this directory. Written so the
generation is reportable in the manuscript and the pools are releasable as
supplementary material.

## What these are

First-person utterances that constitute **linguistic evidence** for each PHQ-9
symptom domain. Used two ways:

- verbatim, as multi-reference embedding targets (`REFERENCE_SETS` in
  `dyadic_sim/core/vocabulary.py`);
- distilled into recurring phrase patterns for the naive lexicon
  (`dyadic_sim/analysis/symptom_lexicon.py`).

## Scope caveat

PHQ-9 is a **screening and severity instrument**. Each item is scored by
frequency over a two-week window, and the instrument was validated as a brief
depression measure — never as a sentence-level classification ontology. Two
consequences, committed to before any result was seen:

1. **Domain overlap is partly inherent to the instrument**, not evidence that
   the exemplars are sloppy. `low_energy` and `lack_of_pleasure` are not cleanly
   separable at sentence level because PHQ-9 never claimed they were. The purity
   confusion matrix should be read as *how much of PHQ-9's domain structure
   survives translation to sentence level*, which is a result about the
   instrument.
2. **An utterance is evidence, not a diagnosis and not a severity score.**
   Nothing here recovers the two-week frequency anchoring that PHQ-9 scoring
   depends on. A person producing one of these sentences is not thereby
   depressed.

## Evidential grading

The axis that makes this more than a synonym list. Independent of intensity.

| grade | definition | example (`lack_of_pleasure`) |
|---|---|---|
| `discriminative` | supports this domain **and rules out** the neighbouring explanation | "I don't enjoy the gym anymore, even when I have the energy" |
| `consistent` | compatible with this domain, but equally explained by another | "I stopped going to the gym" |
| `confounded` | symptom present, cause external — not evidence of a *depressive* symptom | "I slept three hours because the baby was crying" |

`confounded` items are written into `negatives.jsonl`, not the positive pools.

This grading also bounds the detector. Regex over clauses can reach
consistent-grade evidence; discriminative grading requires reasoning about
alternative causes. The naive checker is therefore a screening instrument for
text with the same profile as PHQ-9 itself — sensitive, not specific — and
should not be tuned toward a precision it cannot have.

## Pools

| pool | file | source | role |
|---|---|---|---|
| H | `pool_h.jsonl` | human, project author | uncontaminated anchor |
| A | `pool_a.jsonl` | 9 subagents, one per domain | working pool |
| B | `pool_b.jsonl` | independent second pass | saturation check |
| — | `negatives.jsonl` | same generation pass as A | `SHOULD_NOT_FIRE` cases |

**H∩A** measures whether generation reaches human intuition. **A∩B** measures
whether the concept space is saturated or 100/domain is too few. Both by
embedding similarity, not string match.

Pool H was written and frozen **before** any generated pool existed, so it is
not contaminated by having seen them.

### Pool H composition (observed after tagging)

| | |
|---|---|
| exemplars | 108 (12 per domain × 9) |
| grade | **consistent 104, discriminative 4** |
| subject | **first_person 88, impersonal 19, third_party 1** |

This skew is itself the justification for two requirements in the prompt below.
Writing naturally produces almost entirely consistent-grade, first-person
sentences; discriminative evidence and non-first-person subjects have to be
asked for explicitly or they do not appear. The second matters because
`_SELF_FRAME` in the lexicon requires a first-person subject, so a pool that
never varies the subject would silently fail to exercise the frame gap that
motivated this work.

Text in `pool_h.jsonl` is verbatim as authored. Tags were applied afterwards by
a different author; each record carries `authored_by` and `tagged_by` so the
human anchor stays distinguishable from the annotation.

## Generation prompt, version 1

Used for pools A and B. Pool B differs only in the framing note at the end.
`prompt_version` is recorded per record; any edit bumps the version and produces
a new file rather than overwriting.

````
You are generating a synthetic dataset of natural-language expressions of
depressive symptoms, for evaluating a text classifier.

Generate 100 unique first-person utterances for the symptom domain below.

PHQ-9 DOMAIN: [DOMAIN]
CLINICAL CONCEPT: [DESCRIPTION]

These should be things an ordinary person might spontaneously say in
conversation, a text message, a journal entry, an online post, or to a
clinician.

## Do not consult data
Do NOT read any file under `data/`, and do not look at any transcript or
corpus. These exemplars must be written blind. The validity of the coverage
measurement depends on it.

## Evidential grade (required per exemplar)
Label each exemplar:
- `discriminative` — evidence for THIS domain that also rules out the most
  plausible neighbouring explanation. Example for loss of interest: "I don't
  enjoy the gym anymore, even when I have the energy" (rules out fatigue).
- `consistent` — compatible with this domain but equally explained by
  something else. Example: "I stopped going to the gym".
- `confounded` — the symptom is present but externally caused, so it is NOT
  evidence of a depressive symptom. Example: "I slept three hours because the
  baby was crying".

Target mix: about 40% discriminative, 45% consistent, 15% confounded.
Discriminative exemplars are the most valuable and the hardest to write; do not
skimp on them.

## Vary the grammatical subject
Do not write everything in the first person. Deliberately mix:
- first person — "I can't concentrate", "my sleep is wrecked"
- impersonal / experiential — "everything feels flat", "nothing sounds fun",
  "the days blur together"
- third-party report — "people keep telling me I'm slow", "my partner says I
  barely eat"

Third-party report matters most for psychomotor change, which people rarely
observe in themselves.

## Diversity
Vary vocabulary, sentence structure, formality, emotional intensity, and
explicitness. Include short and long utterances, informal grammar, and mild
through strong expressions. Include people who do not use psychiatric
terminology. Write like real people:
  "I just don't enjoy stuff like I used to."
not like a questionnaire:
  "I am experiencing significant anhedonia."

Do not produce 100 sentences differing by one or two words. Avoid starting
sentences with "I feel" repeatedly.

## Domain purity
Every exemplar must be evidence for the TARGET domain. Do not introduce
symptoms belonging primarily to another PHQ-9 domain unless needed for natural
context.
- FATIGUE: lack of energy and exhaustion, not inability to sleep.
- CONCENTRATION: difficulty sustaining attention, not lack of motivation.
- LOSS OF INTEREST: reduced enjoyment or interest, not general sadness.
- LOW SELF-WORTH: failure, guilt, worthlessness, self-criticism or letting
  others down, not general depressed mood.

## Important
Do NOT assume a person producing one of these statements has depression. An
exemplar represents possible linguistic evidence of the specified symptom only.

For the DEATH / SELF-HARM domain, exemplars must remain non-graphic. Do not
mention methods, means, locations, procedures, or instructions.

## Output
JSONL, one object per line, no surrounding prose or code fence:
{"text": "...", "grade": "discriminative|consistent|confounded",
 "directness": "direct|indirect", "intensity": "mild|moderate|strong",
 "style": "casual|emotional|behavioral|somatic|reflective|other",
 "subject_type": "first_person|impersonal|third_party"}

Then make a second pass and remove duplicates and near-duplicates, exemplars
that primarily evidence a different PHQ-9 domain, overly clinical or unnatural
phrasing, and phrases that would rarely occur in spontaneous conversation.
Replace whatever you removed so the final output still contains exactly 100
unique exemplars.
````

### Negatives addendum

Appended to the prompt for the same generation pass. ~40 per domain, into
`negatives.jsonl`.

````
Additionally generate 40 NEGATIVE examples for this domain: sentences that a
naive keyword detector would plausibly fire on, but which are NOT evidence of
this depressive symptom. Tag each with a category:

- `third_party` — the symptom belongs to someone else: "my dad had insomnia"
- `denial` — explicitly denied: "I'm not depressed, just tired"
- `idiom` — symptom vocabulary used non-symptomatically: "a weight off my
  shoulders", "I let my guard down"
- `aspiration` — wanting the healthy state: "I want to enjoy things again"
- `confounded` — symptom present but externally caused: "jet lag wrecked my
  sleep", "the new medication kills my appetite"
- `neighbouring_domain` — evidence for a DIFFERENT PHQ-9 domain that is
  commonly mistaken for this one: "I'm too tired to go to the gym" is fatigue,
  not loss of interest

For every `neighbouring_domain` negative, also supply the matched
discriminative positive that differs precisely by ruling the alternative out,
as field `matched_positive`. Example:
  negative: "I'm too tired to go to the gym"
  matched_positive: "I don't enjoy the gym anymore, even when I have the energy"

Same JSONL format plus "category" and, where applicable, "matched_positive".
````

## Provenance log

| pool | generated | generator | prompt_version | frozen at commit | n |
|---|---|---|---|---|---|
| H | 2026-08-12 | human (project author) | — | _pending_ | 108 |
| A | 2026-08-12 | claude-opus-5, 9 subagents (one per domain) | v1 | _pending_ | 900 |
| B | 2026-08-12 | claude-opus-5, 3 subagents, clinician-recall framing | v1-recall | _pending_ | 300 |
| negatives | 2026-08-12 | same pass as A | v1 | _pending_ | 360 |

Pool B covers only `depressed_mood`, `low_energy` and `lack_of_pleasure` — it
exists to measure saturation, not for coverage.

Agents disagreed on what to write in `grade` for a negative (one used
`"negative"`, another reused `"confounded"`). Normalised to `"negative"` at
assembly; `category` carries the real distinction.

## Audit results (computed before any lexicon work)

### Purity — leave-one-out nearest centroid

| pool | accuracy |
|---|---|
| H (108) | 82.4% |
| A (765 scored, `confounded` excluded) | 65.4% |

Per domain in pool A, best to worst: `sleep_problems` 95%, `appetite_changes`
86%, `feelings_of_failure_or_guilt` 76%, `thoughts_of_death_or_self_harm` 75%,
`psychomotor_changes` 64%, `concentration_problems` 59%, `lack_of_pleasure` 53%,
`depressed_mood` 52%, **`low_energy` 28%**.

**Purity must not be used to filter exemplars.** Discriminative items name the
neighbouring domain in order to rule it out, so an embedding pulls them toward
it by construction. Split by grade the overall effect is small (discriminative
63.9% vs consistent 66.7%), but it is severe exactly where predicted:
`low_energy` scores **15% discriminative vs 33% consistent**, because its
discriminative form is "shattered *even after a full night's sleep*". Filtering
on purity would delete the most valuable exemplars first.

What the matrix does measure is how much of PHQ-9's structure survives at
sentence level — a ceiling on any detector, set by the instrument. The same
confusions appear independently in three places: the human pool, the generated
pool, and the nine blind agents' own second-pass trim notes
(`depressed_mood`↔`feelings_of_failure_or_guilt`, and the fatigue cluster).

### Overlap — per-domain nearest neighbour, cosine

| direction | mean | ≥0.6 |
|---|---|---|
| H → A | 0.670 | 67% |
| A → H | 0.460 | 18% |
| A → B | 0.503 | 19% |
| B → A | 0.488 | 16% |
| B → H | 0.346 | 2% |

### Saturation: the space is not covered at 100/domain

Fraction of pool B with no counterpart in pool A above 0.6:

| domain | mean nn | novel |
|---|---|---|
| `depressed_mood` | 0.490 | **84%** |
| `lack_of_pleasure` | 0.480 | **86%** |
| `low_energy` | 0.496 | **81%** |

Two independent 100-exemplar passes over the same symptom share roughly a sixth
of their phrasing. For all-MiniLM-L6-v2 a mean nearest-neighbour cosine near
0.49 means "same topic, different wording" — the passes agree on the concept and
disagree on how to say it.

**Consequences, and they are the main result of this exercise:**

1. Any finite lexicon is necessarily narrow. This is not a curation failure that
   more effort fixes; the space is simply much larger than a hand-written list.
2. **Coverage measured against these pools is a LOWER BOUND** on expressive
   range, and must be reported as one. "Of N a-priori phrasings, M appear in the
   corpus" understates the model's range, because N is not the space.
3. It argues for the embedding measure over the lexicon where a choice exists:
   one generalises, the other enumerates. It also shows how badly
   under-specified a *single* reference sentence per symptom was.
4. Pooling A+B is better than either alone. For the three domains where both
   exist, use the union as the reference set.
