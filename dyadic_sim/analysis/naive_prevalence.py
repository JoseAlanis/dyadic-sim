"""Naive symptom prevalence: two panels, PHQ-9 and MADRS.

Answers the plainest possible version of the question. If we asked whether each
PHQ-9 item or MADRS topic shows up in this conversation, using nothing but
vocabulary the patient actually used, how often would we say yes?

Rows are the injected condition, including the `no_symptoms` control. Columns
are the detected symptom. A cell is the fraction of sessions in that condition
where the detector fired. So the **control row is the base rate**, printed in
the same units as everything else, and the **diagonal** is whether injecting a
symptom makes the patient talk about it.

Read the control row first. Some lexicons fire on most sessions regardless of
what was injected, and for those the column carries no information at all - a
broad one like `depressed_mood` can fire in the majority of sessions where
nothing was injected, simply because distress vocabulary is everywhere in a
therapy transcript. Detection only means something where the base rate is low.

Detection is `dyadic_sim.analysis.lexicon_detect`: patient speech only,
negation-scoped, plus therapist-question-with-referential-answer. Every firing
keeps the word that matched and the sentence it came from; `--evidence` dumps
them so any cell can be audited by hand.

    dyadic-sim naive-prevalence \
        --sessions 'data/sessions/session_*' --prefix <prefix> \
        --out-dir data/results/<prefix>

This is the floor, not the measure. Its mean lift over control is small, and
for some symptoms negative - the case narrative produces the symptom whether
or not it was injected. That is a result worth reporting, not a bug to tune
away.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from dyadic_sim.analysis import provenance
from dyadic_sim.analysis.heatmap import DIVERGING, draw_panels
from dyadic_sim.analysis.lexicon_detect import detect_session, detect_therapist
from dyadic_sim.analysis.quality import flagged_turn_indices, qc_fields
from dyadic_sim.analysis.sessions import iter_sessions, patient_turns
from dyadic_sim.analysis.symptom_lexicon import MADRS_PATTERNS, PHQ9_PATTERNS
from dyadic_sim.core.paths import ACTIVE_PREFIX, ACTIVE_RESULTS, ACTIVE_SESSIONS
from dyadic_sim.generation.utterance import clean_utterance
from dyadic_sim.scoring.instruments.madrs import PHQ_TO_TOPIC
from dyadic_sim.scoring.transcript_parser import build_turn_pairs, load_session_source

PANELS = {"PHQ-9": PHQ9_PATTERNS, "MADRS": MADRS_PATTERNS}

# Default = the active corpus, matching symptom_embed.py. The pairing with
# ACTIVE_RESULTS is the point; see the note on ACTIVE_* in paths.py.
DEFAULT_SESSIONS = [ACTIVE_SESSIONS]

# Conditions are named by PHQ-9 key, so the MADRS panel has no literal diagonal.
# The instrument records which topic approximates each PHQ anchor; use it so
# "did injecting X show up as X" is answerable in both panels.
# psychomotor_changes maps to None: MADRS has no corresponding topic.
TARGET_OF = {
    "PHQ-9": {s: s for s in PHQ9_PATTERNS},
    "MADRS": {phq: topic.replace("\u00fc", "ue") for phq, topic in PHQ_TO_TOPIC.items() if topic},
}


# MADRS topic keys are the German labels MADRS-BERT was trained on, so they
# cannot be renamed (tests/test_symptom_lexicon.py pins them). But as column
# headers they mislead: "Gedanken" just means "thoughts", and it is the
# highest-firing MADRS column at 35%, so it reads as the ideation row when it is
# actually pessimism and guilt. The ideation row is `Suizid`, which fires ~0%.
# Display-only gloss; the keys are untouched.
DISPLAY = {
    "Traurigkeit": "Traurigkeit (sadness)",
    "Anspannung": "Anspannung (tension/anx)",
    "Schlaf": "Schlaf (sleep)",
    "Appetit": "Appetit (appetite)",
    "Konzentration": "Konzentration (concentr.)",
    "Antriebslosigkeit": "Antriebslosigkeit (drive)",
    "Gefuehlslosigkeit": "Gefuehlslosigkeit (numbness)",
    "Gedanken": "Gedanken (pessimism/guilt)",
    "Suizid": "Suizid (ideation)",
}


def _glossed(grid):
    """Rename for display only, so a reader cannot mistake Gedanken for Suizid."""
    if grid is None or grid.empty:
        return grid
    return grid.rename(columns=DISPLAY, index=DISPLAY)


def _clean_pairs(pairs):
    """Strip stage directions and role labels from both speakers.

    The lexicon reads raw transcript text, so on generation_version 1 sessions it
    matches inside narration rather than speech: "(shrugs hopelessly)" is a
    `Traurigkeit` hit made entirely of stage direction. Sessions generated with
    the speech-only prompt are clean already, so this matters mainly for older
    transcripts - but without it this module could not be run on the same text
    as symptom_embed.py, which has had the flag all along.
    """
    return [
        replace(pair,
                patient_text=clean_utterance(pair.patient_text or ""),
                therapist_text=clean_utterance(pair.therapist_text or ""))
        for pair in pairs
    ]


def scan(session_globs, prefix, min_turns, clean=False, qc_view="all",
         confirmations=True, statuses=None):
    """One row per (session, panel, symptom).

    Also returns the session directories actually consumed, so provenance
    can fingerprint the exact inputs rather than counting them.
    """
    rows, evidence, used_dirs = [], [], []
    kwargs = {} if statuses is None else {"statuses": statuses}
    for directory, meta, run_prefix, case, symptom, _run in iter_sessions(
            session_globs, **kwargs):
        if prefix and run_prefix != prefix:
            continue
        used_dirs.append(directory)
        # QC on the patient turns, whatever the detector consumes, so a flagged
        # turn stays identifiable in the evidence trail. Never dropped silently:
        # view 1 keeps everything and reports rates, view 2 removes flagged
        # turns, view 3 excludes flagged sessions.
        qc = qc_fields(patient_turns(directory, clean=clean), meta)
        flagged = flagged_turn_indices(qc)
        if qc_view == "drop-flagged-sessions" and flagged:
            continue
        pairs = build_turn_pairs(load_session_source(directory))
        if clean:
            pairs = _clean_pairs(pairs)
        if qc_view == "drop-flagged-turns" and flagged:
            pairs = [p for p in pairs if p.turn_index not in flagged]
        for panel, patterns in PANELS.items():
            # Both speakers, same lexicon and same negation scoping. "patient"
            # is expression; "therapist" is uptake, i.e. whether the injected
            # symptom propagates into the clinician's speech.
            for speaker, found in (("patient", detect_session(
                                       pairs, patterns,
                                       count_confirmations=confirmations)),
                                   ("therapist", detect_therapist(pairs, patterns))):
                for name, detection in found.items():
                    rows.append(
                        {
                            "session_id": meta.get("session_id"),
                            "model": meta.get("patient_model", "?"),
                            "patient_model": meta.get("patient_model", "?"),
                            "therapist_model": meta.get("therapist_model", "?"),
                            "case": case,
                            "injected": symptom,
                            "panel": panel,
                            "speaker": speaker,
                            "detected_symptom": name,
                            "detected": bool(detection.detected(min_turns)),
                            "n_turns": len(detection.turns),
                            "n_negated": len(detection.negated_hits),
                            # Did a flagged turn supply the evidence? Session-level
                            # flag rates cannot answer that, and it is the version
                            # that matters for whether a detection is trustworthy.
                            "evidence_from_flagged_turn": bool(
                                flagged & {h.turn for h in detection.hits}),
                            **qc,
                        }
                    )
                    for hit in detection.hits:
                        evidence.append(
                            {
                                "session_id": meta.get("session_id"),
                                "case": case,
                                "injected": symptom,
                                "panel": panel,
                                "speaker": speaker,
                                "detected_symptom": name,
                                "turn": hit.turn,
                                "matched": hit.matched,
                                "source": hit.source,
                                "turn_flagged": hit.turn in flagged,
                                "evidence": hit.evidence,
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(evidence), used_dirs


def _order_rows(grid):
    """Rows are conditions, named by PHQ-9 key in both panels, so order them by
    the PHQ-9 list. Ordering by the panel's own labels desynchronises the two
    panels and makes them impossible to read side by side."""
    conditions = list(PANELS["PHQ-9"])
    order = [r for r in conditions if r in grid.index]
    order += [r for r in grid.index if r not in order and r != "no_symptoms"]
    if "no_symptoms" in grid.index:
        order = ["no_symptoms"] + order
    return grid.reindex(order)


def matrix(df, panel, speaker="patient"):
    sub = df[(df.panel == panel) & (df.speaker == speaker)]
    grid = sub.pivot_table(
        index="injected", columns="detected_symptom", values="detected", aggfunc="mean"
    )
    grid = grid.reindex(columns=[s for s in PANELS[panel] if s in grid.columns])
    return _order_rows(grid)


def contrast(df, panel, speaker="patient"):
    """Condition rate minus that case's OWN control rate, then pooled.

    Subtracting the pooled control row would be wrong: cases differ enormously
    in what they produce with nothing injected (`Anspannung` is 100% in
    afraid_of_dogs and 40% in empty_and_invisible), so a pooled baseline is an
    average of incomparable things. Here each case is differenced against its
    own control first and the differences are averaged, which is the same
    matched-control logic as `target_z_vs_control` in symptom_embed.py.

    Cases with no control sessions are dropped: there is nothing to subtract.
    """
    sub = df[(df.panel == panel) & (df.speaker == speaker)]
    per_case = []
    for _case, group in sub.groupby("case"):
        grid = group.pivot_table(
            index="injected", columns="detected_symptom", values="detected", aggfunc="mean"
        )
        if "no_symptoms" not in grid.index:
            continue
        per_case.append(grid.subtract(grid.loc["no_symptoms"], axis=1))
    if not per_case:
        return pd.DataFrame()
    grid = pd.concat(per_case).groupby(level=0).mean()
    grid = grid.drop(index="no_symptoms", errors="ignore")  # 0 by construction
    grid = grid.reindex(columns=[s for s in PANELS[panel] if s in grid.columns])
    return _order_rows(grid)


def base_rates(df, panel, speaker="patient"):
    """Detection rate per case, using control sessions only.

    One row per case, computed over `no_symptoms` sessions, so the whole grid
    is base rate with nothing injected. This is the flooding view: a case whose
    theme already supplies the vocabulary lights up here, and every cell in that
    case's condition matrix has to be read against its own row rather than
    against the pooled control.
    """
    sub = df[(df.panel == panel) & (df.injected == "no_symptoms") & (df.speaker == speaker)]
    if sub.empty:
        return pd.DataFrame()
    grid = sub.pivot_table(
        index="case", columns="detected_symptom", values="detected", aggfunc="mean"
    )
    return grid.reindex(columns=[s for s in PANELS[panel] if s in grid.columns])


SPEAKER_QUESTION = {
    "patient": "does the patient's own vocabulary show the symptom?",
    "therapist": "does the THERAPIST take up the symptom's vocabulary?",
}


def write_figures(df, out_dir, title_suffix, by_case=False):
    """Every figure, in one format. Returns the paths written."""
    written = []
    n_cases = df[df.injected == "no_symptoms"].case.nunique()

    for speaker, question in SPEAKER_QUESTION.items():
        tag = "" if speaker == "patient" else "_therapist"

        # 1. Raw rate per condition.
        written.append(draw_panels(
            PANELS,
            lambda panel, sp=speaker: _glossed(matrix(df, panel, sp)),
            os.path.join(out_dir, f"fig_naive_prevalence{tag}.png"),
            targets_of=lambda panel: {k: DISPLAY.get(v, v) for k, v in TARGET_OF[panel].items()},
            title=f"Naive prevalence: {question}  {title_suffix}\n"
                  "Dashed line marks the control row (base rate); blue box marks "
                  "the injected symptom's own cell",
            ylabel="condition (injected symptom)",
            bar_label="fraction of sessions where detected",
        ))

        # 2. Base rate per case, controls only. The flooding view.
        if n_cases:
            written.append(draw_panels(
                PANELS,
                lambda panel, sp=speaker: _glossed(base_rates(df, panel, sp)),
                os.path.join(out_dir, f"fig_naive_prevalence{tag}_base_rate.png"),
                title=f"Base rate by case ({speaker}): what the theme alone produces, "
                      f"nothing injected.  {title_suffix}\n"
                      "A hot row means that case floods those symptoms; its condition "
                      "cells must be read against this row, not the pooled control",
                ylabel="case",
                bar_label="control-session detection rate",
                height=3.2 + 0.75 * n_cases,
                mark_control=False,
            ))

            # 3. Condition minus that case's own control. The measure to read.
            written.append(draw_panels(
                PANELS,
                lambda panel, sp=speaker: _glossed(contrast(df, panel, sp)),
                os.path.join(out_dir, f"fig_naive_prevalence{tag}_vs_control.png"),
                targets_of=lambda panel: {k: DISPLAY.get(v, v) for k, v in TARGET_OF[panel].items()},
                title=f"Lift over matched control ({speaker}): {question}  {title_suffix}\n"
                      "Each case differenced against its OWN control before pooling. "
                      "0 = no more than that case produces with nothing injected",
                ylabel="condition (injected symptom)",
                bar_label="detection rate minus matched control",
                cmap=DIVERGING, vmin=-0.5, vmax=0.5,
                mark_control=False,
            ))

    if by_case:
        for case in sorted(df.case.dropna().unique()):
            subset = df[df.case == case]
            written.append(draw_panels(
                PANELS,
                lambda panel, sub=subset: _glossed(matrix(sub, panel, "patient")),
                os.path.join(out_dir, f"fig_naive_prevalence_{case}.png"),
                targets_of=lambda panel: {k: DISPLAY.get(v, v) for k, v in TARGET_OF[panel].items()},
                title=f"Naive prevalence: {SPEAKER_QUESTION['patient']}  "
                      f"({case}, n={subset.session_id.nunique()})\n"
                      "Dashed line marks the control row; blue box marks the "
                      "injected symptom's own cell",
                ylabel="condition (injected symptom)",
                bar_label="fraction of sessions where detected",
            ))
    return [w for w in written if w]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Defaults must agree: a bare run reads the active corpus and writes to that
    # corpus's results directory. When they disagreed - reading one corpus while
    # writing to another corpus's results directory - a bare run with no
    # arguments silently replaced a finished result set. Repointing one side
    # without the other reintroduces exactly that.
    parser.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS,
                        help="One or more globs of session directories.")
    parser.add_argument("--prefix", default=ACTIVE_PREFIX,
                        help="Only include sessions whose case_name prefix matches. "
                             "Defaults to the active corpus rather than to 'all', so "
                             "that adding a second arm under a new prefix cannot get "
                             "pooled into this one by accident. Pass --prefix '' for "
                             "every session under --sessions.")
    parser.add_argument("--out-dir", default=ACTIVE_RESULTS,
                        help="Where to write CSVs + figures.")
    parser.add_argument(
        "--min-turns",
        type=int,
        default=1,
        help="Turns a symptom must appear in. Default 1, i.e. no recurrence gate; "
             "gating costs more true positives than it removes false ones.",
    )
    parser.add_argument("--evidence", action="store_true",
                        help="Also write every firing with its matched word and sentence")
    parser.add_argument("--clean-text", action="store_true",
                        help="Strip stage directions and role labels before matching. "
                             "Match whatever symptom_embed.py was run with, or the two "
                             "measures describe different text. Only affects "
                             "generation_version 1 corpora; v3 is clean at generation.")
    parser.add_argument(
        "--no-confirmations", action="store_true",
        help="Ignore therapist-question-plus-referential-answer evidence, "
             "counting only what the patient said unprompted. Reported as a "
             "declared sensitivity: confirmation infers WHICH symptom from the "
             "therapist's question, so it is the path most able to turn a "
             "clinician's suggestion into a patient detection.",
    )
    parser.add_argument(
        "--qc-view",
        choices=["all", "drop-flagged-turns", "drop-flagged-sessions"],
        default="all",
        help="Predefined quality views, declared rather than chosen after "
             "looking. 'all' (default) analyses every valid session and reports "
             "flag rates; 'drop-flagged-turns' keeps sessions but removes "
             "role-confused and annotation-only turns; 'drop-flagged-sessions' "
             "excludes any session containing one. Flagged content is never "
             "removed silently under any view.",
    )
    parser.add_argument("--by-case", action="store_true",
                        help="Also write one condition matrix per case. Pooling across "
                             "cases lets a flooded case set the base rate for all of them.")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df, evidence, used_dirs = scan(args.sessions, args.prefix, args.min_turns,
                                   clean=args.clean_text, qc_view=args.qc_view,
                                   confirmations=not args.no_confirmations)
    if df.empty:
        raise SystemExit(f"No sessions matched: {args.sessions} prefix={args.prefix}")

    # Provenance FIRST, before a single artifact is written. It refuses to
    # describe a run incompatible with what is already in this directory, and
    # that refusal is only protection if nothing has been overwritten yet. It
    # used to be called at the end, after the CSVs had already been replaced.
    provenance.write(args.out_dir, script=os.path.basename(__file__), args=args,
                     inputs={"n_sessions": int(df.session_id.nunique())},
                     session_dirs=used_dirs)

    long_path = os.path.join(args.out_dir, "naive_prevalence_long.csv")
    df.to_csv(long_path, index=False)
    if args.evidence and not evidence.empty:
        evidence.to_csv(os.path.join(args.out_dir, "naive_prevalence_evidence.csv"), index=False)

    n_sessions = df.session_id.nunique()
    for panel in PANELS:
        grid = matrix(df, panel)
        if grid.empty:
            continue
        print(f"\n=== {panel} | {n_sessions} sessions ===")
        print((grid * 100).round(0).astype("Int64").to_string())
        if "no_symptoms" in grid.index:
            base = grid.loc["no_symptoms"]
            targets = TARGET_OF[panel]
            lifts = {}
            for condition in grid.index:
                target = targets.get(condition)
                if condition == "no_symptoms" or not target or target not in grid.columns:
                    continue
                lifts[condition] = grid.loc[condition, target] - base[target]
            if lifts:
                print("\n  lift over control for the symptom actually injected:")
                for condition, value in sorted(lifts.items(), key=lambda kv: -kv[1]):
                    print(f"    {condition:34s} -> {targets[condition]:20s} {value:+.0%}")
                print(f"    {'MEAN':34s}    {'':20s} {np.mean(list(lifts.values())):+.0%}")
                unmapped = [c for c in grid.index
                            if c != "no_symptoms" and not targets.get(c)]
                if unmapped:
                    print(f"    (no {panel} topic for: {', '.join(unmapped)})")

    # Base rate per case, control sessions only. Read this before the matrices:
    # a case whose theme already supplies the vocabulary inflates every cell in
    # its own block, and pooling hides that.
    for panel in PANELS:
        grid = base_rates(df, panel)
        if grid.empty:
            continue
        print(f"\n=== {panel} | base rate by case (no_symptoms sessions only) ===")
        print((grid * 100).round(0).astype("Int64").to_string())
        spread = grid.max() - grid.min()
        worst = spread.sort_values(ascending=False).head(3)
        if len(grid.index) > 1 and worst.iloc[0] > 0:
            print("  widest between-case spread: " + ", ".join(
                f"{s} {v:.0%}" for s, v in worst.items() if v > 0))

    # Therapist uptake, raw and matched-control. Read the contrast: a therapist
    # who asks about sleep in every session produces a high raw rate that says
    # nothing about whether the injection reached them.
    for panel in PANELS:
        grid = contrast(df, panel, "therapist")
        if grid.empty:
            continue
        targets = TARGET_OF[panel]
        lifts = {c: grid.loc[c, targets[c]] for c in grid.index
                 if targets.get(c) in grid.columns}
        if not lifts:
            continue
        print(f"\n=== {panel} | THERAPIST uptake of the injected symptom, vs matched control ===")
        for condition, value in sorted(lifts.items(), key=lambda kv: -kv[1]):
            print(f"    {condition:34s} -> {targets[condition]:20s} {value:+.0%}")
        print(f"    {'MEAN':34s}    {'':20s} {np.mean(list(lifts.values())):+.0%}")

    # Provenance beside the result: every setting below has been shown to
    # move numbers, so a CSV without them cannot be compared to anything.
    written = [long_path]
    written += write_figures(df, args.out_dir, f"({args.prefix or 'all'}, n={n_sessions})",
                             by_case=args.by_case)
    print("\n" + "\n".join(f"wrote {p}" for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
