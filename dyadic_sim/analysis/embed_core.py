"""The embedding measure's shared core: references, scoring, and the corpus loop.

Extracted so `symptom_embed.py` and `symptom_embed_robust.py` cannot diverge.
They had, badly: the robustness check concatenated whole sessions and embedded
them directly, which meant it was validating a *truncated opening-only* measure
while claiming to test the turn-level one. A robustness analysis must vary only
its declared dimension - here, the reference wording - and share everything
else.

What lives here is everything both need: how references become one score per
symptom, how a session becomes text units, how units collapse, and what quality
context travels with each row. What stays in the scripts is what genuinely
differs: which reference sets to compare, and how to present the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dyadic_sim.analysis.quality import filter_turns, flagged_turn_indices, qc_fields
from dyadic_sim.analysis.sessions import (
    BOUNDED_UNITS,
    iter_sessions,
    n_tokens,
    patient_turns,
    text_units,
)


def build_references(kind: str, symptoms: list[str], single: dict) -> dict:
    """`single` or `expanded` reference sets, keyed by symptom.

    `single` is the one hand-written sentence per symptom that produced every
    published number. `expanded` is the a-priori exemplar pool. Both are
    returned as tuples so callers need not care which they were given.
    """
    if kind == "single":
        return {s: (single[s],) for s in symptoms}
    from dyadic_sim.analysis.exemplars import reference_sets

    expanded = reference_sets()
    missing = [s for s in symptoms if not expanded.get(s)]
    if missing:
        raise SystemExit(
            f"--references expanded, but no exemplars for: {', '.join(missing)}. "
            f"Run the generation step or use --references single."
        )
    return {s: expanded[s] for s in symptoms}


class ReferenceScorer:
    """Turns texts into one similarity per symptom, under a chosen aggregation.

    `mean` treats a symptom as a region and is unbiased by how many references
    it has. `max` asks whether the text matches any single phrasing, which
    saturates once the set is large - with 85 diverse probes almost any therapy
    text matches one well, raising the floor in control sessions too. `mean` is
    the default for both reasons.
    """

    def __init__(self, model, refs: dict, agg: str = "mean"):
        self.model = model
        self.symptoms = list(refs)
        self.refs = refs
        self.agg = agg
        flat = [text for s in self.symptoms for text in refs[s]]
        self.owner = np.array([k for k, s in enumerate(self.symptoms) for _ in refs[s]])
        self.n_references = len(flat)
        self.ref_emb = model.encode(flat, normalize_embeddings=True, batch_size=64,
                                    show_progress_bar=False)

    def describe(self) -> str:
        sizes = [len(self.refs[s]) for s in self.symptoms]
        return (f"{self.n_references} sentences, {min(sizes)}-{max(sizes)} per symptom, "
                f"agg={self.agg}")

    def score(self, texts: list[str]) -> np.ndarray:
        raw = self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                show_progress_bar=False) @ self.ref_emb.T
        out = np.empty((raw.shape[0], len(self.symptoms)))
        reduce = np.max if self.agg == "max" else np.mean
        for k in range(len(self.symptoms)):
            out[:, k] = reduce(raw[:, self.owner == k], axis=1)
        return out


def collapse_units(per_unit: np.ndarray, unit_agg: str) -> tuple[np.ndarray, np.ndarray]:
    """(scores, winning-unit indices) for one session.

    `max` asks "was it said anywhere" and is historical, but a maximum grows
    with the number of units, so an arm with longer sessions scores higher for
    that reason alone. `top2`/`top3` keep the question while damping that.
    """
    order = np.argsort(-per_unit, axis=0)
    k = (per_unit.shape[0] if unit_agg == "mean"
         else 1 if unit_agg == "max"
         else min(int(unit_agg[3:]), per_unit.shape[0]))
    scores = np.take_along_axis(per_unit, order[:k], axis=0).mean(axis=0)
    return scores, order[0]


def score_corpus(scorer: ReferenceScorer, session_globs, *, prefix=None,
                 unit="bounded-turn", clean=False, unit_agg="max",
                 qc_view="all", statuses=None) -> dict:
    """Score every analysable session. Returns rows, inputs and diagnostics.

    One loop for both scripts, so text units, cleaning, QC and unit counting
    cannot drift between the measure and its own robustness check.
    """
    model = scorer.model
    symptoms = scorer.symptoms
    rows, used_dirs, excluded = [], [], []
    unit_counts, over_limit = [], []

    kwargs = {"excluded": excluded}
    if statuses is not None:
        kwargs["statuses"] = statuses
    for d, meta, run_prefix, case, symptom, run in iter_sessions(
            session_globs, **kwargs):
        if prefix and run_prefix != prefix:
            continue
        used_dirs.append(d)
        qc = qc_fields(patient_turns(d, clean=clean), meta)
        if qc_view == "drop-flagged-sessions" and (
                qc["any_role_confusion"] or qc["any_annotation_only"]):
            continue
        units = text_units(d, unit=unit, clean=clean, model=model)
        if qc_view == "drop-flagged-turns":
            units = filter_turns(units, flagged_turn_indices(qc))

        row = {"model": meta.get("patient_model", "?"),
               "patient_model": meta.get("patient_model", "?"),
               "therapist_model": meta.get("therapist_model", "?"),
               "case": case, "symptom": symptom, "run": run,
               "session_id": meta.get("session_id"), **qc}

        if not units:
            row.update({f"sim_{s}": np.nan for s in symptoms})
            row["n_units"] = 0
            row["n_units_over_limit"] = 0
            rows.append(row)
            unit_counts.append(0)
            over_limit.append(0)
            continue

        sizes = [n_tokens(model, t) for _, t in units]
        n_over = sum(1 for s in sizes if s > model.max_seq_length)
        unit_counts.append(len(units))
        over_limit.append(n_over)

        per_unit = scorer.score([t for _, t in units])
        scores, winners = collapse_units(per_unit, unit_agg)
        for k, s in enumerate(symptoms):
            row[f"sim_{s}"] = float(scores[k])
            row[f"turn_{s}"] = units[winners[k]][0]
        row["n_units"] = len(units)
        row["n_units_over_limit"] = n_over
        # Session-level flag rates cannot answer whether the flagged turn is the
        # one supplying the score. This can.
        row["max_from_flagged_turn"] = bool(
            {units[w][0] for w in winners} & flagged_turn_indices(qc))
        rows.append(row)

    return {
        "rows": rows,
        "used_dirs": used_dirs,
        "excluded": excluded,
        "unit_counts": unit_counts,
        "over_limit": over_limit,
        "total_over_limit": sum(over_limit),
        "total_units": sum(unit_counts),
    }


def report_coverage(result: dict, unit: str, max_seq_length: int) -> None:
    """Refuse silent truncation; warn loudly where it is a reproduction mode."""
    total_over = result["total_over_limit"]
    if not total_over:
        return
    message = (f"{total_over} of {result['total_units']} embedding inputs exceed "
               f"the model's {max_seq_length}-token limit and would be silently "
               f"truncated")
    if unit in BOUNDED_UNITS:
        raise SystemExit(f"{message}. This is a bug in the chunker, not a "
                         f"tolerable condition for --unit {unit}.")
    print(f"  WARNING: {message}. --unit {unit} is a reproduction mode; use "
          f"--unit bounded-turn for guaranteed coverage.")


def add_control_contrast(df, symptoms) -> pd.DataFrame:
    """Lever E: score each injected session against its own case's control.

    `target_z` asks whether the target outranks its 8 siblings *within* one
    session. It cannot tell a working manipulation from a case that already
    leans that way, and cases lean hard. A dog-phobia case produces tension talk
    in essentially every control session, because the case IS an anxiety case; a
    case written to present with "just feeling off" produces emotional numbness
    in every control session, because that is what "feeling off" describes. In
    both, the symptom is there before anything is injected.

    So subtract it. For each session we already have a z per symptom; the
    control mean for the same (case, model) at that same symptom is the lean the
    theme supplies on its own, and what is left is attributable to the
    injection. NaN where a case has no control sessions, which is honest: the
    contrast is not computable there rather than zero.
    """
    controls = df[df.symptom == "no_symptoms"]
    if controls.empty:
        df["target_z_vs_control"] = np.nan
        df["control_z"] = np.nan
        return df

    # (case, model) -> mean z per symptom over that cell's control sessions.
    baselines = controls.groupby(["case", "model"])[[f"z_{s}" for s in symptoms]].mean()

    def lookup(row):
        if row.symptom not in symptoms:
            return np.nan
        try:
            return baselines.loc[(row.case, row.model), f"z_{row.symptom}"]
        except KeyError:
            return np.nan

    df["control_z"] = df.apply(lookup, axis=1)
    df["target_z_vs_control"] = df["target_z"] - df["control_z"]
    return df


BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 0


def summarise(df, column, model, symptom):
    """Mean and 95% half-width for one (model, symptom) cell.

    ONE estimator, used by the printed table, the CSV summary and the figure.
    They previously disagreed: `add_control_contrast` subtracts each case's own
    control and the printed mean averaged those per-session contrasts, while this
    function recomputed a single pooled injected mean minus a single pooled
    control mean. Those are different estimands whenever cases have unequal
    control counts - one case with six controls against another with five is
    enough. The two estimators drifted apart by a margin comparable to the effect
    being reported, which is how the disagreement was noticed.

    The estimand is now declared: **one contrast per case, pooled with equal
    weights.**

    Scope, stated because the implementation cannot support more: only SESSIONS
    are resampled, so the interval is conditional on the cases actually
    observed. It does NOT estimate generalisation to new patient cases, and
    calling cases "the unit of generalisation" would overstate it - with a
    single case and a single model pair, case-level generalisation is not
    estimable at all. Generalising over cases needs many more of them and a
    case-level resampling or hierarchical model.

    The interval is a bootstrap resampling whole SESSIONS within case and
    condition. Parametric SEs got this wrong twice: sd/sqrt(n) over the contrast
    column treats the control mean as known, and the two-sample form ignores the
    stratified-by-case construction. Resampling the actual sampling unit avoids
    having to derive the right formula for a matched, unbalanced design.
    """
    empty = {"point": np.nan, "lo": np.nan, "hi": np.nan, "n_cases": 0,
             "n_injected": 0, "n_control": 0, "mean_injected": np.nan,
             "mean_control": np.nan, "cohens_d": np.nan}
    sub = df[(df.model == model) & (df.symptom == symptom)]
    if sub.empty:
        return empty
    if column != "target_z_vs_control":
        vals = sub[column].dropna()
        if vals.empty:
            return empty
        half = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
        return {**empty, "point": float(vals.mean()),
                "lo": float(vals.mean() - half), "hi": float(vals.mean() + half),
                "n_injected": len(vals)}

    zcol = f"z_{symptom}"
    per_case = []
    for case in sub.case.unique():
        inj = sub[sub.case == case][zcol].dropna().to_numpy()
        ctl = df[(df.model == model) & (df.symptom == "no_symptoms")
                 & (df.case == case)][zcol].dropna().to_numpy()
        if len(inj) and len(ctl):
            per_case.append((inj, ctl))
    if not per_case:
        return empty

    point = float(np.mean([i.mean() - c.mean() for i, c in per_case]))
    all_inj = np.concatenate([i for i, _ in per_case])
    all_ctl = np.concatenate([c for _, c in per_case])
    pooled_sd = np.sqrt((all_inj.var(ddof=1) + all_ctl.var(ddof=1)) / 2)
    base = {
        "point": point, "lo": np.nan, "hi": np.nan,
        "n_cases": len(per_case),
        "n_injected": len(all_inj), "n_control": len(all_ctl),
        "mean_injected": float(all_inj.mean()), "mean_control": float(all_ctl.mean()),
        # Reported beside the interval because a p-value or an interval alone
        # says nothing about whether a difference is worth anything.
        "cohens_d": float(point / pooled_sd) if pooled_sd > 0 else np.nan,
    }
    if any(len(i) < 2 or len(c) < 2 for i, c in per_case):
        return base

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_N)
    for b in range(BOOTSTRAP_N):
        draws[b] = np.mean([
            rng.choice(i, len(i), replace=True).mean()
            - rng.choice(c, len(c), replace=True).mean()
            for i, c in per_case
        ])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # ACTUAL bounds, not a half-width. Collapsing a percentile interval to
    # +/-(hi-lo)/2 throws away its asymmetry and silently recentres it on the
    # point estimate, which a bootstrap interval need not be.
    return {**base, "lo": float(lo), "hi": float(hi)}
