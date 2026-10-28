"""
Reference-robustness check.

`symptom_embed.py` compares patient speech to hand-written reference sentences,
so its magnitudes depend on *our* wording. This repeats the identical analysis
under three reference sets and correlates the per-(model, symptom) contrast. If
the +/- signature survives a reference swap, it reflects the data rather than
the phrasing.

  A: hand-written sentences (the ones the measure actually uses)
  B: the EXACT PHQ-9 labels the simulation put into the patient prompt
  C: the same labels plus the "nearly every day" frequency clause

**Everything except the reference set is shared with the primary analysis.**
Session discovery and inclusion, text units and chunking, cleaning, matched
controls, unit and reference aggregation, QC fields, provenance and the summary
estimand all come from `embed_core`. That was not true before: this script
concatenated whole sessions and embedded them directly, which exceeds the
model's 256-token limit in every session, so it was silently validating a
truncated *opening-only* measure while claiming to test the turn-level one. It
also dropped control sessions, computed no matched-control contrast, and wrote
no provenance. A robustness analysis must vary only its declared dimension.

Outputs (in --out-dir):
  - symptom_embed_robust.csv                  per-session scores per refset
  - fig_symptom_embed_robust.png              A vs B vs C, with intervals
  - symptom_embed_robust.analysis_config.json provenance
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dyadic_sim.analysis import provenance
from dyadic_sim.analysis.embed_core import (
    ReferenceScorer,
    add_control_contrast,
    report_coverage,
    score_corpus,
    summarise,
)
from dyadic_sim.analysis.embeddings import EMBEDDINGS_AVAILABLE, get_model
from dyadic_sim.analysis.sessions import UNITS
from dyadic_sim.core.paths import ACTIVE_PREFIX, ACTIVE_RESULTS, ACTIVE_SESSIONS
from dyadic_sim.core.vocabulary import LABELS, REFERENCES

DEFAULT_SESSIONS = [ACTIVE_SESSIONS]

# Imported, never restated. This script exists to validate the measure
# symptom_embed.py produces; a local copy that drifted from the shared
# REFERENCES would silently stop testing the real thing and still report
# agreement between the reference sets.
SYMPTOMS = list(REFERENCES)

_CLAUSE = "; over the last two weeks, you have been bothered by this symptom nearly every day"
REFSETS = {
    # A: straight from the measure under test.
    "A_handwritten": {s: (REFERENCES[s],) for s in SYMPTOMS},
    # B: the same object dyadic_sim.generation.priors.patient_prior renders, so "what the patient saw"
    # cannot go stale here.
    "B_phq9_label": {s: (LABELS[s],) for s in SYMPTOMS},
    # C: the full prompt line, frequency clause included.
    "C_phq9_full_line": {s: (LABELS[s] + _CLAUSE,) for s in SYMPTOMS},
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    ap.add_argument("--prefix", default=ACTIVE_PREFIX,
                    help="Only include sessions whose case_name prefix matches. Defaults "
                         "to the active corpus; pass --prefix '' for everything.")
    ap.add_argument("--out-dir", default=ACTIVE_RESULTS)
    ap.add_argument("--unit", choices=list(UNITS), default="bounded-turn",
                    help="Same units as the primary analysis. Defaults to "
                         "bounded-turn rather than the historical session, "
                         "which truncated every input.")
    ap.add_argument("--unit-agg", choices=["max", "top2", "top3", "mean"],
                    default="max")
    ap.add_argument("--qc-view", choices=["all", "drop-flagged-turns",
                                          "drop-flagged-sessions"], default="all")
    ap.add_argument("--clean-text", action="store_true",
                    help="Must match the setting used for symptom_embed.py, or "
                         "the two are not comparable.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not EMBEDDINGS_AVAILABLE:
        raise SystemExit("sentence-transformers not available")
    model = get_model()

    frames, used_dirs = [], []
    for name, refs in REFSETS.items():
        scorer = ReferenceScorer(model, refs, agg="mean")
        result = score_corpus(scorer, args.sessions, prefix=args.prefix,
                              unit=args.unit, clean=args.clean_text,
                              unit_agg=args.unit_agg, qc_view=args.qc_view)
        if not result["rows"]:
            raise SystemExit(f"No sessions matched: {args.sessions} prefix={args.prefix}")
        report_coverage(result, args.unit, model.max_seq_length)
        used_dirs = result["used_dirs"]

        df = pd.DataFrame(result["rows"])
        # Within-session z, then the matched-control contrast: the SAME
        # standardisation and the SAME estimand as the primary. Controls are
        # kept, not dropped, because the contrast is computed from them.
        sims = df[[f"sim_{s}" for s in SYMPTOMS]].to_numpy()
        mu, sd = sims.mean(axis=1, keepdims=True), sims.std(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(sd > 0, (sims - mu) / sd, np.nan)
        for k, s in enumerate(SYMPTOMS):
            df[f"z_{s}"] = z[:, k]
        df["target_z"] = [
            row[f"z_{row['symptom']}"] if row["symptom"] in SYMPTOMS else np.nan
            for _, row in df.iterrows()
        ]
        df = add_control_contrast(df, SYMPTOMS)
        df["refset"] = name
        frames.append(df)
        print(f"  {name}: {scorer.describe()}")

    df = pd.concat(frames, ignore_index=True)
    provenance.write(args.out_dir, script=os.path.basename(__file__), args=args,
                     inputs={"n_sessions": int(df.session_id.nunique()),
                             "refsets": list(REFSETS)},
                     session_dirs=used_dirs)
    out_csv = os.path.join(args.out_dir, "symptom_embed_robust.csv")
    df.to_csv(out_csv, index=False)

    # One estimator for every output, shared with the primary analysis.
    rows = []
    for name in REFSETS:
        sub = df[df.refset == name]
        for m in sorted(sub.model.unique()):
            for s in SYMPTOMS:
                if s not in set(sub.symptom):
                    continue
                st = summarise(sub, "target_z_vs_control", m, s)
                rows.append({"refset": name, "model": m, "symptom": s,
                             "contrast": st["point"], "lo": st["lo"],
                             "hi": st["hi"], "cohens_d": st["cohens_d"],
                             "n_injected": st["n_injected"],
                             "n_control": st["n_control"]})
    agg = pd.DataFrame(rows)

    print("\n== per-symptom contrast vs matched control, by reference set ==")
    wide = agg.pivot_table(index=["model", "symptom"], columns="refset", values="contrast")
    print(wide.to_string(float_format=lambda v: f"{v:+.3f}"))

    print("\n== agreement of the signature across reference wordings ==")
    for m in sorted(agg.model.unique()):
        w = wide.loc[m].dropna()
        if len(w) < 3:
            print(f"  {m}: only {len(w)} symptom(s); r not meaningful")
            continue
        names = list(REFSETS)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = np.corrcoef(w[names[i]], w[names[j]])[0, 1]
                print(f"  {m}: r({names[i][0]},{names[j][0]}) = {r:+.2f}")

    present = [s for s in SYMPTOMS if s in set(agg.symptom)]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.7 * len(present) + 1.5)))
    y = np.arange(len(present))
    offs = {"A_handwritten": 0.27, "B_phq9_label": 0.0, "C_phq9_full_line": -0.27}
    for name, off in offs.items():
        sub = (agg[agg.refset == name].groupby("symptom")[["contrast", "lo", "hi"]]
               .mean().reindex(present))
        point = sub["contrast"].to_numpy()
        err = np.vstack([np.nan_to_num(point - sub["lo"].to_numpy(), nan=0.0),
                         np.nan_to_num(sub["hi"].to_numpy() - point, nan=0.0)])
        ax.barh(y + off, point, height=0.26, xerr=err,
                error_kw={"ecolor": "0.3", "lw": 0.8, "capsize": 2}, label=name)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(present, fontsize=9)
    ax.set_xlabel("contrast vs matched control (pooled over models)")
    ax.set_title(f"Reference robustness, unit={args.unit}, agg={args.unit_agg}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_fig = os.path.join(args.out_dir, "fig_symptom_embed_robust.png")
    fig.savefig(out_fig, dpi=130)
    print(f"\nwrote {out_csv}\nwrote {out_fig}")


if __name__ == "__main__":
    main()
