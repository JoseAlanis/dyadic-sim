"""Rerun every corpus through the frozen spec and attribute every difference.

Runs the specification's primary configuration and each of its declared
sensitivities over every named corpus, then reports what changed and *why* -
each variant differs from the primary in exactly one parameter, so a difference
in the result has exactly one candidate cause.

The point is not to find the configuration where the effect looks best. It is to
show the whole surface, including the configurations where it looks worse, and
to let `spec.evaluate` deliver the verdict from the declared rule rather than
from whichever number is largest.

    PYTHONPATH=. python analysis/validation/rerun.py --prefixes <prefix> --out-dir data/results/<name>

Both datasets informed pipeline development. Their reanalysis is exploratory and
evaluates feasibility, robustness and measurement behaviour; it is not an
independent confirmatory test.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from analysis.embeddings import EMBEDDINGS_AVAILABLE, get_model
from analysis.validation import provenance
from analysis.validation import spec as spec_mod
from analysis.validation.embed_core import (
    ReferenceScorer,
    add_control_contrast,
    build_references,
    score_corpus,
    summarise,
)
from analysis.validation.naive_prevalence import scan
from analysis.validation.paths import ARCHIVE
from analysis.validation.sessions import COMPLETE, CRISIS_TERMINATED
from symptom_scoring.prior_vocabulary import REFERENCES

SYMPTOMS = list(REFERENCES)
SESSION_GLOB = ["data/sessions/session_*"]

#: What each varied parameter would explain if it moved the result.
ATTRIBUTION = {
    "unit": "token coverage",
    "unit_agg": "aggregation choice",
    "references": "reference choice",
    "reference_agg": "aggregation choice",
    "confirmations": "confirmation logic",
    "qc_view": "QC handling",
    "crisis_terminated": "data inclusion",
}


def configurations(spec: dict) -> list[dict]:
    """Primary plus one-parameter variants, so a change has one candidate cause."""
    base = dict(spec["parameters"])
    base["crisis_terminated"] = "include"
    out = [{"label": "primary", "varied": None, "params": base}]
    for sens in spec["sensitivities"]:
        key = sens["id"]
        for value in sens["values"]:
            if str(base.get(key)) == str(value):
                continue  # that is the primary
            params = dict(base)
            params[key] = value
            out.append({"label": f"{key}={value}", "varied": key, "params": params})
    return out


def _statuses(params):
    return ((COMPLETE,) if params["crisis_terminated"] == "exclude"
            else (COMPLETE, CRISIS_TERMINATED))


def lexicon_rates(prefix: str, params: dict) -> pd.DataFrame:
    """Detection rate per (injected arm, detected symptom), patient speech only."""
    df, _evidence, _dirs = scan(
        SESSION_GLOB, prefix, params["min_turns"],
        clean=params["clean_text"], qc_view=params["qc_view"],
        confirmations=params["confirmations"], statuses=_statuses(params),
    )
    p = df[(df.panel == "PHQ-9") & (df.speaker == "patient")]
    return p.pivot_table(index="injected", columns="detected_symptom",
                         values="detected", aggfunc="mean")


def embedding_frame(model, prefix: str, params: dict) -> pd.DataFrame:
    scorer = ReferenceScorer(
        model,
        build_references(params["references"], SYMPTOMS, REFERENCES),
        agg=params["reference_agg"],
    )
    result = score_corpus(
        scorer, SESSION_GLOB, prefix=prefix, unit=params["unit"],
        clean=params["clean_text"], unit_agg=params["unit_agg"],
        qc_view=params["qc_view"], statuses=_statuses(params),
    )
    df = pd.DataFrame(result["rows"])
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
    return add_control_contrast(df, SYMPTOMS)


def cell_stats(lex: pd.DataFrame, emb: pd.DataFrame, target: str) -> dict:
    """Everything the spec's criteria need for one injected symptom."""
    out = {"lexicon_lift_pp": None, "max_leak_pp": None,
           "embedding_contrast": None, "embedding_d": None,
           "embedding_lo": None, "embedding_hi": None}

    if target in lex.columns and "no_symptoms" in lex.index:
        control = lex.loc["no_symptoms", target]
        if target in lex.index:
            out["lexicon_lift_pp"] = 100 * (lex.loc[target, target] - control)
        # Specificity: the SAME detector in arms where something else was
        # injected. Absence of another arm is `None`, never a pass.
        others = [a for a in lex.index if a not in ("no_symptoms", target)]
        if others:
            out["max_leak_pp"] = 100 * max(lex.loc[a, target] - control for a in others)

    models = sorted(emb.model.dropna().unique())
    if models and target in set(emb.symptom):
        st = summarise(emb, "target_z_vs_control", models[0], target)
        out.update({"embedding_contrast": st["point"], "embedding_d": st["cohens_d"],
                    "embedding_lo": st["lo"], "embedding_hi": st["hi"]})
    return out


def historical_row(prefix: str, archive_root: str, target: str) -> dict | None:
    """The archived result, scored with the CURRENT estimand.

    Using today's estimator on yesterday's data isolates what the PIPELINE
    changed from what the ESTIMATOR changed: a like-for-like comparison with
    the archived numbers as originally reported.
    """
    path = os.path.join(archive_root, prefix, "symptom_embed_long.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "model" not in df.columns or target not in set(df.symptom):
        return None
    models = sorted(df.model.dropna().unique())
    st = summarise(df, "target_z_vs_control", models[0], target)
    return {"embedding_contrast": st["point"], "embedding_d": st["cohens_d"],
            "embedding_lo": st["lo"], "embedding_hi": st["hi"],
            "lexicon_lift_pp": None, "max_leak_pp": None}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prefixes", nargs="+", default=["power01"])
    ap.add_argument("--out-dir", default=os.path.join("data", "results", "tierC"))
    ap.add_argument("--archive-root",
                    default=os.path.join(ARCHIVE, "pre_pipeline_rebuild_2026-08-13"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not EMBEDDINGS_AVAILABLE:
        raise SystemExit("sentence-transformers not available")
    model = get_model()
    spec = spec_mod.load()
    configs = configurations(spec)

    print(f"spec {spec['version']} ({spec['status']}, {spec['dated']})  "
          f"{len(configs)} configurations x {len(args.prefixes)} corpora\n")

    rows = []
    for prefix in args.prefixes:
        for cfg in configs:
            params = cfg["params"]
            lex = lexicon_rates(prefix, params)
            emb = embedding_frame(model, prefix, params)
            injected = [a for a in lex.index if a != "no_symptoms"]
            for target in injected:
                stats = cell_stats(lex, emb, target)
                verdict = spec_mod.evaluate(
                    lexicon_lift_pp=stats["lexicon_lift_pp"],
                    embedding_d=stats["embedding_d"],
                    embedding_contrast=stats["embedding_contrast"],
                    max_leak_pp=stats["max_leak_pp"],
                    spec=spec,
                )
                rows.append({
                    "corpus": prefix, "config": cfg["label"],
                    "varied": cfg["varied"] or "-",
                    "attributes_to": ATTRIBUTION.get(cfg["varied"], "-"),
                    "symptom": target, **stats,
                    "verdict": verdict["verdict"],
                    "failed": ";".join(verdict["failed"]),
                })
            print(f"  {prefix:9s} {cfg['label']:28s} "
                  f"{len(injected)} injected arm(s)")

        hist = {}
        for target in {r["symptom"] for r in rows if r["corpus"] == prefix}:
            h = historical_row(prefix, args.archive_root, target)
            if h:
                hist[target] = h
        for target, h in hist.items():
            rows.append({"corpus": prefix, "config": "historical (archived)",
                         "varied": "-", "attributes_to": "pre-rebuild pipeline",
                         "symptom": target, **h, "verdict": "-", "failed": ""})

    df = pd.DataFrame(rows)
    provenance.write(args.out_dir, script=os.path.basename(__file__), args=args,
                     inputs={"n_configurations": len(configs),
                             "prefixes": args.prefixes},
                     extra={"analysis_spec": spec})
    out = os.path.join(args.out_dir, "tierC_comparison.csv")
    df.to_csv(out, index=False)

    for prefix in args.prefixes:
        sub = df[df.corpus == prefix]
        for target in sorted(sub.symptom.unique()):
            cell = sub[sub.symptom == target]
            print(f"\n=== {prefix} / {target} ===")
            show = cell[["config", "attributes_to", "lexicon_lift_pp",
                         "max_leak_pp", "embedding_contrast", "embedding_d",
                         "verdict"]].copy()
            print(show.to_string(index=False, na_rep="-",
                                 float_format=lambda v: f"{v:+.2f}"))
            primary = cell[cell.config == "primary"]
            if not primary.empty:
                base = primary.iloc[0]["verdict"]
                flips = cell[(cell.verdict != base) & (cell.verdict != "-")]
                if not flips.empty:
                    print(f"  ! verdict changes under: "
                          f"{', '.join(flips.config)} - report this, do not pick")
                else:
                    print(f"  verdict '{base}' is stable across all sensitivities")

    print(f"\nwrote {out}")
    print("\nBoth datasets informed pipeline development. This reanalysis is "
          "exploratory;\nit evaluates feasibility, robustness and measurement "
          "behaviour, and is not an\nindependent confirmatory test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
