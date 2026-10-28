"""
Embedding-based manipulation check (robust to lexicon breadth).

For each session we embed the patient's utterances and compare them to short
reference descriptions of each of the 9 PHQ-9 symptoms. The key quantity is a
*within-session* z-score: how much more similar is the patient's speech to the
INJECTED symptom than to the other 8 symptoms, in that same session. Within-
session standardisation controls for case, verbosity, and overall "clinical
tone", isolating the effect of the manipulation itself.

  target_z > 0  -> patient leans toward the injected symptom (manipulation works)
  target_z ~ 0  -> injected symptom is not preferentially expressed

Three variants are written, all per session, none replacing another, so a
result computed under one remains reproducible after another is added:

  target_z             the original. Target against all 9 references including
                       itself.
  target_z_loo         target against the OTHER 8. Including the target in its
                       own reference distribution raises both mu and sd when the
                       target is genuinely elevated, shrinking its own z, so the
                       bias is largest where the effect is largest, which
                       understates the headline in a known direction.
  target_z_vs_control  target_z minus the mean z for that same symptom over the
                       `no_symptoms` sessions of the same case and model. This
                       is the between-session contrast: it asks whether the
                       injection did anything beyond what the case leans toward
                       on its own. Read this one when cases differ in theme.
                       NaN where a case has no control sessions.

Control sessions are kept in the CSV rather than dropped, since the contrast is
computed from them.

`--unit` chooses what gets embedded, and it is a substantive choice, not a
performance knob. Two modes reproduce history and two guarantee coverage:

  session       ⚠ REPRODUCTION ONLY. Concatenates all patient turns into one
                string, which exceeds the model's 256-token limit in every
                session measured, and Sentence Transformers truncates silently.
                The score describes roughly the opening of the session, not the
                session. Retained only to reproduce numbers computed this way
                before the limit was noticed; those numbers are NOT
                whole-session measures. Calling this "dilution" would be
                wrong: later turns are discarded, not averaged.
  turn          ⚠ REPRODUCTION. One unit per turn, each symptom keeping its best,
                recorded in `turn_<symptom>` so a cell can be read back against
                the transcript. Much safer than `session` but not safe: a long
                turn still truncates.
  bounded-turn  As `turn`, but splits only the turns that overflow. The closest
                safe equivalent of historical turn scoring, and the recommended
                default for new work.
  chunk         Sentence-packed chunks over all patient text, ignoring turn
                boundaries.

Over-limit inputs are a hard error in the bounded modes (it would mean the
chunker is broken) and a loud warning in the reproduction modes.

**Coverage statistics are corpus-specific and are measured per run, not
asserted here.** Every run prints how many inputs exceeded the limit. An
earlier version of this docstring quoted median token counts and a retained
percentage as if they were general facts about the method; they were facts
about one corpus, and they were wrong for the next one by a factor of three.
Numbers baked into comments go stale silently, so measure them per run.

`--unit-agg` controls how a session's units collapse to one score. `max` is
historical and asks "was it said anywhere", but a maximum grows with the number
of units, so an arm with longer sessions scores higher for that reason alone.
The units-per-session table is printed for exactly this reason. Averaging the
top k attenuates a real effect rather than removing it, so a contrast that
survives `top2` and `top3` is not the artifact of a single peak unit - and one
that vanishes at `top2` was.

Respecting the token limit does NOT rescue the measure. Against the lexicon as
an independent yardstick, convergent validity is near zero whether the input is
truncated, per turn, or token-bounded - the three barely differ. The truncation
was a real bug; it was not the reason the measure is weak.

CIs and point estimates for `target_z_vs_control` come from `summarise()`, which
is the ONLY estimator any output should use: one contrast per case, pooled with
equal weights, with a bootstrap over whole sessions within case and condition.
Two earlier approaches were wrong - sd/sqrt(n) over the per-session contrast
column treats the control mean as known, and a pooled two-sample SE ignores the
matched-by-case construction. While the table and the figure used different
estimators they disagreed by a margin comparable to the effect being reported.

Usage:
  # the active corpus (default): data/sessions -> data/results/<ACTIVE_PREFIX>
  PYTHONPATH=. python analysis/validation/symptom_embed.py

  # spec-primary configuration, written explicitly:
  PYTHONPATH=. python analysis/validation/symptom_embed.py \
      --prefix <prefix> --unit bounded-turn --unit-agg max \
      --references expanded --reference-agg mean --clean-text

Outputs (in --out-dir):
  - symptom_embed_long.csv      per-session similarities + target_z
  - fig_symptom_embed.png       target_z by injected symptom x model
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.embeddings import EMBEDDINGS_AVAILABLE, get_model
from analysis.validation import provenance
from analysis.validation.embed_core import (
    ReferenceScorer,
    add_control_contrast,
    build_references,
    report_coverage,
    score_corpus,
    summarise,
)
from analysis.validation.heatmap import DIVERGING, draw_panels
from analysis.validation.paths import ACTIVE_PREFIX, ACTIVE_RESULTS, ACTIVE_SESSIONS
from analysis.validation.sessions import UNITS
from symptom_scoring.prior_vocabulary import REFERENCES

# Default = the active corpus, paired with ACTIVE_RESULTS as the output. Keep
# the two in agreement; see the note on ACTIVE_* in paths.py for what happens
# when they disagree.
DEFAULT_SESSIONS = [ACTIVE_SESSIONS]

# The reference sentences live with the rest of the PHQ-9 vocabulary. Imported
# rather than declared so this CLI script is not the source of truth for a
# constant that generation-side code and other analyses also need.
SYMPTOMS = list(REFERENCES)


def z_matrix(df, model, kind, symptoms=None):
    """condition x symptom mean z, as a matrix. `kind` picks the view.

    The bar figure only ever showed the diagonal — the injected symptom's own z.
    That hides where the signal actually goes. These matrices show all nine
    columns per row, so a condition that raises a NEIGHBOURING symptom instead of
    its own is visible rather than invisible.

      "rate"      mean z per (condition, symptom)
      "base"      mean z per (case, symptom), control sessions only
      "contrast"  condition minus that case's own control, then pooled

    Same three views, same order, as naive_prevalence.py, so the lexicon and the
    embedding can be read side by side.
    """
    symptoms = symptoms or SYMPTOMS
    zcols = [f"z_{s}" for s in symptoms]
    sub = df[df.model == model]
    if sub.empty:
        return pd.DataFrame()

    if kind == "base":
        ctl = sub[sub.symptom == "no_symptoms"]
        if ctl.empty:
            return pd.DataFrame()
        grid = ctl.groupby("case")[zcols].mean()
    elif kind == "rate":
        grid = sub.groupby("symptom")[zcols].mean()
    elif kind == "contrast":
        per_case = []
        for _case, g in sub.groupby("case"):
            if "no_symptoms" not in set(g.symptom):
                continue
            m = g.groupby("symptom")[zcols].mean()
            per_case.append(m.subtract(m.loc["no_symptoms"], axis=1))
        if not per_case:
            return pd.DataFrame()
        grid = pd.concat(per_case).groupby(level=0).mean().drop(index="no_symptoms",
                                                               errors="ignore")
    else:
        raise ValueError(kind)

    grid.columns = [c[2:] for c in grid.columns]
    if kind != "base":
        order = [s for s in symptoms if s in grid.index]
        order += [r for r in grid.index if r not in order and r != "no_symptoms"]
        if "no_symptoms" in grid.index:
            order = ["no_symptoms"] + order
        grid = grid.reindex(order)
    return grid


def write_heatmaps(df, out_dir, title_suffix, symptoms=None, by_case=False):
    """The three matrix views, one panel per model. Returns paths written.

    With `by_case`, also writes the rate and contrast views per case. Worth
    having for the same reason the lexicon splits: pooling averages cases whose
    base rates differ more than any injection moves them. A pooled off-diagonal
    cell can be produced entirely by one case: splitting is what reveals that
    an apparent "injecting X raises Y" effect lives in a single bed.
    """
    symptoms = symptoms or SYMPTOMS
    models = sorted(df.model.unique())
    targets = {s: s for s in symptoms}
    written = []

    written.append(draw_panels(
        models, lambda m: z_matrix(df, m, "rate", symptoms),
        os.path.join(out_dir, "fig_symptom_embed_heatmap.png"),
        title=f"Embedding z by condition and symptom  {title_suffix}\n"
              "Rows are what was injected, columns what the text leans toward; "
              "blue box marks the injected symptom's own cell",
        ylabel="condition (injected symptom)", xlabel="symptom the text leans toward",
        bar_label="mean within-session z", targets_of=lambda m: targets,
        cmap=DIVERGING, vmin=-1.5, vmax=1.5, fmt="{:+.2f}"))

    written.append(draw_panels(
        models, lambda m: z_matrix(df, m, "base", symptoms),
        os.path.join(out_dir, "fig_symptom_embed_heatmap_base_rate.png"),
        title=f"Base rate by case: z with nothing injected  {title_suffix}\n"
              "A hot row means that case leans that way on its own; its condition "
              "cells must be read against this, not against zero",
        ylabel="case", xlabel="symptom the text leans toward",
        bar_label="mean within-session z, control sessions",
        cmap=DIVERGING, vmin=-1.5, vmax=1.5, height=3.4, mark_control=False,
        fmt="{:+.2f}"))

    written.append(draw_panels(
        models, lambda m: z_matrix(df, m, "contrast", symptoms),
        os.path.join(out_dir, "fig_symptom_embed_heatmap_vs_control.png"),
        title=f"z minus matched control  {title_suffix}\n"
              "Each case differenced against its OWN control before pooling. "
              "0 = no more than that case leans anyway",
        ylabel="condition (injected symptom)", xlabel="symptom the text leans toward",
        bar_label="z minus matched control", targets_of=lambda m: targets,
        cmap=DIVERGING, vmin=-0.8, vmax=0.8, mark_control=False, fmt="{:+.2f}"))

    if by_case:
        for case in sorted(df.case.dropna().unique()):
            sub = df[df.case == case]
            written.append(draw_panels(
                models, lambda m, s=sub: z_matrix(s, m, "rate", symptoms),
                os.path.join(out_dir, f"fig_symptom_embed_heatmap_{case}.png"),
                title=f"Embedding z by condition and symptom — {case}  {title_suffix}\n"
                      "Blue box marks the injected symptom's own cell",
                ylabel="condition (injected symptom)",
                xlabel="symptom the text leans toward",
                bar_label="mean within-session z", targets_of=lambda m: targets,
                cmap=DIVERGING, vmin=-1.5, vmax=1.5, fmt="{:+.2f}"))
            written.append(draw_panels(
                models, lambda m, s=sub: z_matrix(s, m, "contrast", symptoms),
                os.path.join(out_dir, f"fig_symptom_embed_heatmap_{case}_vs_control.png"),
                title=f"z minus this case's own control — {case}  {title_suffix}\n"
                      "0 = no more than this case leans anyway",
                ylabel="condition (injected symptom)",
                xlabel="symptom the text leans toward",
                bar_label="z minus matched control", targets_of=lambda m: targets,
                cmap=DIVERGING, vmin=-0.8, vmax=0.8, mark_control=False,
                fmt="{:+.2f}"))

    return [w for w in written if w]


def draw(df, out_path, title_suffix):
    """Within-session target_z next to the same thing with the matched control
    subtracted. Side by side because the gap between them IS the finding, and a
    single-panel figure hides it."""
    cond = df[df.symptom != "no_symptoms"]
    present = [s for s in SYMPTOMS if s in set(cond.symptom)]
    if not present:
        return None
    models = sorted(cond.model.unique())
    y = np.arange(len(present))
    measures = [
        ("target_z", "within-session target_z\n(0 = no preferential expression)"),
        ("target_z_vs_control", "target_z minus matched control\n(0 = no more than the case leans anyway)"),
    ]
    # sharex matters: both panels are in the same units, so independent scales
    # would shrink the left panel's bars and inflate the right panel's, hiding
    # exactly the collapse the figure exists to show.
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True,
                             figsize=(16, max(4, 0.6 * len(present) + 2)))
    nm = len(models)
    offsets = np.linspace(0.3, -0.3, nm) if nm > 1 else [0.0]
    height = 0.7 / max(nm, 1)
    for ax, (column, xlabel) in zip(axes, measures, strict=True):
        if cond[column].notna().sum() == 0:
            ax.text(0.5, 0.5, "no control sessions\nin this set", ha="center",
                    va="center", transform=ax.transAxes, fontsize=11, color="0.4")
            ax.set_xticks([])
            continue
        for mname, off in zip(models, offsets, strict=True):
            stats = [summarise(df, column, mname, s) for s in present]
            mean = np.array([st["point"] for st in stats])
            # Asymmetric bars from the ACTUAL percentile bounds. A symmetric
            # half-width silently recentres the interval on the point estimate.
            lo = np.array([st["lo"] for st in stats])
            hi = np.array([st["hi"] for st in stats])
            err = np.vstack([np.nan_to_num(mean - lo, nan=0.0),
                             np.nan_to_num(hi - mean, nan=0.0)])
            ax.barh(y + off, mean, height=height, xerr=err,
                    error_kw={"ecolor": "0.3", "lw": 1, "capsize": 3}, label=mname)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(xlabel, fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(present, fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle("Embedding manipulation check: does the patient lean toward the injected symptom?  "
                 f"{title_suffix}\n"
                 "Right panel subtracts what the case produces with nothing injected; bars are 95% CI",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS,
                    help="One or more globs of session directories.")
    ap.add_argument("--prefix", default=ACTIVE_PREFIX,
                    help="Only include sessions whose case_name prefix matches. Defaults "
                         "to the active corpus, so a second arm under a new prefix cannot "
                         "be pooled in by accident. Pass --prefix '' for everything.")
    ap.add_argument("--out-dir", default=ACTIVE_RESULTS, help="Where to write CSV + figure.")
    ap.add_argument(
        "--unit",
        choices=list(UNITS),
        default="turn",
        help="What gets embedded. 'session' and 'turn' are REPRODUCTION modes: "
             "both can exceed the model's 256-token limit and Sentence "
             "Transformers truncates silently, so neither guarantees the text "
             "was read. 'bounded-turn' splits only the turns that overflow and "
             "is the closest safe equivalent of historical turn scoring; "
             "'chunk' repacks all patient text into sentence chunks under the "
             "limit. Over-limit inputs are an error in the bounded modes and a "
             "loud warning in the reproduction modes.",
    )
    ap.add_argument(
        "--references",
        choices=["single", "expanded"],
        default="single",
        help="Reference targets per symptom. 'single' (default) uses the one "
             "hand-written sentence in symptom_scoring.prior_vocabulary, which is "
             "what every published number used. 'expanded' uses the a-priori "
             "exemplar pools, reducing them per symptom with --reference-agg "
             "(default mean). A single probe is badly under-specified: two independent "
             "100-exemplar samples of one symptom share about a sixth of their "
             "phrasing.",
    )
    ap.add_argument(
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
    ap.add_argument(
        "--unit-agg",
        choices=["max", "top2", "top3", "mean"],
        default="max",
        help="How a session's units collapse to one score per symptom. 'max' "
             "(default, historical) asks whether the symptom appeared anywhere, "
             "but a maximum grows with the number of units, so arms with longer "
             "sessions score higher for that reason alone - see the units-per-"
             "session table. 'top2'/'top3' average the best k units, keeping the "
             "'anywhere' question while damping the length dependence.",
    )
    ap.add_argument(
        "--reference-agg",
        choices=["max", "mean"],
        default="mean",
        help="How to reduce many references to one score per symptom. 'mean' "
             "(default) treats the domain as a region and is unbiased by how many "
             "references it has. 'max' asks whether the text matches any single "
             "phrasing, which saturates once the reference set is large and is "
             "biased upward by set size. Ignored with --references single.",
    )
    ap.add_argument("--by-case", action="store_true",
                    help="Also write one figure per case. Cases differ far more than "
                         "conditions do, so the pooled figure is a weighted average of "
                         "unlike things.")
    ap.add_argument("--clean-text", action="store_true",
                    help="Strip stage directions and role labels before embedding. "
                         "Changes results (~16%% of llama characters are annotation); "
                         "the frozen baselines were computed WITHOUT this.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not EMBEDDINGS_AVAILABLE:
        raise SystemExit("sentence-transformers not available")
    model = get_model()

    # References and their aggregation live in embed_core, so this script and
    # its own robustness check cannot end up scoring differently.
    scorer = ReferenceScorer(
        model,
        build_references(args.references, SYMPTOMS, REFERENCES),
        agg=args.reference_agg,
    )
    print(f"references: {args.references} ({scorer.describe()})")

    # One corpus loop, shared with the robustness check and rerun.py, so text
    # units, cleaning, QC and unit counting cannot drift between the measure and
    # its own robustness check. `qc_view` is applied inside it: view 2 drops
    # flagged turns, view 3 drops flagged sessions.
    result = score_corpus(scorer, args.sessions, prefix=args.prefix,
                          unit=args.unit, clean=args.clean_text,
                          unit_agg=args.unit_agg, qc_view=args.qc_view)
    if not result["rows"]:
        raise SystemExit(f"No sessions matched: {args.sessions}")
    used_dirs = result["used_dirs"]  # exactly the sessions this run consumed
    # Refuses silent truncation in the bounded modes; warns in the reproduction
    # modes (`session`, `turn`), which are allowed to overflow - loudly.
    report_coverage(result, args.unit, model.max_seq_length)

    rows = []
    for row in result["rows"]:
        d = dict(row)
        srow = np.array([d[f"sim_{s}"] for s in SYMPTOMS], dtype=float)
        mu, sd = srow.mean(), srow.std()
        # Per-symptom z for EVERY session, controls included. Controls have no
        # target, but they do have a lean, and A1 needs it: without a z per
        # symptom on control sessions there is nothing to subtract.
        for k, s in enumerate(SYMPTOMS):
            d[f"z_{s}"] = float((srow[k] - mu) / sd) if sd > 0 else np.nan
            # A2: leave-one-out. Including the target in its own reference
            # distribution inflates both mu and sd when the target is genuinely
            # elevated, shrinking its own z. The bias scales with effect size.
            others = np.delete(srow, k)
            osd = others.std()
            d[f"zloo_{s}"] = float((srow[k] - others.mean()) / osd) if osd > 0 else np.nan
        if d["symptom"] in SYMPTOMS and sd > 0:
            tgt = srow[SYMPTOMS.index(d["symptom"])]
            d["target_sim"] = float(tgt)
            d["target_z"] = d[f"z_{d['symptom']}"]
            d["target_z_loo"] = d[f"zloo_{d['symptom']}"]
            d["target_rank"] = int((srow > tgt).sum() + 1)  # 1 = most similar
        else:
            d["target_sim"] = d["target_z"] = d["target_z_loo"] = np.nan
            d["target_rank"] = np.nan
        rows.append(d)

    df = pd.DataFrame(rows)
    df = add_control_contrast(df, SYMPTOMS)

    # Provenance FIRST, before a single artifact is written: the refusal to
    # overwrite an incompatible run only protects anything if nothing has been
    # written yet. It used to run at the end, after the CSV was already gone.
    provenance.write(args.out_dir, script=os.path.basename(__file__), args=args,
                     inputs={"n_sessions": int(df.session_id.nunique())},
                     session_dirs=used_dirs)

    df.to_csv(os.path.join(args.out_dir, "symptom_embed_long.csv"), index=False)

    cond = df[df.symptom != "no_symptoms"].copy()
    models = sorted(cond.model.unique())
    # Required reporting, not diagnostics: the contrast is a difference of
    # maxima, and a max over more units is larger in expectation. If the arms
    # differ here, part of any contrast is unit count rather than expression.
    units_by_arm = df.groupby("symptom")["n_units"].agg(["mean", "min", "size"])
    print(f"== units per session by condition (unit={args.unit}) ==")
    print(units_by_arm.to_string(float_format=lambda v: f"{v:.2f}"))
    spread = units_by_arm["mean"].max() - units_by_arm["mean"].min()
    print(f"   spread across conditions: {spread:.2f} units"
          f"{'  <- unbalanced; read the contrast with this in mind' if spread > 0.2 else ''}\n")

    print(f"== Within-session target_z | {len(df)} sessions, models={models} ==")
    print("   target_z>0 means patient speech leans toward the INJECTED symptom.\n")
    for mname in models:
        sub = cond[cond.model == mname]
        present_syms = [s for s in SYMPTOMS if s in set(sub.symptom)]
        g = sub.groupby("symptom").agg(
            target_z=("target_z", "mean"),
            z_loo=("target_z_loo", "mean"),
            z_sd=("target_z", "std"),
            mean_rank=("target_rank", "mean"),
            top1_rate=("target_rank", lambda r: (r == 1).mean()),
            n=("target_z", "size"),
        ).reindex(present_syms)
        # From summarise(), the SAME estimator the figure uses: one contrast per
        # case, equal-weight pooled, bootstrap interval. Averaging the
        # per-session contrast column here instead is a different estimand and
        # disagreed with the plot by up to 0.050.
        stats = {s: summarise(df, "target_z_vs_control", mname, s) for s in present_syms}
        g.insert(2, "z_vs_ctrl", [stats[s]["point"] for s in present_syms])
        g.insert(3, "ci_lo", [stats[s]["lo"] for s in present_syms])
        g.insert(4, "ci_hi", [stats[s]["hi"] for s in present_syms])
        g.insert(5, "d", [stats[s]["cohens_d"] for s in present_syms])
        print(f"--- {mname} ---")
        print(g.to_string(formatters={c: "{:.2f}".format for c in
                                      ["target_z", "z_loo", "z_vs_ctrl", "ci_lo",
                                       "ci_hi", "d", "z_sd", "mean_rank",
                                       "top1_rate"]}))
        print(f"  overall mean target_z = {sub.target_z.mean():.3f} | "
              f"top-1 rate = {(sub.target_rank==1).mean():.2f} (chance={1/9:.2f}) | "
              f"mean rank = {sub.target_rank.mean():.2f} (chance=5.0)")
        vals = [stats[s]["point"] for s in present_syms
                if not np.isnan(stats[s]["point"])]
        if vals:
            print(f"  vs matched control  = {np.mean(vals):+.3f}  "
                  f"(mean over {len(vals)} symptoms of the per-case pooled contrast)")
        else:
            print("  vs matched control  = n/a (no no_symptoms sessions in this set)")
        print(f"  leave-one-out       = {sub.target_z_loo.mean():+.3f}\n")

    # figure with 95% CI error bars
    # Provenance beside the result: every setting below has been shown to
    # move numbers, so a CSV without them cannot be compared to anything.
    written = [draw(df, os.path.join(args.out_dir, "fig_symptom_embed.png"),
                    f"({args.prefix or 'all'}, unit={args.unit}, n={len(df)})")]
    if args.by_case:
        for case in sorted(df.case.dropna().unique()):
            sub = df[df.case == case]
            written.append(draw(
                sub, os.path.join(args.out_dir, f"fig_symptom_embed_{case}.png"),
                f"({case}, unit={args.unit}, n={len(sub)})"))
    written += write_heatmaps(
        df, args.out_dir,
        f"({args.prefix or 'all'}, unit={args.unit}, refs={args.references})",
        SYMPTOMS, by_case=args.by_case)
    written = [w for w in written if w]
    for w in written:
        print(f"wrote {w}")


if __name__ == "__main__":
    main()