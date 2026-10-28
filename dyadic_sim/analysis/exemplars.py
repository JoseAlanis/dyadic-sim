"""Load, audit and compare the PHQ-9 exemplar pools.

The pools live in `dyadic_sim/analysis/exemplars/` and are described by
`GENERATION.md`, which is the provenance record. Two audits run here, both
BEFORE the exemplars are allowed to influence the lexicon:

**Purity.** Every exemplar is embedded and assigned to its nearest domain
centroid. An exemplar written for X but nearest to Y is a candidate leak. The
resulting confusion matrix is not only a quality check on the pool: it measures
how much of PHQ-9's domain structure survives translation to sentence level at
all, which upper-bounds what any detector can achieve. PHQ-9 is a screening and
severity instrument scored by frequency, never validated as a sentence-level
ontology, so some domain overlap is inherent to it — `low_energy` and
`lack_of_pleasure` are not cleanly separable because the instrument never
claimed they were. Read a muddy matrix as a fact about PHQ-9 first and about the
pool second.

Centroids are LEAVE-ONE-OUT: an exemplar does not contribute to the centroid it
is scored against. Including it pulls the centroid toward the point and
flatters the result, the same bias that `target_z_loo` corrects in
symptom_embed.py.

**Overlap.** Nearest-neighbour similarity between pools, in both directions,
because the two directions answer different questions:

    H -> A   is every human exemplar reached by generation?  (recall of human
             intuition; low means the generator has blind spots)
    A -> H   is every generated exemplar something a human wrote?  (novelty; low
             is GOOD, it means generation found phrasings we would not have)
    A -> B   do two independent generation passes cover the same space?
             (saturation; low means 100/domain is too few)

    dyadic-sim exemplars --purity --overlap
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

EXEMPLAR_DIR = os.path.join(os.path.dirname(__file__), "exemplars")

# Reported alongside the mean, because a single threshold on cosine is arbitrary
# and the shape of the distribution is what actually tells you whether two pools
# cover the same ground.
THRESHOLDS = (0.6, 0.7, 0.8)


def load_pool(name: str) -> list[dict]:
    """Read one JSONL pool. `name` is the stem, e.g. "pool_h"."""
    path = os.path.join(EXEMPLAR_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def available() -> list[str]:
    if not os.path.isdir(EXEMPLAR_DIR):
        return []
    return sorted(
        f[:-6] for f in os.listdir(EXEMPLAR_DIR) if f.endswith(".jsonl")
    )


def reference_sets(
    pools: tuple[str, ...] = ("pool_a",),
    grades: tuple[str, ...] = ("discriminative", "consistent"),
) -> dict[str, tuple[str, ...]]:
    """Exemplars as multi-reference embedding targets, keyed by PHQ-9 symptom.

    The expanded counterpart to `dyadic_sim.core.vocabulary.REFERENCES`,
    which holds ONE hand-written sentence per symptom. One sentence turned out to
    be badly under-specified: two independent 100-exemplar samples of the same
    symptom share only about a sixth of their phrasing (see GENERATION.md), so a
    single probe can only ever catch the corner of the space it happens to sit
    in.

    `confounded` exemplars are excluded by default. They carry the symptom's
    vocabulary but are deliberately NOT evidence for it (symptom present, cause
    external), so admitting them as reference targets would make the measure
    fire on precisely the cases it should ignore.

    Lives here rather than in `dyadic_sim.scoring` because these are analysis
    artifacts: the prompt vocabulary is generation-side and canonical, this is
    measurement-side and revisable.

    **Pool A only by default, and the reason is not arbitrary.** Aggregation over
    a symptom's references is configurable in `symptom_embed.py`
    (`--reference-agg`, default `mean`). Under `max` a score is biased upward by
    how many references it is taken over; under `mean` set size changes what is
    being averaged. Either way, unequal sets are not comparable across symptoms.
    Pool B covers only three of the nine domains, so including it would give
    those three roughly twice the references and shift their scores for that
    reason alone. Pool A is uniform at 85 per domain; pool B is for measuring
    saturation, not for scoring.

    Any caller passing multiple pools should check the counts stay balanced;
    `_warn_if_unbalanced` does it for you.
    """
    keep = set(grades)
    by_symptom: dict[str, list[str]] = defaultdict(list)
    for pool in pools:
        for record in load_pool(pool):
            if record.get("grade") in keep and record.get("text"):
                by_symptom[record["domain"]].append(record["text"])
    sets = {s: tuple(dict.fromkeys(texts)) for s, texts in by_symptom.items()}
    _warn_if_unbalanced(sets)
    return sets


def _warn_if_unbalanced(sets: dict[str, tuple[str, ...]], tolerance: float = 0.25) -> None:
    """Unequal reference-set sizes are not comparable across symptoms.

    Strongly so under `--reference-agg max`, where a maximum grows with the
    number of draws; more subtly under `mean`, where set size changes what the
    average is over.
    """
    if not sets:
        return
    counts = {s: len(v) for s, v in sets.items()}
    lo, hi = min(counts.values()), max(counts.values())
    if lo and (hi - lo) / hi > tolerance:
        import warnings

        warnings.warn(
            f"Reference sets are unbalanced ({lo}-{hi} per symptom), so scores are "
            f"not comparable across symptoms: under --reference-agg max the larger "
            f"sets score higher for that reason alone, and under mean they average "
            f"over different amounts of the space. Counts: {counts}",
            stacklevel=2,
        )


def _embed(texts: list[str]):
    from dyadic_sim.analysis.embeddings import EMBEDDINGS_AVAILABLE, get_model

    if not EMBEDDINGS_AVAILABLE:
        raise SystemExit("sentence-transformers not available")
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, batch_size=64,
                        show_progress_bar=False)


def purity(records: list[dict]):
    """Leave-one-out nearest-centroid assignment. Returns (confusion, misses)."""
    domains = sorted({r["domain"] for r in records})
    index = {d: i for i, d in enumerate(domains)}
    emb = _embed([r["text"] for r in records])
    labels = np.array([index[r["domain"]] for r in records])

    # Sum per domain once, then subtract the point itself for its own domain.
    sums = np.zeros((len(domains), emb.shape[1]))
    counts = np.zeros(len(domains))
    for vec, lab in zip(emb, labels, strict=True):
        sums[lab] += vec
        counts[lab] += 1

    confusion = np.zeros((len(domains), len(domains)), dtype=int)
    misses = []
    for record, vec, lab in zip(records, emb, labels, strict=True):
        loo_sums = sums.copy()
        loo_counts = counts.copy()
        loo_sums[lab] -= vec
        loo_counts[lab] -= 1
        centroids = loo_sums / np.maximum(loo_counts, 1)[:, None]
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.maximum(norms, 1e-12)
        assigned = int(np.argmax(centroids @ vec))
        confusion[lab, assigned] += 1
        if assigned != lab:
            misses.append((record, domains[assigned]))
    return domains, confusion, misses


def nearest_neighbour(source: list[dict], target: list[dict]) -> dict:
    """For each source exemplar, cosine to its most similar target exemplar.

    Computed per domain: a `sleep_problems` exemplar is only compared against
    `sleep_problems` exemplars, since cross-domain similarity would answer a
    different question.
    """
    by_domain_target = defaultdict(list)
    for record in target:
        by_domain_target[record["domain"]].append(record["text"])

    sims, rows = [], []
    for domain in sorted({r["domain"] for r in source}):
        src = [r for r in source if r["domain"] == domain]
        tgt = by_domain_target.get(domain, [])
        if not tgt:
            continue
        se = _embed([r["text"] for r in src])
        te = _embed(tgt)
        best = (se @ te.T).max(axis=1)
        sims.extend(best.tolist())
        rows.extend(zip(src, best.tolist(), strict=True))
    if not sims:
        return {}
    arr = np.array(sims)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        **{f"frac>={t}": float((arr >= t).mean()) for t in THRESHOLDS},
        "rows": rows,
    }


def _print_confusion(domains, confusion):
    width = max(len(d) for d in domains) + 1
    short = {d: d[:11] for d in domains}
    print(" " * width + "".join(f"{short[d]:>12s}" for d in domains) + "    acc")
    for i, d in enumerate(domains):
        row = confusion[i]
        acc = row[i] / max(row.sum(), 1)
        print(f"{d:{width}s}" + "".join(f"{v:>12d}" for v in row) + f"  {acc:5.0%}")
    total = confusion.sum()
    correct = np.trace(confusion)
    print(f"\n  overall nearest-centroid accuracy: {correct}/{total} = {correct/max(total,1):.1%}")
    print("  (a muddy matrix is a fact about PHQ-9 before it is a fact about the pool)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--purity", action="store_true",
                        help="Leave-one-out nearest-centroid confusion matrix")
    parser.add_argument("--overlap", action="store_true",
                        help="Nearest-neighbour similarity between pools")
    parser.add_argument("--pools", nargs="+", default=None,
                        help="Pool stems to include (default: every pool_* file found)")
    parser.add_argument("--show-misses", type=int, default=15,
                        help="How many purity misses to print")
    parser.add_argument(
        "--grades",
        nargs="+",
        default=["discriminative", "consistent"],
        help="Which evidential grades to include. Default excludes `confounded`, "
             "which is deliberately NOT evidence for its domain (symptom present, "
             "cause external) and would blur the centroids it is scored against. "
             "Pass `--grades discriminative consistent confounded` to include it.",
    )
    args = parser.parse_args()

    stems = args.pools or [p for p in available() if p.startswith("pool_")]
    if not stems:
        raise SystemExit(f"No pools found in {EXEMPLAR_DIR}")

    keep = set(args.grades)
    pools = {s: [r for r in load_pool(s) if r.get("grade") in keep] for s in stems}
    print("=== pools ===")
    for stem, records in pools.items():
        if not records:
            print(f"  {stem:12s} (empty or missing)")
            continue
        grades = Counter(r.get("grade", "?") for r in records)
        subjects = Counter(r.get("subject_type", "?") for r in records)
        print(f"  {stem:12s} n={len(records):4d}  domains={len({r['domain'] for r in records})}"
              f"  grade={dict(grades)}  subject={dict(subjects)}")

    if args.purity:
        for stem, records in pools.items():
            if not records:
                continue
            print(f"\n=== purity: {stem} (leave-one-out nearest centroid) ===")
            domains, confusion, misses = purity(records)
            _print_confusion(domains, confusion)
            if misses and args.show_misses:
                print(f"\n  candidate leaks (first {args.show_misses}):")
                for record, assigned in misses[:args.show_misses]:
                    print(f"    [{record['domain']} -> {assigned}] {record['text'][:82]}")

    if args.overlap:
        stems_present = [s for s in stems if pools[s]]
        print("\n=== overlap (per-domain nearest neighbour, cosine) ===")
        for a in stems_present:
            for b in stems_present:
                if a == b:
                    continue
                stats = nearest_neighbour(pools[a], pools[b])
                if not stats:
                    continue
                fr = "  ".join(f"{t}:{stats[f'frac>={t}']:.0%}" for t in THRESHOLDS)
                print(f"  {a:10s} -> {b:10s}  n={stats['n']:4d}  "
                      f"mean={stats['mean']:.3f}  median={stats['median']:.3f}   {fr}")
        print("\n  H->A low  = generation missed things the human wrote")
        print("  A->H low  = generation found phrasings the human did not (good)")
        print("  A->B low  = the two passes cover different ground; 100/domain is too few")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
