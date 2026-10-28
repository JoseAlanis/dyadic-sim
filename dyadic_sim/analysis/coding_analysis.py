"""Score returned human labels: coder agreement, then instrument validity.

Tier D, second half. Run after coders have filled in `answer` on the blinded
items produced by `coding_sample.py`.

Order matters and is enforced here. **Coder agreement is computed and reported
first**, because instrument validity measured against unreliable labels is
meaningless — if two coders cannot agree on whether a patient reported sleep
problems, "the lexicon agrees with humans 80% of the time" says nothing. A low
kappa is a finding about the construct, not a reason to pick the more agreeable
coder.

Only then does it ask the question the whole project turns on: does the group
separation the instruments report correspond to something a person reading the
transcript can see, or is it a systematic linguistic proxy?

    dyadic-sim coding-analysis \\
        --labels coding/coder_a.jsonl coding/coder_b.jsonl \\
        --key coding/KEY_DO_NOT_OPEN.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os

import pandas as pd

YES = {"yes", "y", "1", "true"}
NO = {"no", "n", "0", "false"}


def read_labels(path: str) -> dict:
    """item_id -> bool, skipping unanswered and unparseable rows."""
    out = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            answer = str(row.get("answer", "")).strip().lower()
            if answer in YES:
                out[row["item_id"]] = True
            elif answer in NO:
                out[row["item_id"]] = False
    return out


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Chance-corrected agreement. Raw agreement is misleading when a class is rare."""
    n = len(a)
    if not n:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return float("nan")
    return (observed - expected) / (1 - expected)


def confusion(truth: list[bool], pred: list[bool]) -> dict:
    tp = sum(t and p for t, p in zip(truth, pred, strict=True))
    fp = sum((not t) and p for t, p in zip(truth, pred, strict=True))
    fn = sum(t and (not p) for t, p in zip(truth, pred, strict=True))
    tn = sum((not t) and (not p) for t, p in zip(truth, pred, strict=True))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    ok = not (math.isnan(precision) or math.isnan(recall)) and precision + recall
    f1 = 2 * precision * recall / (precision + recall) if ok else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "n": tp + fp + fn + tn}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--labels", nargs="+", required=True,
                    help="One returned file per coder.")
    ap.add_argument("--key", default=os.path.join("coding", "KEY_DO_NOT_OPEN.jsonl"))
    ap.add_argument("--out-dir", default="coding")
    args = ap.parse_args()

    key = {}
    with open(args.key, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                key[row["item_id"]] = row

    coders = {os.path.basename(p): read_labels(p) for p in args.labels}
    print(f"coders: {', '.join(f'{k} ({len(v)} labelled)' for k, v in coders.items())}")

    # ---- 1. Agreement FIRST. Validity against unreliable labels is meaningless.
    names = list(coders)
    consensus: dict[str, bool] = {}
    if len(names) >= 2:
        print("\n=== coder agreement (computed before any validity claim) ===")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                shared = sorted(set(coders[names[i]]) & set(coders[names[j]]))
                if not shared:
                    continue
                a = [coders[names[i]][x] for x in shared]
                b = [coders[names[j]][x] for x in shared]
                raw = sum(x == y for x, y in zip(a, b, strict=True)) / len(shared)
                kappa = cohens_kappa(a, b)
                print(f"  {names[i]} vs {names[j]}: n={len(shared)}  "
                      f"raw={raw:.2f}  kappa={kappa:.2f}")
                if kappa < 0.6:
                    print("    ! kappa below 0.6: the construct is not reliably "
                          "codable from this material. Read the validity numbers "
                          "below as an upper bound at best.")
        # Consensus = unanimous only. Disagreements are excluded and counted
        # rather than resolved by majority, which would invent certainty.
        common = set.intersection(*(set(c) for c in coders.values()))
        disputed = 0
        for item in common:
            values = {c[item] for c in coders.values()}
            if len(values) == 1:
                consensus[item] = values.pop()
            else:
                disputed += 1
        print(f"  consensus items: {len(consensus)}; disputed and excluded: {disputed}")
    else:
        consensus = dict(next(iter(coders.values())))
        print("\n! single coder: agreement is not estimable, so nothing below is "
              "validated against a reliable label.")

    # ---- 2. Instrument validity, only now.
    rows = []
    for item, human in consensus.items():
        meta = key.get(item)
        if not meta:
            continue
        rows.append({**meta, "human": human})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no labelled items matched the key")

    for kind, machine_col in (("session", "detected"), ("turn", "lexicon_fired")):
        sub = df[(df.kind == kind) & df[machine_col].notna()] if kind in set(df.kind) \
            else df.iloc[0:0]
        if sub.empty:
            continue
        print(f"\n=== lexicon vs human, {kind} level ===")
        stats = confusion(list(sub.human), list(sub[machine_col].astype(bool)))
        print(f"  n={stats['n']}  precision={stats['precision']:.2f}  "
              f"recall={stats['recall']:.2f}  f1={stats['f1']:.2f}")
        print(f"  tp={stats['tp']} fp={stats['fp']} fn={stats['fn']} tn={stats['tn']}")

        # The question the project turns on: does the ARM separate for humans?
        if kind == "session" and "arm" in sub.columns:
            # REWEIGHTED. The sample draws equal detected/undetected within each
            # arm, so an unweighted rate is ~50% in every arm by construction and
            # says nothing. Weights restore the population base rate.
            print("  human-judged report rate by arm, reweighted to the corpus")
            print("  (this is the question the project turns on):")
            has_w = "stratum_weight" in sub.columns
            for arm, g in sub.groupby("arm"):
                if has_w and g.stratum_weight.notna().all():
                    w = g.stratum_weight.astype(float)
                    rate = float((g.human.astype(float) * w).sum() / w.sum())
                    print(f"    {arm:8s} {rate:.0%}  (n={len(g)}, weighted)")
                else:
                    print(f"    {arm:8s} {g.human.mean():.0%}  (n={len(g)}, UNWEIGHTED "
                          f"- not comparable across arms)")

        if "role_confused" in sub.columns and sub.role_confused.any():
            clean = sub[~sub.role_confused.astype(bool)]
            rough = sub[sub.role_confused.astype(bool)]
            if len(clean) and len(rough):
                print(f"  agreement on clean generations: "
                      f"{(clean.human == clean[machine_col].astype(bool)).mean():.0%} "
                      f"(n={len(clean)})")
                print(f"  agreement on role-confused    : "
                      f"{(rough.human == rough[machine_col].astype(bool)).mean():.0%} "
                      f"(n={len(rough)})")

    out = os.path.join(args.out_dir, "coding_scored.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
