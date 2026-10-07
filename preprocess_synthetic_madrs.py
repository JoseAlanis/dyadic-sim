"""Convert SyntheticData.xlsx into subject-grouped patient-only MADRS splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dialogue_madrs.data import (
    build_synthetic_workbook_samples,
    grouped_split,
    save_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = build_synthetic_workbook_samples(args.workbook)
    splits = grouped_split(samples, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"source": str(args.workbook), "seed": args.seed, "splits": {}}
    for name, items in splits.items():
        save_samples(items, args.output_dir / f"{name}.jsonl")
        summary["splits"][name] = {
            "subjects": len({item.patient_id for item in items}),
            "samples": len(items),
            "scores": dict(
                sorted(Counter(item.severity_score for item in items).items())
            ),
        }
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
