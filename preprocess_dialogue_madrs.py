"""Build patient-only, symptom-labelled MADRS dialogue samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from dialogue_madrs.data import build_dialogue_samples, grouped_split, save_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts-root", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--transcript-name", default="transcript.jsonl")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = build_dialogue_samples(
        args.transcripts_root,
        args.labels,
        transcript_name=args.transcript_name,
    )
    splits = grouped_split(
        samples,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    for name, split_samples in splits.items():
        save_samples(split_samples, args.output_dir / f"{name}.jsonl")
        print(f"{name}: {len(split_samples)} samples")


if __name__ == "__main__":
    main()
