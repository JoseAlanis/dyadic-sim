"""Score one saved dyadic session, or every session in a directory."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dyadic_sim.scoring.config import ScoringConfig
from dyadic_sim.scoring.instruments import REGISTRY, get_instrument
from dyadic_sim.scoring.pipeline import SymptomScoringPipeline
from dyadic_sim.scoring.result_writer import (
    write_directory_summary,
    write_session_json,
    write_turn_csv,
)

DEFAULT_OUTPUT_DIR = Path("data/results/symptom_scores")
_EXPERIMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def experiment_output_dir(base_dir: str | Path, experiment_name: str | None) -> Path:
    base = Path(base_dir)
    if experiment_name is None:
        return base
    if not _EXPERIMENT_NAME.fullmatch(experiment_name) or experiment_name in {".", ".."}:
        raise ValueError(
            "Experiment names must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, or hyphens."
        )
    return base / experiment_name


def session_output_paths(
    output_root: str | Path,
    session_id: str,
    *,
    json_name: str = "symptom_scores.json",
    csv_name: str = "turn_scores.csv",
) -> tuple[Path, Path]:
    safe_session_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(session_id)).strip("._")
    if not safe_session_id:
        safe_session_id = "unknown_session"
    session_dir = Path(output_root) / safe_session_id
    return session_dir / Path(json_name).name, session_dir / Path(csv_name).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session-path", help="Session directory or transcript.jsonl path")
    source.add_argument("--sessions-dir", help="Directory whose child folders are sessions")
    parser.add_argument(
        "--instrument",
        choices=sorted(REGISTRY),
        default="madrs",
        help="Rating instrument (topic taxonomy) to score against",
    )
    parser.add_argument(
        "--model-id",
        default="webesama/MADRS-BERT",
        help="Rater checkpoint; must match the chosen --instrument",
    )
    parser.add_argument(
        "--model-revision",
        help=(
            "Pin the rater checkpoint to a git revision (commit SHA, tag, or branch). "
            "Unpinned runs record the resolved revision in the output instead."
        ),
    )
    parser.add_argument(
        "--translator-model-id",
        default="Helsinki-NLP/opus-mt-en-de",
    )
    parser.add_argument(
        "--translator-revision",
        help="Pin the translation checkpoint to a git revision",
    )
    parser.add_argument("--source-language", choices=["en", "de"], default="en")
    parser.add_argument("--aggregation", choices=["maximum"], default="maximum")
    parser.add_argument("--relevance-mode", choices=["hybrid", "rules"], default="hybrid")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--output-path",
        help="Optional JSON filename inside the session result folder",
    )
    parser.add_argument(
        "--turn-csv",
        help="Optional turn-level CSV filename inside the session result folder",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--experiment-name",
        help=(
            "Create an optional experiment parent folder under --output-dir. "
            "Every evaluated session always receives its own child folder."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ScoringConfig:
    return ScoringConfig(
        instrument=get_instrument(args.instrument),
        model_id=args.model_id,
        model_revision=args.model_revision,
        translator_model_id=args.translator_model_id,
        translator_revision=args.translator_revision,
        source_language=args.source_language,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        relevance_mode=args.relevance_mode,
        aggregation_method="maximum_relevant_evidence",
    )


def main() -> int:
    args = parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    try:
        experiment_dir = experiment_output_dir(args.output_dir, args.experiment_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = build_config(args)
    print(
        f"Scoring against {config.instrument.name} "
        f"({len(config.instrument.topics)} topics) with {config.model_id}"
    )
    print("Loading translation, relevance, and rating models once...")
    pipeline = SymptomScoringPipeline(config)

    if args.session_path:
        result = pipeline.score_session(args.session_path)
        output_path, turn_csv_path = session_output_paths(
            experiment_dir,
            result["session_id"],
            json_name=(
                Path(args.output_path).name
                if args.output_path
                else "symptom_scores.json"
            ),
            csv_name=(
                Path(args.turn_csv).name
                if args.turn_csv
                else "turn_scores.csv"
            ),
        )
        write_session_json(result, output_path)
        write_turn_csv(result, turn_csv_path)
        print(f"Wrote session result: {output_path}")
        print(f"Wrote turn details: {turn_csv_path}")
        return 0

    sessions_root = Path(args.sessions_dir)
    candidates = sorted(
        path
        for path in sessions_root.iterdir()
        if path.is_dir() and (path / "transcript.jsonl").exists()
    )
    if not candidates:
        raise SystemExit(f"No session directories found under {sessions_root}")

    output_dir = experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    errors = []
    for index, session_path in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] Scoring {session_path.name}")
        try:
            result = pipeline.score_session(session_path)
            json_path, csv_path = session_output_paths(
                output_dir,
                result["session_id"],
            )
            write_session_json(result, json_path)
            write_turn_csv(result, csv_path)
            results.append(result)
        except Exception as exc:
            error = {
                "session_path": str(session_path),
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            errors.append(error)
            print(f"  ERROR: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise

    write_directory_summary(results, output_dir / "sessions_summary.csv")
    if errors:
        error_path = output_dir / "errors.jsonl"
        with error_path.open("w", encoding="utf-8") as handle:
            for error in errors:
                handle.write(json.dumps(error, ensure_ascii=False) + "\n")
        print(f"Completed with {len(errors)} errors; see {error_path}")
    print(f"Wrote {len(results)} session results under {output_dir}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
