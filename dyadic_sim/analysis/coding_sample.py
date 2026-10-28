"""Draw a blinded, stratified sample for human coding, and seal the key.

Tier D. Every automated number in this project describes an instrument, not a
patient: nothing here has been checked against human judgement. Until that
exists, "the lexicon found 19%" is a statement about the lexicon, and group
separation could equally be a systematic linguistic proxy rather than symptom
expression.

Two samples, because they answer different questions:

  session items  "Reading this whole conversation, does the patient report X?"
                 Tests whether the group separation the instruments report is
                 visible to a person at all.
  turn items     "Is this turn evidence that the patient has X?"
                 Tests the evidence trail directly: precision and recall of the
                 lexicon at the level where it actually fires.

Both are stratified so that the sample cannot flatter the instruments. Coders
see equal numbers of detected and undetected items, injected and control
sessions, and the strata deliberately include role-confused and crisis-paused
sessions rather than quietly excluding the awkward ones.

**Blinding.** The written items carry no condition, case name, session id, run
number or detector verdict, and are shuffled. The key is written to a separate
file that coders must not open. Item ids are opaque digests, so ordering leaks
nothing either.

    dyadic-sim coding-sample --out-dir coding
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

from dyadic_sim.analysis.naive_prevalence import scan
from dyadic_sim.analysis.sessions import (
    CRISIS_TERMINATED,
    iter_sessions,
    patient_turns,
)
from dyadic_sim.core.vocabulary import LABELS
from dyadic_sim.scoring.transcript_parser import build_turn_pairs, load_session_source

SESSION_GLOB = ["data/sessions/session_*"]

#: Plain-language question per symptom, so a coder never needs the codebook to
#: parse the item. Deliberately about REPORTING, not about diagnosis.
QUESTIONS = {
    key: f"Does the patient report this: {label.lower()}?"
    for key, label in LABELS.items()
}


def _item_id(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


def collect(prefixes: list[str]) -> list[dict]:
    """One record per (session, symptom) with the detector verdict attached."""
    out = []
    for prefix in prefixes:
        df, evidence, _dirs = scan(SESSION_GLOB, prefix, 1, clean=True)
        patient = df[(df.panel == "PHQ-9") & (df.speaker == "patient")]
        ev = evidence[(evidence.panel == "PHQ-9") & (evidence.speaker == "patient")] \
            if len(evidence) else evidence
        turns_by_session = {}
        for d, meta, run_prefix, case, symptom, _run in iter_sessions(SESSION_GLOB):
            if run_prefix == prefix:
                turns_by_session[meta["session_id"]] = (d, meta, case, symptom)
        for row in patient.itertuples():
            if row.session_id not in turns_by_session:
                continue
            d, meta, case, injected = turns_by_session[row.session_id]
            hits = []
            if len(ev):
                sub = ev[(ev.session_id == row.session_id)
                         & (ev.detected_symptom == row.detected_symptom)]
                hits = sorted({int(t) for t in sub.turn})
            out.append({
                "prefix": prefix, "session_dir": d, "session_id": row.session_id,
                "case": case, "injected": injected,
                "symptom": row.detected_symptom,
                "detected": bool(row.detected),
                "evidence_turns": hits,
                "role_confused": bool(row.any_role_confusion),
                "crisis": meta.get("_status") == CRISIS_TERMINATED,
                # target / other / control, the arm distinction that matters
                "arm": ("control" if injected == "no_symptoms"
                        else "target" if injected == row.detected_symptom
                        else "other"),
            })
    return out


#: Fraction of each cell reserved for role-confused or crisis-terminated
#: sessions. They are ~4% of the corpus, so a proportional sample would contain
#: almost none and could not say whether the instruments survive them. Boosting
#: to a quarter keeps them testable WITHOUT making the sample unrepresentative:
#: an earlier version sorted all awkward cases first and produced a set that was
#: 62% role-confused, from which no population-level agreement rate could be
#: estimated at all.
AWKWARD_SHARE = 0.25


def stratum_weights(records: list[dict], picked: list[dict]) -> dict:
    """Population size / sampled size per (arm, detected) cell.

    Required, not decorative. The sample deliberately draws equal numbers of
    detected and undetected items WITHIN each arm, which destroys the arm's base
    rate: an unweighted "human report rate by arm" comes out near 50% in every
    arm by construction and answers nothing. Reweighting by these factors
    recovers the population rate, which is the number the whole project turns
    on.
    """
    pop, samp = {}, {}
    for r in records:
        pop[(r["arm"], r["detected"])] = pop.get((r["arm"], r["detected"]), 0) + 1
    for r in picked:
        samp[(r["arm"], r["detected"])] = samp.get((r["arm"], r["detected"]), 0) + 1
    return {k: pop[k] / samp[k] for k in samp if samp.get(k)}


def stratified(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Equal draw per (arm, detected) cell, with a capped awkward quota.

    Sampling proportionally overall would fill the set with undetected control
    items, where every instrument agrees and a coder learns nothing. Equal cells
    put the disagreements in front of the coder; the quota keeps hard cases
    present without letting them take over.
    """
    buckets: dict[tuple, list[dict]] = {}
    for r in records:
        buckets.setdefault((r["arm"], r["detected"]), []).append(r)
    per = max(1, n // max(len(buckets), 1))
    quota = max(1, round(per * AWKWARD_SHARE))
    picked = []
    for key in sorted(buckets):
        pool = buckets[key]
        rng.shuffle(pool)
        awkward = [r for r in pool if r["role_confused"] or r["crisis"]]
        ordinary = [r for r in pool if not (r["role_confused"] or r["crisis"])]
        take = awkward[:quota]
        take += ordinary[:per - len(take)]
        # If a cell has too few ordinary sessions, backfill rather than
        # returning a short cell.
        if len(take) < per:
            take += awkward[len(take):per]
        picked.extend(take[:per])
    rng.shuffle(picked)
    return picked


def session_text(session_dir: str) -> str:
    """Full dialogue, patient turns marked. Context matters to a human reader."""
    pairs = build_turn_pairs(load_session_source(session_dir))
    lines = []
    for pair in pairs:
        if not pair.valid:
            continue
        if pair.therapist_text:
            lines.append(f"THERAPIST: {pair.therapist_text.strip()}")
        if pair.patient_text:
            lines.append(f"PATIENT:   {pair.patient_text.strip()}")
    return "\n\n".join(lines)


def turn_items(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Turns the lexicon fired on, plus matched turns from the same sessions.

    Balanced on purpose: a sample of only firings measures precision and says
    nothing about what was missed. Firings are drawn from the FULL record set
    rather than only the session sample - there are far fewer of them, and
    restricting to the session sample yielded 39 where 100 were asked for.
    Quiet turns are then drawn from the same sessions as the firings, so a coder
    cannot separate the two classes by topic or writing style.
    """
    fired, by_session = [], {}
    for r in records:
        turns = dict(patient_turns(r["session_dir"], clean=True))
        by_session[(r["session_id"], r["symptom"])] = [
            {**r, "turn": i, "turn_text": t} for i, t in turns.items()
            if i not in r["evidence_turns"]
        ]
        for index in r["evidence_turns"]:
            if index in turns:
                fired.append({**r, "turn": index, "turn_text": turns[index]})
    rng.shuffle(fired)
    half = max(1, n // 2)
    picked = fired[:half]
    quiet_pool = []
    for item in picked:
        quiet_pool.extend(by_session.get((item["session_id"], item["symptom"]), []))
    rng.shuffle(quiet_pool)
    seen = set()
    for q in quiet_pool:
        key = (q["session_id"], q["symptom"], q["turn"])
        if key not in seen:
            seen.add(key)
            picked.append(q)
        if len(picked) >= 2 * half:
            break
    rng.shuffle(picked)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prefixes", nargs="+", default=["power01"])
    ap.add_argument("--n-sessions", type=int, default=60)
    ap.add_argument("--n-turns", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--out-dir", default="coding")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = random.Random(args.seed)
    records = collect(args.prefixes)
    print(f"candidate (session, symptom) records: {len(records)}")

    picked = stratified(records, args.n_sessions, rng)
    weights = stratum_weights(records, picked)
    turns = turn_items(records, args.n_turns, rng)

    sess_path = os.path.join(args.out_dir, "session_items.jsonl")
    turn_path = os.path.join(args.out_dir, "turn_items.jsonl")
    key_path = os.path.join(args.out_dir, "KEY_DO_NOT_OPEN.jsonl")

    key = []
    with open(sess_path, "w", encoding="utf-8") as fh:
        for r in picked:
            iid = _item_id("session", r["session_id"], r["symptom"], args.seed)
            fh.write(json.dumps({
                "item_id": iid, "kind": "session",
                "question": QUESTIONS[r["symptom"]],
                "transcript": session_text(r["session_dir"]),
                "answer": "", "confidence": "", "notes": "",
            }, ensure_ascii=False) + "\n")
            key.append({"item_id": iid, "kind": "session",
                        "stratum_weight": weights.get((r["arm"], r["detected"]), 1.0),
                        **{k: r[k] for k in ("prefix", "session_id", "case",
                                             "injected", "symptom", "detected",
                                             "arm", "role_confused", "crisis")}})

    with open(turn_path, "w", encoding="utf-8") as fh:
        for r in turns:
            iid = _item_id("turn", r["session_id"], r["symptom"], r["turn"], args.seed)
            fh.write(json.dumps({
                "item_id": iid, "kind": "turn",
                "question": QUESTIONS[r["symptom"]],
                "turn_text": r["turn_text"],
                "answer": "", "confidence": "", "notes": "",
            }, ensure_ascii=False) + "\n")
            key.append({"item_id": iid, "kind": "turn", "turn": r["turn"],
                        "lexicon_fired": r["turn"] in r["evidence_turns"], **{
                            k: r[k] for k in ("prefix", "session_id", "case",
                                              "injected", "symptom", "arm",
                                              "role_confused", "crisis")}})

    with open(key_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(k, ensure_ascii=False) + "\n" for k in key)

    def dist(rows, field):
        counts = {}
        for r in rows:
            counts[r[field]] = counts.get(r[field], 0) + 1
        return dict(sorted(counts.items()))

    print(f"\nwrote {sess_path}: {len(picked)} session items")
    print(f"  arm      {dist(picked, 'arm')}")
    print(f"  detected {dist(picked, 'detected')}")
    print(f"  symptom  {dist(picked, 'symptom')}")
    print(f"  awkward  role_confused={sum(r['role_confused'] for r in picked)}, "
          f"crisis={sum(r['crisis'] for r in picked)}")
    print(f"wrote {turn_path}: {len(turns)} turn items "
          f"({sum(t['turn'] in t['evidence_turns'] for t in turns)} the lexicon fired on)")
    print(f"wrote {key_path}  <- coders must not open this")
    print(f"\nseed {args.seed}; rerunning with the same seed reproduces this sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
