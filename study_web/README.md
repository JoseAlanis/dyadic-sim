# Clinician rating packages

Offline rating for simulated sessions. Each clinician gets **one HTML file**. They
open it in a browser from disk, rate, save a JSON file and send it back. There is
no server, no login and no network access; the page's content security policy
blocks every request. The look follows the BibUnion docs theme, whose colour
tokens are copied into `page/theme.css`.

```
select_sessions  batch -> manifest.json        which sessions, for whom, drawn how
build_package    manifest -> <rater>.html      blinded page, one per rater
                          -> sealed/           key and frozen text: never send
   (rater: open, rate, Save to file, send the newest file back)
import_returns   files -> returns/imported/    checked against sealed/, then tables
```

Everything written lives under `data/results/clinician_study/`, which is
gitignored. Only code, the draft codebook and the synthetic fixtures are tracked.

## 1. Select sessions from a batch

Any batch works. `--prefix` is the run prefix in each `case_name`, and
`--sessions` takes the same globs as the analysis CLIs, so an imported batch in
`data/external/<name>/` is read like a local one. Only metadata is opened here.

```bash
# Two shared sessions per no_complaint cell and one per empty_and_invisible
# cell for every rater, one unique no_complaint session per cell for each
# rater, three drift repeats
uv run python -m study_web.select_sessions --prefix <prefix> \
  --shared no_complaint=2,empty_and_invisible=1 --unique no_complaint=1 \
  --repeats 3 --raters R1,R2,R3 --seed 1 \
  --study-version <study> \
  --exclude-manifest data/results/clinician_study/manifests/<earlier-study>.json \
  --out data/results/clinician_study/manifests/<study>.json
```

- **Cells.** Sampling is stratified by `(case, condition)`. A count is one number
  for every cell, `all`, or per case (`case=N,...`).
- **Sets.** `--shared` sessions go to every rater. `--unique` sessions go to one
  rater each and are disjoint. `--repeats` re-rates shared sessions at the end,
  under fresh ids.
- **Filters.** `--cases`, `--conditions` and `--runs 1-10` narrow the frame.
  Complete sessions only. `--exclude-manifest` keeps demo, calibration and study
  sets disjoint.
- **Refusals.** A cell too small for the request is an error unless
  `--allow-short`. The manifest records each cell's frame size, what was drawn,
  the seed and the command, so inclusion probabilities can be computed from it.
- **Dry run.** Without `--out` the draw is only printed.

## 2. Build the packages

```bash
uv run python -m study_web.build_package \
  --manifest data/results/clinician_study/manifests/<study>.json
```

This writes `data/results/clinician_study/<study_version>/`:

| path | what | who sees it |
|---|---|---|
| `packages/<rater>.html` | the rating page with that rater's sessions | that rater only |
| `packages/MANIFEST.json` | package ids and file hashes | you |
| `sealed/key.json` | assignment → source session, set, repeat link, manifest, commit | **never send** |
| `sealed/snapshots/*.json` | the exact displayed text, with its source fields | **never send** |

Blinding is checked twice. The page payload is built from an allowlist of
fields. Each finished HTML file is then scanned for session ids, case names, run
prefixes, `_run_N`, file paths and generation timestamps. If any package fails,
nothing is written. A study version is written once; rebuilding it is refused,
because returned files are bound to its package ids.

Text is shown verbatim and in true order. The therapist opener comes first, then
each row's patient line followed by the therapist reply, final reply included. A
missing opener is recorded, never reconstructed. Do not hand out the delivered
PDFs: page one names the case and the injected symptom.

## 3. What the rater does

1. Opens the file from disk. The instructions and codebook show on first visit
   and stay available under **Instructions**.
2. Selects words in a patient turn. The panel asks for symptom, code, elicited,
   optional context lines and a note. Several claims can share a passage, and a
   claim can cite several passages of its own turn. **Use the whole turn as
   evidence** avoids precise selection.
3. Marks every patient turn reviewed. An unmarked turn is missing, not negative.
4. Fills in the session judgement once all turns are reviewed: expression and
   frequency per symptom, optional evidence links, most prominent symptom,
   presenting complaint and realism.
5. Submits. The answers are frozen and a file is saved. Reopening needs a reason;
   the earlier submission is kept.

**Saving.** The browser keeps a safety copy after every change, in local storage
keyed to the package. The record is the JSON file from **Save to file**, which
the page also writes on every submit. **Load saved file** resumes on another
browser or machine. The header says when changes are newer than the last file,
and closing the page with unsaved changes asks first. If the same package is
open in two tabs, the second one to save stops and says why.

## 4. Import returned files

```bash
uv run python -m study_web.import_returns \
  --study-dir data/results/clinician_study/<study>  ~/Downloads/rating_R1_*.json
```

Nothing in a file is trusted. It must match its package, rater and versions.
Every reference must exist, and every span must lie in a patient turn with
`text[start:end] == quote` in code points. Every submitted session must be
complete.

- Files are copied unchanged to `returns/raw/`.
- For each rater, the newest valid file feeds `returns/imported/`: `ratings.jsonl`,
  `claims.csv`, `turn_labels.csv`, `session_symptoms.csv`, `session_globals.csv`
  and `import_report.json`.
- The tables use opaque ids only. Join `sealed/key.json` for sources and
  conditions.

Turn labels use the draft reduction `turn-v0-draft`:

| turn state | label |
|---|---|
| unreviewed | `missing` |
| reviewed, no claim | `no_evidence` |
| one code | that code |
| several codes | `unclear`, with `mixed=True` |

## Tests

```bash
uv run pytest tests/study_web -q                       # builder, blinding, importer, selector
STUDY_WEB_BROWSER=1 uv run pytest tests/study_web -q   # plus a real Chrome round trip
```

The browser test rates a fixture session in headless Chrome, then imports the
file the page saved. The fixture puts an emoji before the phrase it selects, so
UTF-16 and code-point offsets differ. The driver is `tests/study_web/browser_e2e.mjs`
and needs Node 22 or newer.

## Status and known gaps

- **Browsers.** Chrome is tested end to end. Firefox is checked only for loading
  and rendering. Safari and Windows are untested. Test the clinicians' actual
  browsers and machines before the pilot.
- **Codebook.** It is `draft-0` in `codebook.py`. Wording, examples and the turn
  reduction rule are to be fixed in calibration.
- **Not implemented yet.**
  - Form-first counterbalancing.
  - Sampling enriched for instrument disagreement.
  - Evidence links for frequency that point to spans rather than to claims.
  - Adjudication.
  - A progress view: progress is visible only in returned files.
- **Spans.** A claim's spans must all lie in its own turn. Evidence in another
  turn gets its own claim, which keeps the turn-by-symptom reduction
  deterministic.
- **Keeping sealed data sealed** is a matter of practice. Nothing stops you from
  opening `sealed/` or reading validation labels early.
