# Patient-dialogue MADRS experiment

This is a separate fine-tuning experiment. It does not change the existing
`symptom_scoring` pipeline or its `Topic:/Dialogue:/Interviewer:/Patient:` input.
The new model sees only:

```text
SYMPTOM: <MADRS item> [SEP] <patient utterances in order>
```

Scores retain the MADRS item range (0–6) and are learned with the existing
MADRS-BERT single-output regression head. Predictions are clipped to 0–6 only
for reporting; training uses the continuous regression output.

## SyntheticData.xlsx

The supplied workbook has 60 synthetic subjects. `SPEAKER_01` is the patient;
`SPEAKER_00` is the interviewer. The `Score` column repeats the same 0–6 score
on several patient utterances for a subject and topic. Preprocessing combines
the patient utterances from each topic segment into one sample per subject and
MADRS item, discards
`Allgemein` (not a MADRS item), and groups all items from a subject into the
same train, validation, or test split. It does not copy interviewer text.
For this workbook, provide similarly topic-relevant patient speech at inference.

```powershell
python preprocess_synthetic_madrs.py --workbook "C:\Users\Farbod\Downloads\SyntheticData.xlsx" --output-dir data/dialogue_madrs_synthetic
python train_dialogue_madrs.py --data-dir data/dialogue_madrs_synthetic --output-dir data/results/dialogue_madrs_synthetic --batch-size 2 --epochs 8 --patience 2 --device cuda
python evaluate_dialogue_madrs.py --model data/results/dialogue_madrs_synthetic/best_model --test-data data/dialogue_madrs_synthetic/test.jsonl --output-dir data/results/dialogue_madrs_synthetic/test --device cuda
```

The supplied workbook (SHA-256
`6c19a75a9cc91b8b048dfc0bf14e1dc665ce5007875cf9092867a14e5a9c1cd1`)
produced 539 samples across 60 subjects: 377 train (42 subjects), 81
validation (9), and 81 test (9). The best validation checkpoint was epoch 1;
early stopping occurred after epoch 3. On the test split it obtained MAE
0.418, RMSE 0.635, Pearson 0.960, 70.4% exact rounded-score accuracy, and
97.5% within-one accuracy. The original checkpoint on that same split and
input format had MAE 0.816 and 48.1% exact accuracy. A per-item training-mean
baseline had MAE 2.046 and 13.6% exact accuracy. Per-item results are in
`data/results/dialogue_madrs_synthetic/test/metrics.json` (nine test samples
per item).

The checkpoint authors also distribute a file named `SyntheticData.xlsx` and
report using synthetic interviews to train MADRS-BERT. We have not established
whether this local file is byte-for-byte identical, but overlap with the
starting checkpoint's training data is plausible. Therefore these test metrics
measure this local adaptation and **must not be presented as independent
generalization to new patients**. The workbook labels are synthetic and do
not establish clinical validity. A new, independently sourced and rated corpus
is needed for that claim.

## 1. Prepare labels and patient-only samples

Create a CSV or JSONL label file with one row per session/item:

```csv
session_id,patient_id,symptom,severity_score
S001,P001,Reported Sadness,4
S001,P001,Reduced Sleep,2
```

`patient_id` is optional. If present, it must be present on every row and the
split is patient-grouped; otherwise it is session-grouped. A label row may also
provide `transcript_path`. Without it, preprocessing reads
`<transcripts-root>/<session_id>/transcript.jsonl`. Supported transcript shapes
are the dyadic-sim `patient_text` field and speaker/role rows whose speaker is
patient/client/participant/subject. Therapist text is never copied.

```powershell
python preprocess_dialogue_madrs.py --transcripts-root data/sessions --labels labels.csv --output-dir data/dialogue_madrs
```

The command writes `train.jsonl`, `validation.jsonl`, and `test.jsonl`, keeping
all symptoms from the same patient/session in one split.

## 2. Train and evaluate

```powershell
python train_dialogue_madrs.py --data-dir data/dialogue_madrs --output-dir runs/dialogue_madrs
python evaluate_dialogue_madrs.py --model runs/dialogue_madrs/best_model --test-data data/dialogue_madrs/test.jsonl --output-dir runs/dialogue_madrs/test
```

Training saves the best validation-MAE checkpoint, a last checkpoint, optimizer
and scheduler state, and epoch history. Early stopping defaults to three stale
epochs. Evaluation writes predictions plus overall and per-symptom MAE, RMSE,
Pearson correlation, exact rounded-score accuracy, and within-one accuracy.
Pearson is `null` when fewer than two observations or a constant series makes
the correlation undefined.

## 3. Inference

```powershell
python infer_dialogue_madrs.py --model runs/dialogue_madrs/best_model --symptom "Reduced Sleep" --dialogue "I wake several times and barely sleep."
```

This is a research model, not a diagnostic or clinical decision tool.
