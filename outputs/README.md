# Local outputs

Run `scripts/evaluate.py` to compare all three methods and run the memory-bank sweep on the official test split. It writes:

- `outputs/metrics/<category>/`: comparison metrics, per-image CSV, sweep JSON/CSV, and `experiment_metadata.json`.
- `outputs/plots/<category>/research/`: image ROC, confusion matrices, score distributions, and the bank-sweep plot.
- `outputs/examples/<category>/research/`: ranked false-positive and false-negative figures plus a human-review `failure_analysis.md`.

`scripts/evaluate.py --smoke` writes to separate `smoke_test` folders and labels those outputs as non-research checks. The standalone `scripts/evaluate_detector.py` and `scripts/evaluate_classical.py` commands remain available for separate reports. Generated outputs are ignored by Git; this repository contains no sample metrics or screenshots before genuine data runs.
