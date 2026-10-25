# Evidence-Fit

A small, auditable ARC-AGI-2 baseline. It enumerates geometry, cropping,
components, repetition, compression and gravity hypotheses, keeps only models
that exactly reproduce demonstration pairs, and emits two valid attempts.
Color mappings are inferred exclusively from demonstration pairs.

No task-ID lookup, external API, weights or test-output labels are used.
It is a limited baseline, not a claim of frontier performance or a prize forecast.

## Tests
python3 -m unittest -v

## Build the offline notebook
python3 make_notebook.py
Attach the official ARC-AGI-2 competition data in Kaggle, turn internet off and
run all cells. The notebook writes /kaggle/working/submission.json.
The actual current competition schema must still be checked in the submission UI.

## Public-data evaluation protocol
The catalog is fixed before public evaluation results are inspected.
Synthetic unit tests do not use ARC task examples.
The evaluation runner removes test-output fields before invoking the solver.
Both per-output exact match and all-test task accuracy are reported.
Public-data scores are not private leaderboard or judge results.

## Observed result, 25 October 2026
Code was frozen at SHA-256 574b3b9cba9c5b4638e7f7b2c953dc4766b2b20b10c0d080bd664d01aed55b3e before evaluating.
Official data archive: arcprize-ARC-AGI-2-f3283f7.
On the public training split, 39/1,076 test outputs matched (3.62%); 33/1,000 tasks
were fully correct. On public evaluation, 0/167 outputs and 0/120 tasks matched.
Training-only recoloring ablation matched 33/1,076 outputs (3.07%).
These results demonstrate a severe expressivity limit, not competitive accuracy.
Full machine-readable reports and the frozen protocol are in results/.
All 11 synthetic tests passed. The packaged notebook solver was also executed on
a synthetic fixture. Its Kaggle input adapter and judge execution are unverified.

## Sources
Dataset: https://github.com/arcprize/ARC-AGI-2
Contest: https://arcprize.org/competitions/2026/arc-agi-2
Paper linkage: https://arcprize.org/competitions/2026/paper

Implementation and documentation are AI-assisted. Applicant identity, acceptance
of rules and a confirmed Kaggle submission remain separate requirements.
