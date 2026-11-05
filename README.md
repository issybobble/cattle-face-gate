# Cattle face gate prototype (CIS4016 Coursework 1)

Simulation only: HOG + linear SVM, threshold 0.30. No physical gate is controlled.

Dataset (not included): https://github.com/aideep1400/Cattely-Cattle-Face-Images-Dataset

## Setup
Python 3.14.4, then `pip install -r requirements.txt`. Run every script from the project root, with the dataset in `data/Cattely-Cattle-Face-Images-Dataset`.

## Run order
1. `python src/make_splits.py`
2. `python src/train_model.py`
3. `python src/evaluate_model.py`
4. `python src/select_threshold.py`
5. `python src/compare_thresholds.py`
6. `python src/evaluate_gate.py`
7. `python src/make_figures.py`
8. `python src/confidence_intervals.py`
9. `python src/identity_analysis.py`
10. `python src/recognise_cow.py`

`src/inspect_dataset.py` and `src/preview_dataset.py` are dataset helpers.

## Notes
- Paths in the CSVs are Windows-style.
- The `selected_threshold` column (0.40) in `evaluation_results.csv` is the original validation result; 0.30 was adopted later (see the article, Section III-B6).
- The dataset and trained model are not included.
