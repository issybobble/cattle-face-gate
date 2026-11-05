# Cattle face gate prototype (CIS4016 Coursework 1)
   Simulation only: HOG + linear SVM, threshold 0.30. No physical gate.
   Dataset (Cattely, not included): https://github.com/aideep1400/Cattely-Cattle-Face-Images-Dataset
   Setup: Python 3.14.4, `pip install -r requirements.txt`. Run scripts from the project root.
   Order: make_splits, train_model, evaluate_model, select_threshold, compare_thresholds,
   evaluate_gate, make_figures, src/confidence_intervals, src/identity_analysis, recognise_cow.
   Note: Windows-style paths. The selected_threshold column (0.40) in
   evaluation_results.csv is the original validation result; 0.30 was adopted later.
