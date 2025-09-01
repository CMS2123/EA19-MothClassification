# Model Card — EfficientNet-B0 (50 moth species)

**Intended use:** decision support; low-confidence cases deferred to a human.
**Data:** Kaggle moths (50 species), 60/20/20 split (seed=42).
**Performance (test):** Top-1 ≈ 0.98; macro/weighted-F1 ≈ 0.98.
**Calibration:** temp. scaling lowers ECE/Brier; accuracy unchanged.
**Robustness:** most sensitive to noise/blur; mild to JPEG compression.
**Limitations:** controlled images; field generalisation not guaranteed; confusable pairs remain.
**Safety & rights:** no personal data; images subject to dataset terms; weights/arrays inherit dataset restrictions.
