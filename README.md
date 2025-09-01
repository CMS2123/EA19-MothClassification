**Project overview:** A compact, reproducible baseline for **moth species recognition (50 classes)** that pairs strong accuracy with **transparent evaluation**—calibration, robustness, and visual explanations—plus ready-to-run code and artifacts.

## What this repo contains
- **Notebook pipeline (Colab/GPU-friendly)** that trains an **EfficientNet-B0** classifier on 50 moth species and saves all outputs (figures/CSVs/arrays).
- **Full evaluation suite**
  - Accuracy, per-class precision/recall/F1 and a **row-normalised confusion matrix**
  - **Calibration**: ECE/MCE/Brier and **temperature scaling** (T* fitted on **VAL**, applied to **TEST**)
  - **Robustness** to JPEG compression, Gaussian noise, Gaussian blur (severity sweeps)
  - **Selective prediction** (accuracy–coverage curves)
  - **Explainability**: Grad-CAM/Grad-CAM++, plus **deletion** and **occlusion** sanity checks
  - **Embeddings**: UMAP for both backbones; PCA/t-SNE for MobileNetV2 (baseline)
- **One-click export** of all artifacts to a zip for easy sharing.

## Data
- Dataset: **Moths Image Dataset – Classification (50 species)** on Kaggle.  
- Expected layout (already used by the notebook):
  ```text
  split_data/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
    test/<class_name>/*.jpg
  ```
  The loader uses:
  ```python
  tf.keras.preprocessing.image_dataset_from_directory(
      "split_data/train", image_size=(224,224), batch_size=32,
      label_mode="categorical", shuffle=True
  )
  # val/test use shuffle=False for stable filenames ↔ predictions
  ```

## Environment
- Runs on **Colab** or locally with Python 3.10+.
- Suggested pins (add to `requirements.txt` if running locally):
  ```text
  tensorflow==2.17.0
  keras==3.4.1
  numpy==1.26.4
  pandas==2.2.2
  scikit-learn==1.5.1
  umap-learn==0.5.6
  matplotlib==3.8.4
  pillow==10.3.0
  scipy==1.13.1
  ```

## Quickstart — Notebook (end-to-end)
1. **Open the notebook** (`ERP_code_FINAL.ipynb`; the Colab PDF mirrors the flow).
2. **Ensure data present** under `split_data/{train,val,test}/…` as above.
3. **Run training**
   - Backbone: **EfficientNet-B0** (`include_top=False`)
   - Preprocessing: **`tf.keras.applications.efficientnet.preprocess_input` inside the model**
   - Head: `GlobalAveragePooling2D → Dropout(0.2) → Dense(50, softmax)`
   - Optimiser/Loss: **Adam + categorical_crossentropy**, **class weights** computed from training counts
4. **Run evaluations**
   - Confusion matrix, per-class report
   - Calibration: reliability diagram before/after **temperature scaling** (T* fitted on **VAL**, applied to **TEST**; accuracy unchanged by design)
   - Robustness: accuracy vs severity for **JPEG / noise / blur**
   - Selective prediction: **accuracy–coverage** from confidence threshold sweep
   - Explainability: **Grad-CAM/Grad-CAM++**, **deletion**, **occlusion**
5. **(Optional)** run the **export cell** to create `erp_artifacts_bundle.zip` with all figures/CSVs/arrays used in the report.

## What the code actually does (to keep wording aligned)
- **Preprocessing** is embedded in-graph (EfficientNet’s `preprocess_input`) so **train and test see the exact same transform**.
- **Labels** are **one-hot**; loss is **categorical cross-entropy**.
- **Calibration** fits temperature on **validation NLL** and applies it to test (ECE/MCE/Brier updated; accuracy unchanged).
- **UMAP** is run for **both** backbones; **PCA/t-SNE** are shown for **MobileNetV2** as qualitative baselines.
- The notebook can **zip all artifacts** at the end for easy archiving.

## Outputs
- `figs_final/` — confusion, reliability (before/after T*), robustness curves, deletion/occlusion bars, Grad-CAM panels, UMAP/PCA/t-SNE.
- `tables/` — per-class classification report(s), optional overall metrics.
- `npys/` — cached arrays (e.g., `test_probs.npy`, `test_labels.npy`) enabling figure regeneration without retraining.
- `erp_artifacts_bundle.zip` — one-click archive of the above.

## Optional: reproduce key figures **without training**
If you add the helper scripts under `tools/`, you can regenerate figures from cached arrays:

```bash
# Print headline metrics from a saved model + splits
python tools/restore_and_eval.py   --weights artifacts/models/efficientnet_b0_best.keras   --splits splits/   --classes artifacts/class_index.json

# Rebuild confusion matrix from cached predictions
python tools/make_figs_from_cache.py   --probs artifacts/preds/test_probs.npy   --labels artifacts/preds/test_labels.npy   --classes artifacts/class_index.json   --out figs_final/
```

> If you publish artifacts via **GitHub Releases**, include checksums in `artifacts/manifest.json` so results can be verified.

## Intended use, licensing & data rights
- **Intended use:** decision support; apply a calibrated confidence threshold to auto‑label when confident and **defer** uncertain cases.
- **Code license:** MIT (see `LICENSE`).
- **Data/images:** follow the **Kaggle dataset terms**; do **not** redistribute original images. Derived model weights/arrays inherit the dataset’s usage constraints (see `DATASET.md`).
- See `MODEL_CARD.md` for performance (Top‑1/F1), calibration, robustness, limitations, and threshold guidance.
