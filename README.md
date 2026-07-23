# Real-Time Radiometric Thermal Fire Detection with OTFAN

This repository contains the code, trained model, and evaluation artifacts for **OTFAN (Optimized Thermal Fire Attention Network)** — a lightweight CNN for wildfire detection from **radiometric thermal TIFF imagery**, built on a MobileNetV4 Conv Small 0.35 backbone with a custom **Optimized Thermal Channel Attention (OTCA)** module.

> **Paper:** *Real-Time Radiometric Thermal Fire Detection with OTFAN: Accuracy Meets Deployability*
> Ahmed Mabood, Saahil Shahzad — National University of Sciences and Technology (NUST), Islamabad
> **Dataset:** [FLAME-3](https://arxiv.org/abs/2412.02831) thermal subset (Sycan Marsh site), 738 images (622 fire / 116 no-fire)

## What OTFAN does

OTFAN takes 32-bit radiometric thermal TIFFs (per-pixel temperature data, not ordinary grayscale images) and classifies each frame as **fire** or **no-fire**. Two things make it different from typical fire-detection CNNs:

1. **Radiometric-aware preprocessing** — a six-step pipeline (invalid-value handling → percentile clipping → 8-bit normalization → pseudo-RGB stacking → resize → ImageNet normalization) that turns raw temperature arrays into inputs a standard ImageNet-pretrained backbone can use, without destroying the thermal signal in the process.
2. **OTCA (Optimized Thermal Channel Attention)** — a channel-attention module with a three-layer bottleneck MLP (deeper than a standard squeeze-and-excitation block) that recalibrates feature channels for thermal-specific cues.

### Headline results (stratified 5-fold cross-validation, FLAME-3 thermal subset)

| Metric | Mean ± Std |
|---|---|
| Accuracy | 96.75% ± 1.20 |
| Precision | 97.02% ± 0.74 |
| Recall | 99.20% ± 0.68 |
| F1-score | 98.09% ± 0.71 |

| Efficiency | Value |
|---|---|
| Parameters | 842,146 (3.51 MB) |
| MFLOPs / inference | 46.68 |
| Inference speed (NVIDIA T4 GPU) | ~201 FPS |
| Inference speed (CPU) | ~63 FPS |

## Repository structure

```
realtime-forest-fire-detection/
├── cpu_run/
│   ├── model_pipeline.ipynb        # End-to-end pipeline: preprocessing, OTFAN model, training,
│   │                                #   5-fold CV, evaluation — run on CPU (used for FPS/latency
│   │                                #   benchmarking on CPU hardware)
│   ├── cross_validation_results.csv   # Per-fold accuracy/precision/recall/F1 (CPU run)
│   ├── efficiency_metrics.csv         # Params, FLOPs, model size, FPS (CPU run)
│   └── confusion_matrices.npy         # Normalized confusion matrix per fold (CPU run)
│
├── gpu_run/
│   ├── model_pipeline.ipynb        # Same pipeline as cpu_run, executed on GPU (used for the
│   │                                #   headline accuracy/F1 numbers and GPU FPS reported in the paper)
│   ├── cross_validation_results.csv
│   ├── efficiency_metrics.csv
│   └── confusion_matrices.npy
│
└── thermal_fire_detection_app/
    ├── app.py                      # Streamlit demo app: upload a radiometric thermal TIFF,
    │                                #   see it converted to pseudo-RGB and classified live
    ├── otfan_fire_detection.pth    # Trained OTFAN model weights
    └── requirements.txt            # Dependencies for the app / pipeline notebooks
```

**Note:** `cpu_run` and `gpu_run` contain the *same* modeling pipeline (data loading → preprocessing → OTFAN + OTCA → 5-fold stratified CV → metrics/efficiency logging) run on two different hardware targets, so you can compare accuracy/consistency across runs and see how inference speed changes between CPU and GPU. The numbers in Table III of the paper come from `gpu_run`; the CPU FPS figure (63.36 FPS) comes from `cpu_run`.

## Getting started

### 1. Clone the repo
```bash
git clone https://github.com/Saahil-Shahzad/realtime-forest-fire-detection.git
cd realtime-forest-fire-detection
```

### 2. Set up the environment
```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r thermal_fire_detection_app/requirements.txt
```

### 3. Get the data
The FLAME-3 radiometric thermal subset used for training/evaluation is available at the Google Drive link in the paper. Download it and update the data path at the top of whichever `model_pipeline.ipynb` you run.

### 4. Reproduce training / evaluation
Open either notebook in Jupyter (or Google Colab) and run top to bottom:
```bash
jupyter notebook gpu_run/model_pipeline.ipynb   # for GPU
jupyter notebook cpu_run/model_pipeline.ipynb   # for CPU
```
Each notebook performs stratified 5-fold cross-validation and writes out `cross_validation_results.csv`, `efficiency_metrics.csv`, and `confusion_matrices.npy`.

### 5. Try the live demo app
```bash
cd thermal_fire_detection_app
streamlit run app.py
```
Upload a radiometric thermal `.tiff`/`.tif` file — the app applies the same preprocessing pipeline described in the paper and runs it through the pretrained OTFAN weights (`otfan_fire_detection.pth`) to return a fire / no-fire prediction.

## Citation

If you use this code or model, please cite:

```bibtex
@article{mabood2026otfan,
  title   = {Real-Time Radiometric Thermal Fire Detection with OTFAN: Accuracy Meets Deployability},
  author  = {Mabood, Ahmed and Shahzad, Saahil},
  year    = {2026},
  note    = {National University of Sciences and Technology (NUST), Islamabad}
}
```

## Acknowledgments

This work uses the [FLAME-3](https://arxiv.org/abs/2412.02831) dataset (Hopkins et al.). We thank the dataset creators for making the radiometric thermal imagery publicly available.
