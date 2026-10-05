# Deep Learning for Automatic Classification of Lung Diseases from Chest X-ray Images

Bachelor's thesis, School of Electrical Engineering, University of Belgrade (September 2026).
Author: Jovana Simić · Supervisor: Assist. Prof. Vladimir Jocović

This work addresses automatic classification of chest X-ray images into three mutually exclusive classes:
**COVID-19**, **PNEUMONIA** and **NORMAL**.

The goal is not only high accuracy, but also answering two questions: **how reliable are the results, and what do the models base their decisions on?**

## What was done

- Comparison of **4 architectures** under identical conditions: a custom CNN trained from scratch, ResNet50, DenseNet121 and EfficientNet-B0 (transfer learning).
- **Bayesian hyperparameter optimization** (KerasTuner, 20 trials per architecture) for every architecture.
- **Reliability assessment**: every configuration trained 3 times with different random seeds (mean ± standard deviation).
- **Decision interpretation** with Grad-CAM, plus a controlled experiment that removes regions outside the chest.
- **16 configurations** in total (4 architectures × optimized or not × full or cropped images).

## Key results

Test set (1,288 images). A trivial classifier that always predicts the majority class reaches 66.38% accuracy.

| Configuration | Accuracy | Macro F1 | COVID-19 recall |
|---|---|---|---|
| ResNet50 (optimized, cropped images) | **0.9643** | **0.9638** | 0.9741 |
| DenseNet121 (optimized, full images) | 0.9612 | 0.9622 | 0.9828 |
| ResNet50 (optimized, full images) | 0.9573 | 0.9579 | 0.9741 |
| EfficientNet-B0 (optimized, full images) | 0.9565 | 0.9573 | 0.9741 |

Best configuration for the COVID-19 class: precision **1.00**, recall **0.9741** (113 of 116 cases found, no false alarms).

### Reliability (macro F1, 3 training runs)

| Architecture | Default hyperparameters | Optimized |
|---|---|---|
| DenseNet121 | 0.9459 ± 0.0038 | 0.9658 ± 0.0039 |
| EfficientNet-B0 | 0.9278 ± 0.0053 | 0.9658 ± 0.0030 |
| ResNet50 | 0.9338 ± 0.0100 | 0.9594 ± 0.0054 |
| CNN (from scratch) | 0.8165 ± 0.1744 | 0.9379 ± 0.0075 |

## Main findings

1. **Hyperparameter optimization** improves every architecture tested, by more than the measured variance. Without it, the ranking of architectures would be different.
2. **Transfer learning is much more stable**: the network trained from scratch shows variance two orders of magnitude larger (before optimization, one of three runs essentially failed).
3. **DenseNet121 and EfficientNet-B0 cannot be reliably distinguished** (same mean 0.9658, difference smaller than the variance).
4. **Grad-CAM** shows that for the NORMAL class the models focus outside lung tissue (upper mediastinum, area above the clavicles). Removing those regions does not hurt performance, which points to **possible dataset bias** (shortcut learning).
5. High metric values alone are not sufficient evidence that a system is usable.

## Dataset

The public [Chest X-ray (Covid-19 & Pneumonia)](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia) dataset from Kaggle is used (6,432 images).

| Class | Train | Test | Total |
|---|---|---|---|
| COVID19 | 460 | 116 | 576 |
| NORMAL | 1,266 | 317 | 1,583 |
| PNEUMONIA | 3,418 | 855 | 4,273 |

15% of the training part was split off (stratified) for validation. The test set was not used for any decision during development.
**The dataset is not included in this repository**, download it from the link above.

## Technologies

Python 3.12 · TensorFlow 2.20 · Keras 3.13 · KerasTuner 1.4 · scikit-learn 1.5 · NumPy · pandas · Matplotlib

Experiments were run on Kaggle (Tesla P100 / T4 GPUs).

## Code structure

| Module | Purpose |
|---|---|
| `paths` | locating the dataset and output directory |
| `data` | indexing, splitting, preprocessing, class weights |
| `models` | definitions of all four architectures |
| `trening` | two-phase model training |
| `optimizacija` | Bayesian hyperparameter search |
| `evaluacija` | metrics, confusion matrices, ROC curves |
| `gradcam` | model decision interpretation |
| `visestruko` | multi-seed training and stability analysis |
| `poredjenje_isecanja` | comparing cropped vs. full-image variants |

All hyperparameters live in a single configuration file, so the whole experiment can be reproduced by editing one file.

> Adjust module and file names to match the actual repository structure.

## Getting started

```bash
git clone https://github.com/YOUR_USERNAME/chest-xray-covid-pneumonia-classification.git
cd chest-xray-covid-pneumonia-classification
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it in the `data/` folder, then run the scripts in order:
data indexing and checks → training → optimization → evaluation → Grad-CAM → multi-seed training.

> Fill in the exact commands for your scripts.

## Limitations

- No external validation on independent clinical data, so results apply only to the dataset used.
- The dataset has no patient identifiers, so images of the same patient may fall into different subsets.
- The origin of the labels (diagnoses) is unknown.
- Three training runs are not enough to distinguish the two best architectures.
- Grad-CAM is not applicable to the CNN trained from scratch.
- Cropping reduces the input area, so higher effective resolution is an alternative explanation for the improvement.

**This system is not a diagnostic tool and must not be used in clinical practice.**
