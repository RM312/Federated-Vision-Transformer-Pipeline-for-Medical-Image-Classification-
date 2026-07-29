# Federated Vision Transformer for Mpox Skin Lesion Classification

A modular PyTorch implementation of **Federated Learning (FL)** using a **Vision Transformer (ViT)** for automated **Mpox (Monkeypox) skin lesion classification**. The project employs the **FedAvg** aggregation algorithm to collaboratively train a global model from multiple distributed clients without sharing raw medical images, thereby preserving data privacy.

The implementation also includes **Grad-CAM** and **Grad-CAM++** based explainability to visualize the regions of skin lesion images that contribute most to the model's predictions.

---

## Features

- Federated Learning using **FedAvg**
- Vision Transformer (ViT-Base-Patch16-224)
- Multi-client distributed training
- Binary and Multi-class skin lesion classification
- Training and validation performance visualization
- ROC Curve generation
- Confusion Matrix generation
- Grad-CAM visualization
- Grad-CAM++ visualization
- Modular and reusable codebase
- Reproducible experimental pipeline

---

## Project Structure

```text
Federated_Transformer/
│
├── main.py
├── config.py
├── client_data.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── vit_model.py
│   ├── aggregation.py
│   └── model_utils.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── federated.py
│
├── visualization/
│   ├── plots.py
│   ├── confusion_matrix.py
│   └── roc_curve.py
│
├── explainability/
│   ├── hooks.py
│   ├── gradcam.py
│   ├── gradcampp.py
│   └── generate_heatmaps.py
│
├── utils/
│   ├── seed.py
│   ├── logger.py
│   ├── helpers.py
│   └── file_utils.py
│
├── Results/
│   ├── csv/
│   ├── plots/
│   ├── confusion_matrix/
│   ├── roc/
│   └── ExplainableAI/
│
└── client_data/
    ├── ClientA/
    └── ClientB/
```

---

# Methodology

The proposed framework consists of the following stages:

1. Data preprocessing and augmentation
2. Client-wise data partitioning
3. Vision Transformer initialization
4. Local client training
5. Federated Averaging (FedAvg)
6. Global model aggregation
7. Performance evaluation
8. Explainability using Grad-CAM and Grad-CAM++

---

# Datasets

This project utilizes two publicly available Mpox skin lesion datasets.

## Dataset 1 — Monkeypox Skin Lesion Dataset (MSLD)

- **Purpose:** Binary Classification
- **Classes:**
  - Monkeypox
  - Others (Chickenpox + Measles)

Dataset Link:

https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset

The original MSLD dataset was developed for binary classification of Mpox versus visually similar skin diseases and contains original as well as augmented skin lesion images. :contentReference[oaicite:0]{index=0}

---

## Dataset 2 — Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)

- **Purpose:** Multi-class Classification
- **Classes:**
  - Mpox
  - Chickenpox
  - Cowpox
  - Measles
  - Hand-Foot-Mouth Disease (HFMD)
  - Healthy

Dataset Link:

https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20

MSLD v2.0 extends the original dataset with six clinically relevant classes, curated images from diverse patients, predefined train/validation/test folds, and dermatologist verification. :contentReference[oaicite:1]{index=1}

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Federated_Transformer.git

cd Federated_Transformer
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

Download both datasets from Kaggle.

Organize the client folders as follows:

```text
client_data/

ClientA/
    train/
    val/
    test/

ClientB/
    train/
    val/
    test/
```

Update the dataset path inside

```python
client_data.py
```

if required.

---

# Running the Project

```bash
python main.py
```

---

# Output

After training, the following outputs are generated automatically.

```
Results/

csv/
plots/
roc/
confusion_matrix/
ExplainableAI/
```

Including

- Trained Federated ViT Model
- Training Loss
- Validation Loss
- Accuracy Curves
- Precision
- Recall
- F1-score
- MCC
- AUC
- ROC Curves
- Confusion Matrices
- Grad-CAM Heatmaps
- Grad-CAM++ Heatmaps

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- HuggingFace Transformers
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn
- tqdm

---

# Explainability

To improve model interpretability, the project integrates:

- Grad-CAM
- Grad-CAM++

These techniques highlight the image regions that most influence the Vision Transformer’s predictions, providing visual explanations that can assist in understanding model decisions.

---

# Citation

If you use this repository in your research, please cite the appropriate publications for the datasets and any related work associated with this implementation.

---

# License

This project is intended for research and educational purposes only.

Please follow the licensing terms associated with the original datasets before using them for commercial or clinical applications.

---

# Acknowledgements

We gratefully acknowledge the creators of:

- Monkeypox Skin Lesion Dataset (MSLD)
- Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)

for making these datasets publicly available to support research in computer-aided diagnosis of Mpox skin lesions. :contentReference[oaicite:2]{index=2}
