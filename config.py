import os
import torch

# ============================================================
# Device
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Image Parameters
# ============================================================
IMG_SIZE = 224

# ============================================================
# Training Hyperparameters
# ============================================================
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_GLOBAL_ROUNDS = 10
LOCAL_EPOCHS = 1

# ============================================================
# Model
# ============================================================
MODEL_NAME = "google/vit-base-patch16-224-in21k"
NUM_CLASSES = 2

# ============================================================
# Output Directories
# ============================================================
RESULTS_DIR = "Results"

PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
ROC_DIR = os.path.join(RESULTS_DIR, "roc")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrix")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")
XAI_DIR = os.path.join(RESULTS_DIR, "ExplainableAI")

for directory in [
    RESULTS_DIR,
    PLOTS_DIR,
    ROC_DIR,
    CM_DIR,
    CSV_DIR,
    XAI_DIR,
]:
    os.makedirs(directory, exist_ok=True)