import os
import torch
from pathlib import Path

IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    DATA_ROOT = Path("/kaggle/input/datasets/divg07/casia-20-image-tampering-detection-dataset/CASIA2")
    CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
    GRADCAM_DIR = Path("/kaggle/working/gradcam_visualizations")
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_ROOT = ROOT_DIR / "data" / "raw" / "casia_v2" / "CASIA2"
    CHECKPOINT_DIR = ROOT_DIR / "outputs" / "checkpoints"
    GRADCAM_DIR = ROOT_DIR / "outputs" / "gradcam_visualizations"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2 if IS_KAGGLE else 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
NUM_CLASSES = 2

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")