import os
from pathlib import Path
import torch
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
CHECKPOINT_DIR = ROOT_DIR / "outputs" / "checkpoints"
GRADCAM_DIR = ROOT_DIR / "outputs" / "gradcam_visualizations"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
BACKBONE = "resnet50"      # or "efficientnet_b0"
NUM_CLASSES = 2            # authentic vs tampered

