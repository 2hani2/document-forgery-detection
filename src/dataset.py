import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

VALID_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp"}


class CasiaDataset(Dataset):
    """
    Loads CASIA v2.0 from the structure:
        casia_v2/CASIA2/Au/*.jpg          -> label 0 (authentic)
        casia_v2/CASIA2/Tp/*.jpg/.tif     -> label 1 (tampered)
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # list of (filepath, label)

        au_dir = self.root_dir / "Au"
        tp_dir = self.root_dir / "Tp"

        for f in au_dir.iterdir():
            if f.suffix.lower() in VALID_EXTS:
                self.samples.append((f, 0))

        for f in tp_dir.iterdir():
            if f.suffix.lower() in VALID_EXTS:
                self.samples.append((f, 1))

        print(f"[CasiaDataset] Loaded {len(self.samples)} images "
              f"({sum(1 for _, l in self.samples if l == 0)} authentic, "
              f"{sum(1 for _, l in self.samples if l == 1)} tampered)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)