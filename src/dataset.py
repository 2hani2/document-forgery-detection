from torch.utils.data import Subset, random_split
from sklearn.model_selection import train_test_split
import numpy as np


def get_stratified_splits(dataset, val_frac=0.1, test_frac=0.1, seed=42):
    """
    Returns train/val/test Subset objects, stratified by label,
    so authentic:tampered ratio is preserved in each split.
    """
    labels = np.array([label for _, label in dataset.samples])
    indices = np.arange(len(dataset))

    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_frac + test_frac),
        stratify=labels, random_state=seed
    )
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_frac / (val_frac + test_frac),
        stratify=temp_labels, random_state=seed
    )

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)