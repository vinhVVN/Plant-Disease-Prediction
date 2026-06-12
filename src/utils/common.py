import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def save_class_names(class_names, path):
    with open(path, "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]