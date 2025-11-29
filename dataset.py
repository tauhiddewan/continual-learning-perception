import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from config import BASE_CLASS_IDS, NEW_CLASS_IDS, IGNORE_INDEX

def mask_labels(label_np, keep_ids, ignore_index=IGNORE_INDEX):
    """
    label_np: HxW, values 0..21
    keep_ids: list of ids we want to learn in this phase (including 0).
    Pixels with other ids -> ignore_index.
    """
    keep_ids = set(keep_ids)
    out = label_np.copy()
    mask_keep = np.isin(out, list(keep_ids))
    out[~mask_keep] = ignore_index
    return out

class YCBSegformerDataset(Dataset):
    def __init__(self, list_file, phase="base", transform=None, resize=(384, 384)):
        """
        list_file: text file with "image_path label_path" per line.
        phase: "base" or "new".
        resize: (W,H) to resize both image and label.
        """
        assert phase in ["base", "new"]
        self.phase = phase
        self.transform = transform
        self.resize = resize

        self.samples = []
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"List file not found: {list_file}")
        with open(list_file, "r") as f:
            for line in f:
                img_path, lbl_path = line.strip().split()
                self.samples.append((img_path, lbl_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path)

        if self.resize is not None:
            img = img.resize(self.resize, Image.BILINEAR)
            label = label.resize(self.resize, Image.NEAREST)

        img_np = np.array(img).astype(np.float32) / 255.0   # H,W,3
        label_np = np.array(label).astype(np.int64)         # H,W

        if self.phase == "base":
            keep_ids = [0] + BASE_CLASS_IDS
        else:
            keep_ids = [0] + NEW_CLASS_IDS

        label_np = mask_labels(label_np, keep_ids)

        img_t = torch.from_numpy(img_np).permute(2, 0, 1)   # C,H,W
        label_t = torch.from_numpy(label_np)                # H,W

        if self.transform is not None:
            # if you later add Albumentations or torchvision transforms, apply here
            pass

        return img_t, label_t
