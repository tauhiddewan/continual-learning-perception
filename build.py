import os
from glob import glob
import random

DATA_DIR = "./data"  

os.makedirs("splits", exist_ok=True)
train_list_path = "splits/train_all.txt"
val_list_path   = "splits/val_all.txt"

print("Looking in:", os.path.abspath(DATA_DIR))

pairs = []
for seq in sorted(os.listdir(DATA_DIR)):
    seq_dir = os.path.join(DATA_DIR, seq)
    if not os.path.isdir(seq_dir):
        continue
    # your files look like: 000001-color.jpg
    color_paths = sorted(glob(os.path.join(seq_dir, "*-color.jpg")))
    for cpath in color_paths:
        # labels are .png with same stem
        lpath = cpath.replace("-color.jpg", "-label.png")
        if os.path.exists(lpath):
            pairs.append((cpath, lpath))

print("Total frames found:", len(pairs))

if not pairs:
    print("WARNING: no pairs found – check DATA_DIR and file patterns.")
    exit()

random.shuffle(pairs)
split_idx = int(0.9 * len(pairs))
train_pairs = pairs[:split_idx]
val_pairs   = pairs[split_idx:]

with open(train_list_path, "w") as f:
    for c, l in train_pairs:
        f.write(f"{c} {l}\n")

with open(val_list_path, "w") as f:
    for c, l in val_pairs:
        f.write(f"{c} {l}\n")

print("Wrote:")
print(" ", train_list_path, len(train_pairs))
print(" ", val_list_path, len(val_pairs))
