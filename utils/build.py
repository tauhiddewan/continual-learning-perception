import os
import random
from glob import glob
from config import *

DATA_DIR = "./data"  

os.makedirs("outputs/splits", exist_ok=True)

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

with open(TRAIN_SPLIT, "w") as f:
    for c, l in train_pairs:
        f.write(f"{c} {l}\n")

with open(VAL_SPLIT, "w") as f:
    for c, l in val_pairs:
        f.write(f"{c} {l}\n")

print("Wrote:")
print(" ", TRAIN_SPLIT, len(train_pairs))
print(" ", VAL_SPLIT, len(val_pairs))
