import os
import json
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from dotenv import load_dotenv, dotenv_values

from utils.dataset import YCBDataset
from utils.logger import get_logger
from utils.config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = get_logger(LOG_FILE_PATH)

def get_dataloaders(batch_size=BATCH_SIZE, resize=RESIZE):
    train_ds = YCBDataset(TRAIN_SPLIT, phase="base", resize=resize)
    val_ds = YCBDataset(VAL_SPLIT, phase="base", resize=resize)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def create_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.config.ignore_index = IGNORE_INDEX
    return model


def compute_miou(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    mask = target != IGNORE_INDEX
    pred = pred[mask]
    target = target[mask]

    ious = []
    eval_classes = [0] + BASE_CLASS_IDS
    for cls in eval_classes:
        pred_i = pred == cls
        target_i = target == cls
        intersection = (pred_i & target_i).sum().item()
        union = (pred_i | target_i).sum().item()
        if union > 0:
            ious.append(intersection / (union + 1e-6))

    return sum(ious) / len(ious) if ious else 0.0


def train():
    train_loader, val_loader = get_dataloaders()
    model = create_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    results = {
        "config": {
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "resize": RESIZE,
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
            "base_class_ids": BASE_CLASS_IDS,
        },
        "epochs": []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(pixel_values=imgs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # validation
        model.eval()
        miou = 0.0
        n_batches = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(pixel_values=imgs).logits
                logits_up = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                pred = logits_up.argmax(1)

                miou += compute_miou(pred.cpu(), labels.cpu())
                n_batches += 1

        miou /= max(n_batches, 1)
        logger.info(f"Epoch {epoch}: train loss = {avg_loss:.4f} |  val mIoU = {miou:.4f}")

        results["epochs"].append({
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "val_miou": float(miou),
        })

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    logger.info(f"Saved model to {CHECKPOINT_PATH}")

    os.makedirs("logs", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to {RESULTS_PATH}")


if __name__ == "__main__":
    train()
