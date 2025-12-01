import os
import json
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from utils.dataset import YCBDataset
from utils.config import *
from utils.logger import get_logger

logger = get_logger(NAIVE_LOG_FILE_PATH)


def get_dataloaders(batch_size=BATCH_SIZE, resize=RESIZE):
    # val loaders for both phases
    val_base = YCBDataset(
        VAL_SPLIT,
        phase="base",
        resize=resize,
    )
    val_new = YCBDataset(
        VAL_SPLIT,
        phase="new",
        resize=resize,
    )

    train_new = YCBDataset(
        TRAIN_SPLIT,
        phase="new",
        resize=resize,
    )

    train_new_loader = DataLoader(
        train_new, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_base_loader = DataLoader(
        val_base, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    val_new_loader = DataLoader(
        val_new, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    return train_new_loader, val_base_loader, val_new_loader


def create_model_from_base(ckpt_path: str):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.config.ignore_index = IGNORE_INDEX
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def compute_miou(pred, target, num_classes, ignore_index, eval_class_ids):
    """
    pred, target: (N, H, W)
    eval_class_ids: list of class ids to include in mIoU (e.g. [0]+BASE_CLASS_IDS)
    """
    pred = pred.view(-1)
    target = target.view(-1)

    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    ious = []
    for cls in eval_class_ids:
        pred_i = pred == cls
        target_i = target == cls
        intersection = (pred_i & target_i).sum().item()
        union = (pred_i | target_i).sum().item()
        if union == 0:
            continue
        ious.append(intersection / (union + 1e-6))
    if not ious:
        return 0.0
    return sum(ious) / len(ious)


def evaluate(model, loader, eval_class_ids, desc=""):
    model.eval()
    miou = 0.0
    n_batches = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(pixel_values=imgs)
            logits = outputs.logits  # (B, C, h', w')

            # upsample to label size
            logits_up = F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pred = logits_up.argmax(1)  # (B, H, W)

            miou += compute_miou(
                pred.cpu(), labels.cpu(),
                NUM_CLASSES, IGNORE_INDEX,
                eval_class_ids,
            )
            n_batches += 1

    miou = miou / max(n_batches, 1)
    logger.info(f"{desc} mIoU (classes {eval_class_ids}): {miou:.4f}")
    return miou


def train_incremental(
    base_ckpt=CHECKPOINT_PATH,  # you can also import CHECKPOINT_PATH from config
    num_epochs=NUM_EPOCHS,      # or pass directly
    lr=NAIVE_LR,
    batch_size=BATCH_SIZE,
):
    from utils.config import CHECKPOINT_PATH, NUM_EPOCHS  # to avoid circular import at top

    train_new_loader, val_base_loader, val_new_loader = get_dataloaders(
        batch_size=batch_size
    )
    model = create_model_from_base(base_ckpt).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    eval_base_classes = [0] + BASE_CLASS_IDS
    eval_new_classes = [0] + NEW_CLASS_IDS

    # JSON log structure
    results = {
        "config": {
            "base_ckpt": base_ckpt,
            "num_epochs": num_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "resize": RESIZE,
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
            "base_class_ids": BASE_CLASS_IDS,
            "new_class_ids": NEW_CLASS_IDS,
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT,
        },
        "before": {},
        "epochs": [],
        "after": {},
    }

    # 1) BEFORE fine-tuning
    logger.info("==> Evaluation BEFORE incremental fine-tuning")
    base_miou_before = evaluate(
        model, val_base_loader, eval_base_classes,
        desc="[Before] Base-set"
    )
    new_miou_before = evaluate(
        model, val_new_loader, eval_new_classes,
        desc="[Before] New-set"
    )

    results["before"] = {
        "base_miou": float(base_miou_before),
        "new_miou": float(new_miou_before),
    }

    # 2) Incremental training on NEW classes only
    logger.info("\n==> Starting incremental fine-tuning on NEW classes...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_new_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(pixel_values=imgs)
            logits = outputs.logits

            logits_up = F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = criterion(logits_up, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(train_new_loader.dataset)
        logger.info(f"Epoch {epoch}: train (new classes) loss = {avg_loss:.4f}")

        new_miou_epoch = evaluate(
            model, val_new_loader, eval_new_classes,
            desc=f"[Epoch {epoch}] New-set"
        )

        results["epochs"].append({
            "epoch": epoch,
            "train_loss_new": float(avg_loss),
            "val_miou_new": float(new_miou_epoch),
        })

    # 3) AFTER fine-tuning
    logger.info("\n==> Evaluation AFTER incremental fine-tuning")
    base_miou_after = evaluate(
        model, val_base_loader, eval_base_classes,
        desc="[After] Base-set"
    )
    new_miou_after = evaluate(
        model, val_new_loader, eval_new_classes,
        desc="[After] New-set"
    )

    results["after"] = {
        "base_miou": float(base_miou_after),
        "new_miou": float(new_miou_after),
    }

    os.makedirs(os.path.dirname(NAIVE_CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), NAIVE_CHECKPOINT_PATH)
    logger.info(f"Saved incremental model to {NAIVE_CHECKPOINT_PATH}")

    os.makedirs(os.path.dirname(NAIVE_RESULTS_PATH), exist_ok=True)
    with open(NAIVE_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to {NAIVE_RESULTS_PATH}")

    return base_miou_after, new_miou_after


if __name__ == "__main__":
    from utils.config import CHECKPOINT_PATH, NUM_EPOCHS
    train_incremental(base_ckpt=CHECKPOINT_PATH, num_epochs=NUM_EPOCHS)
