# train_incremental_naive.py

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from dataset import YCBSegformerDataset
from config import (
    NUM_CLASSES,
    IGNORE_INDEX,
    BASE_CLASS_IDS,
    NEW_CLASS_IDS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_dataloaders(batch_size=2, resize=(384, 384)):
    # val loaders for both phases
    val_base = YCBSegformerDataset(
        "splits/val_all.txt",
        phase="base",
        resize=resize,
    )
    val_new = YCBSegformerDataset(
        "splits/val_all.txt",
        phase="new",
        resize=resize,
    )

    train_new = YCBSegformerDataset(
        "splits/train_all.txt",
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
    print(f"{desc} mIoU (classes {eval_class_ids}): {miou:.4f}")
    return miou


def train_incremental(
    base_ckpt="checkpoints/segformer_ycb_base.pth",
    num_epochs=5,
    lr=6e-5,
    batch_size=2,
):
    train_new_loader, val_base_loader, val_new_loader = get_dataloaders(
        batch_size=batch_size
    )
    model = create_model_from_base(base_ckpt).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # ------------------------------------------------------------------ #
    # 1) BEFORE fine-tuning: evaluate base model on base + new classes
    # ------------------------------------------------------------------ #
    print("==> Evaluation BEFORE incremental fine-tuning")
    eval_base_classes = [0] + BASE_CLASS_IDS
    eval_new_classes = [0] + NEW_CLASS_IDS

    evaluate(
        model, val_base_loader, eval_base_classes,
        desc="[Before] Base-set"
    )
    evaluate(
        model, val_new_loader, eval_new_classes,
        desc="[Before] New-set"
    )

    # ------------------------------------------------------------------ #
    # 2) Incremental training on NEW classes only
    # ------------------------------------------------------------------ #
    print("\n==> Starting incremental fine-tuning on NEW classes...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_new_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(pixel_values=imgs)
            logits = outputs.logits

            # logits are small spatial size (e.g. 96x96); upsample to labels
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
        print(f"Epoch {epoch}: train (new classes) loss = {avg_loss:.4f}")

        # quick eval each epoch (optional)
        evaluate(
            model, val_new_loader, eval_new_classes,
            desc=f"[Epoch {epoch}] New-set"
        )

    # ------------------------------------------------------------------ #
    # 3) AFTER fine-tuning: evaluate again on base + new
    # ------------------------------------------------------------------ #
    print("\n==> Evaluation AFTER incremental fine-tuning")
    base_miou_after = evaluate(
        model, val_base_loader, eval_base_classes,
        desc="[After] Base-set"
    )
    new_miou_after = evaluate(
        model, val_new_loader, eval_new_classes,
        desc="[After] New-set"
    )

    torch.save(model.state_dict(), "checkpoints/segformer_ycb_incremental_naive.pth")
    print("Saved incremental model to segformer_ycb_incremental_naive.pth")

    return base_miou_after, new_miou_after


if __name__ == "__main__":
    train_incremental()
