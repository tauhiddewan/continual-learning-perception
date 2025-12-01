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

logger = get_logger(LORA_LOG_FILE_PATH)

class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.orig = orig_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_dim = orig_linear.in_features
        out_dim = orig_linear.out_features

        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        for p in self.orig.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.orig(x) + self.scaling * self.lora_B(self.lora_A(x))


def getattr_module_by_name(model: nn.Module, full_name: str):
    parts = full_name.split(".")
    m = model
    for p in parts:
        if p.isdigit():
            m = m[int(p)]
        else:
            m = getattr(m, p)
    return m


def _get_parent_module(model: nn.Module, module_name: str):
    parts = module_name.split(".")
    attr_name = parts[-1]
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, attr_name


def inject_lora_into_segformer(model: nn.Module, r: int = 8, alpha: int = 16):
    replace_keywords = ["query", "key", "value", "proj"]

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(k in name for k in replace_keywords):
            parent, attr = _get_parent_module(model, name)
            orig_linear = getattr(parent, attr)
            setattr(parent, attr, LoRALinear(orig_linear, r=r, alpha=alpha))

    # Freeze everything, then selectively unfreeze LoRA + classifier
    for name, param in model.named_parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            for p in module.lora_A.parameters():
                p.requires_grad = True
            for p in module.lora_B.parameters():
                p.requires_grad = True

    if hasattr(model, "decode_head"):
        for name, p in model.decode_head.named_parameters():
            if "classifier" in name:
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA-injected model: trainable params={trainable}, "
        f"total={total}, ratio={trainable/total:.4f}"
    )
    return model


# ---------------- Dataset / eval helpers ---------------- #

def get_dataloaders(batch_size=2, resize=(384, 384)):
    train_new = YCBDataset(
        TRAIN_SPLIT,
        phase="new",
        resize=resize,
    )
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


def create_lora_model_from_base(ckpt_path: str, r=8, alpha=16):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.config.ignore_index = IGNORE_INDEX
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model = inject_lora_into_segformer(model, r=r, alpha=alpha)
    return model


def compute_miou(pred, target, num_classes, ignore_index, eval_class_ids):
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
            logits = outputs.logits

            logits_up = F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pred = logits_up.argmax(1)

            miou += compute_miou(
                pred.cpu(), labels.cpu(),
                NUM_CLASSES, IGNORE_INDEX,
                eval_class_ids,
            )
            n_batches += 1

    miou = miou / max(n_batches, 1)
    logger.info(f"{desc} mIoU (classes {eval_class_ids}): {miou:.4f}")
    return miou


# ---------------- Training loop ---------------- #

def train_incremental_lora(
    base_ckpt=CHECKPOINT_PATH,
    num_epochs=NUM_EPOCHS,
    lr=LORA_LR,
    batch_size=2,
):
    train_new_loader, val_base_loader, val_new_loader = get_dataloaders(
        batch_size=batch_size
    )
    model = create_lora_model_from_base(base_ckpt, r=LORA_R, alpha=LORA_ALPHA).to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    eval_base_classes = [0] + BASE_CLASS_IDS
    eval_new_classes = [0] + NEW_CLASS_IDS

    # JSON results structure
    results = {
        "config": {
            "base_ckpt": base_ckpt,
            "num_epochs": num_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
            "base_class_ids": BASE_CLASS_IDS,
            "new_class_ids": NEW_CLASS_IDS,
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
        },
        "before": {},
        "epochs": [],
        "after": {},
    }

    # 1) BEFORE LoRA training
    logger.info("==> Evaluation BEFORE custom-LoRA incremental training")
    base_before = evaluate(
        model, val_base_loader, eval_base_classes,
        desc="[Before LoRA] Base-set"
    )
    new_before = evaluate(
        model, val_new_loader, eval_new_classes,
        desc="[Before LoRA] New-set"
    )
    results["before"] = {
        "base_miou": float(base_before),
        "new_miou": float(new_before),
    }

    # 2) Train LoRA + head on NEW classes
    logger.info("\n==> Starting custom-LoRA incremental training on NEW classes...")
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
        logger.info(f"Epoch {epoch}: LoRA train (new classes) loss = {avg_loss:.4f}")

        new_miou_epoch = evaluate(
            model, val_new_loader, eval_new_classes,
            desc=f"[Epoch {epoch} LoRA] New-set"
        )

        results["epochs"].append({
            "epoch": epoch,
            "train_loss_new": float(avg_loss),
            "val_miou_new": float(new_miou_epoch),
        })

    # 3) AFTER LoRA training
    logger.info("\n==> Evaluation AFTER custom-LoRA incremental training")
    base_after = evaluate(
        model, val_base_loader, eval_base_classes,
        desc="[After LoRA] Base-set"
    )
    new_after = evaluate(
        model, val_new_loader, eval_new_classes,
        desc="[After LoRA] New-set"
    )

    results["after"] = {
        "base_miou": float(base_after),
        "new_miou": float(new_after),
    }

    # Save model + metrics
    os.makedirs(os.path.dirname(LORA_CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), LORA_CHECKPOINT_PATH)
    logger.info(f"Saved custom-LoRA incremental model to {LORA_CHECKPOINT_PATH}")

    os.makedirs(os.path.dirname(LORA_RESULTS_PATH), exist_ok=True)
    with open(LORA_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to {LORA_RESULTS_PATH}")

    return base_after, new_after


if __name__ == "__main__":
    train_incremental_lora()
