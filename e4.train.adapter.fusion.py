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

logger = get_logger(FUSION_LOG_FILE_PATH)


# ---------- LoRA building blocks ---------- #

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


def _get_parent_module(model: nn.Module, module_name: str):
    parts = module_name.split(".")
    attr_name = parts[-1]
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, attr_name


def inject_lora_into_segformer(model: nn.Module, r: int = 8, alpha: int = 16):
    """
    Replace Linear layers with names containing 'query', 'key', 'value', or 'proj'
    by LoRALinear wrappers.
    """
    keywords = ["query", "key", "value", "proj"]

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(k in name for k in keywords):
            parent, attr = _get_parent_module(model, name)
            orig_linear = getattr(parent, attr)
            setattr(parent, attr, LoRALinear(orig_linear, r=r, alpha=alpha))

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze LoRA params + classifier head
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
        f"[inject_lora_into_segformer] trainable={trainable}, "
        f"total={total}, ratio={trainable/total:.4f}"
    )
    return model


# --------------------- Data & model helpers ---------------------- #

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


def load_base_model(ckpt_path=CHECKPOINT_PATH):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.config.ignore_index = IGNORE_INDEX
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(DEVICE)


def create_adapter_model_from_base(ckpt_path=CHECKPOINT_PATH, r=8, alpha=16):
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


# ---------------------------- Metrics ------------------------------------- #

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
        inter = (pred_i & target_i).sum().item()
        union = (pred_i | target_i).sum().item()
        if union == 0:
            continue
        ious.append(inter / (union + 1e-6))
    if not ious:
        return 0.0
    return sum(ious) / len(ious)


def evaluate_base_only(model, loader, eval_class_ids, desc=""):
    model.eval()
    miou = 0.0
    n_batches = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model(pixel_values=imgs)
            logits = out.logits
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
    miou /= max(n_batches, 1)
    logger.info(f"{desc} mIoU (classes {eval_class_ids}): {miou:.4f}")
    return miou


def evaluate_fused(base_model, adapter_model, loader, eval_class_ids, desc=""):
    """
    Fusion: base handles base classes, adapter handles new classes.
    """
    base_model.eval()
    adapter_model.eval()
    miou = 0.0
    n_batches = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            out_base = base_model(pixel_values=imgs)
            logits_base = out_base.logits
            logits_base_up = F.interpolate(
                logits_base,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            out_adapt = adapter_model(pixel_values=imgs)
            logits_adapt = out_adapt.logits
            logits_adapt_up = F.interpolate(
                logits_adapt,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            fused = logits_base_up.clone()
            for cid in NEW_CLASS_IDS:
                fused[:, cid:cid+1, :, :] = logits_adapt_up[:, cid:cid+1, :, :]

            pred = fused.argmax(1)

            miou += compute_miou(
                pred.cpu(), labels.cpu(),
                NUM_CLASSES, IGNORE_INDEX,
                eval_class_ids,
            )
            n_batches += 1

    miou /= max(n_batches, 1)
    logger.info(f"{desc} mIoU (classes {eval_class_ids}): {miou:.4f}")
    return miou


# ------------------------------- Training ---------------------------------- #

def train_adapter_fusion(
    base_ckpt=CHECKPOINT_PATH,
    num_epochs=NUM_EPOCHS,
    lr=FUSION_LR,
    batch_size=2,
):
    train_new_loader, val_base_loader, val_new_loader = get_dataloaders(
        batch_size=batch_size
    )

    base_model = load_base_model(base_ckpt)
    adapter_model = create_adapter_model_from_base(base_ckpt, r=LORA_R, alpha=LORA_ALPHA).to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, adapter_model.parameters()),
        lr=lr,
    )
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

    # 1) BEFORE training: base-only evaluation
    logger.info("==> BEFORE training adapter")
    base_base_miou = evaluate_base_only(base_model, val_base_loader, eval_base_classes,
                                        desc="[Base-only] Base-set")
    base_new_miou = evaluate_base_only(base_model, val_new_loader, eval_new_classes,
                                       desc="[Base-only] New-set")

    results["before"] = {
        "base_only_base_miou": float(base_base_miou),
        "base_only_new_miou": float(base_new_miou),
    }

    # 2) Train adapter on NEW classes only
    logger.info("\n==> Training adapter (LoRA) on NEW classes...")
    for epoch in range(1, num_epochs + 1):
        adapter_model.train()
        total_loss = 0.0

        for imgs, labels in train_new_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            out = adapter_model(pixel_values=imgs)
            logits = out.logits
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
        logger.info(f"Epoch {epoch}: adapter train loss (new classes) = {avg_loss:.4f}")

        fused_new_miou = evaluate_fused(
            base_model, adapter_model, val_new_loader,
            eval_new_classes,
            desc=f"[Epoch {epoch}] Fused New-set",
        )

        results["epochs"].append({
            "epoch": epoch,
            "train_loss_new": float(avg_loss),
            "fused_new_miou": float(fused_new_miou),
        })

    # 3) AFTER training: fused evaluation
    logger.info("\n==> AFTER training adapter (fusion)")
    fused_base_miou = evaluate_fused(
        base_model, adapter_model, val_base_loader,
        eval_base_classes,
        desc="[Fused] Base-set",
    )
    fused_new_miou = evaluate_fused(
        base_model, adapter_model, val_new_loader,
        eval_new_classes,
        desc="[Fused] New-set",
    )

    results["after"] = {
        "fused_base_miou": float(fused_base_miou),
        "fused_new_miou": float(fused_new_miou),
    }

    os.makedirs(os.path.dirname(FUSION_CHECKPOINT_PATH), exist_ok=True)
    torch.save(adapter_model.state_dict(), FUSION_CHECKPOINT_PATH)
    logger.info(f"Saved adapter model to {FUSION_CHECKPOINT_PATH}")

    os.makedirs(os.path.dirname(FUSION_RESULTS_PATH), exist_ok=True)
    with open(FUSION_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to {FUSION_RESULTS_PATH}")


if __name__ == "__main__":
    train_adapter_fusion()
