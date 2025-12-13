# train.incremental.lora_kd.py

import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation

from utils.dataset import YCBDataset
from utils.config import *
from utils.logger import get_logger

logger = get_logger(KD_LORA_LOG_FILE_PATH)


# ---------- LoRA blocks (same style as before) ---------- #

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


def inject_lora(model: nn.Module, r: int = 8, alpha: int = 16):
    keywords = ["query", "key", "value", "proj"]

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(k in name for k in keywords):
            parent, attr = _get_parent_module(model, name)
            orig_linear = getattr(parent, attr)
            setattr(parent, attr, LoRALinear(orig_linear, r=r, alpha=alpha))

    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze LoRA params + classifier head
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora_A.parameters():
                p.requires_grad = True
            for p in m.lora_B.parameters():
                p.requires_grad = True

    if hasattr(model, "decode_head"):
        for name, p in model.decode_head.named_parameters():
            if "classifier" in name:
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"[inject_lora] trainable={trainable}, total={total}, ratio={trainable/total:.4f}")
    return model


# --------------------- Data & models --------------------- #

def get_dataloaders(batch_size=BATCH_SIZE, resize=RESIZE):
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


def load_teacher(base_ckpt=CHECKPOINT_PATH):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.config.ignore_index = IGNORE_INDEX
    state_dict = torch.load(base_ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(DEVICE)


def create_student_lora(base_ckpt=CHECKPOINT_PATH, r=LORA_R, alpha=LORA_ALPHA):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.config.ignore_index = IGNORE_INDEX
    state_dict = torch.load(base_ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    model = inject_lora(model, r=r, alpha=alpha)
    return model


# --------------------- Metrics --------------------- #

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


def evaluate(model, loader, eval_class_ids, desc=""):
    model.eval()
    miou = 0.0
    n_batches = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outs = model(pixel_values=imgs)
            logits = outs.logits
            logits_up = F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pred = logits_up.argmax(1)

            miou += compute_miou(
                pred.cpu(), labels.cpu(),
                NUM_CLASSES, IGNORE_INDEX, eval_class_ids
            )
            n_batches += 1
    miou /= max(n_batches, 1)
    logger.info(f"{desc} mIoU (classes {eval_class_ids}): {miou:.4f}")
    return miou


# --------------------- Training with KD --------------------- #

def train_incremental_lora_kd(
    base_ckpt=CHECKPOINT_PATH,
    num_epochs=NUM_EPOCHS,
    lr=KD_LORA_LR,
    batch_size=BATCH_SIZE,
    lambda_kd=KD_LORA_LAMBDA,
    T=KD_LORA_T,
):
    train_new_loader, val_base_loader, val_new_loader = get_dataloaders(
        batch_size=batch_size
    )

    teacher = load_teacher(base_ckpt)
    student = create_student_lora(base_ckpt).to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr,
    )
    ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

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
            "lambda_kd": lambda_kd,
            "T": T,
        },
        "before": {},
        "epochs": [],
        "after": {},
    }

    logger.info("==> BEFORE KD-LoRA training")
    teacher_base_miou = evaluate(teacher, val_base_loader, eval_base_classes,
                                 desc="[Teacher] Base-set")
    teacher_new_miou = evaluate(teacher, val_new_loader, eval_new_classes,
                                desc="[Teacher] New-set")

    results["before"] = {
        "teacher_base_miou": float(teacher_base_miou),
        "teacher_new_miou": float(teacher_new_miou),
    }

    for epoch in range(1, num_epochs + 1):
        student.train()
        total_loss = 0.0

        for imgs, labels in train_new_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Teacher logits (no grad)
            with torch.no_grad():
                t_out = teacher(pixel_values=imgs)
                t_logits = t_out.logits
                t_logits_up = F.interpolate(
                    t_logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # Student logits
            s_out = student(pixel_values=imgs)
            s_logits = s_out.logits
            s_logits_up = F.interpolate(
                s_logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            # CE on new classes (labels already phase="new")
            loss_new = ce_loss(s_logits_up, labels)

            # Distillation on base classes [0] + BASE_CLASS_IDS
            base_ids = [0] + BASE_CLASS_IDS
            base_ids_t = torch.tensor(base_ids, device=DEVICE)

            t_base = t_logits_up.index_select(1, base_ids_t)  # (B,|B|,H,W)
            s_base = s_logits_up.index_select(1, base_ids_t)

            # KD over channels, averaged over batch+space
            t_soft = F.softmax(t_base / T, dim=1)
            s_log_soft = F.log_softmax(s_base / T, dim=1)

            loss_kd = F.kl_div(
                s_log_soft, t_soft,
                reduction="mean"   # keep this small-ish
            ) * (T * T)

            loss = loss_new + lambda_kd * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(train_new_loader.dataset)
        logger.info(f"Epoch {epoch}: train loss (new + KD) = {avg_loss:.4f}")

        student_new_miou = evaluate(
            student, val_new_loader, eval_new_classes,
            desc=f"[Epoch {epoch}] Student New-set"
        )

        results["epochs"].append({
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "student_new_miou": float(student_new_miou),
        })

    logger.info("\n==> AFTER KD-LoRA training (student only)")
    student_base_miou = evaluate(student, val_base_loader, eval_base_classes,
                                 desc="[Student] Base-set")
    student_new_miou = evaluate(student, val_new_loader, eval_new_classes,
                                desc="[Student] New-set")

    results["after"] = {
        "student_base_miou": float(student_base_miou),
        "student_new_miou": float(student_new_miou),
    }

    os.makedirs(os.path.dirname(KD_LORA_CHECKPOINT_PATH), exist_ok=True)
    torch.save(student.state_dict(), KD_LORA_CHECKPOINT_PATH)
    logger.info(f"Saved KD-LoRA incremental model to {KD_LORA_CHECKPOINT_PATH}")

    os.makedirs(os.path.dirname(KD_LORA_RESULTS_PATH), exist_ok=True)
    with open(KD_LORA_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to {KD_LORA_RESULTS_PATH}")


if __name__ == "__main__":
    train_incremental_lora_kd()
