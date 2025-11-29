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


# ---------------- LoRA building blocks ---------------- #

class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.orig = orig_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_dim = orig_linear.in_features
        out_dim = orig_linear.out_features

        # Low-rank matrices
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)

        # Init: make LoRA initially "zero" so model starts as base model
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original weights
        for p in self.orig.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.orig(x) + self.scaling * self.lora_B(self.lora_A(x))


def _get_parent_module(model: nn.Module, module_name: str):
    """
    Given full module name like 'encoder.block.0.layer.0.attn.query',
    return (parent_module, 'query').
    """
    parts = module_name.split(".")
    attr_name = parts[-1]
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, attr_name


def inject_lora_into_segformer(model: nn.Module, r: int = 8, alpha: int = 16):
    """
    Replace Linear layers whose module names contain 'query', 'key',
    'value', or 'proj' with LoRALinear wrappers.
    """
    replace_keywords = ["query", "key", "value", "proj"]

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(k in name for k in replace_keywords):
            parent, attr = _get_parent_module(model, name)
            orig_linear = getattr(parent, attr)
            setattr(parent, attr, LoRALinear(orig_linear, r=r, alpha=alpha))

    # Freeze everything except LoRA layers + decode head classifier
    for name, param in model.named_parameters():
        if isinstance(getattr_module_by_name(model, name), LoRALinear):
            param.requires_grad = True
        elif "decode_head.classifier" in name:
            param.requires_grad = True
        else:
            # For safety, we'll set requires_grad=False here,
            # and rely on LoRALinear to re-enable its A/B params.
            param.requires_grad = False

    # Re-enable LoRA A/B explicitly
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for p in module.lora_A.parameters():
                p.requires_grad = True
            for p in module.lora_B.parameters():
                p.requires_grad = True

    # Print summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA-injected model: trainable params={trainable}, total={total}, ratio={trainable/total:.4f}")

    return model


def getattr_module_by_name(model: nn.Module, full_name: str):
    parts = full_name.split(".")
    m = model
    for p in parts:
        if p.isdigit():
            m = m[int(p)]
        else:
            m = getattr(m, p)
    return m


# ---------------- Dataset / eval helpers ---------------- #

def get_dataloaders(batch_size=2, resize=(384, 384)):
    train_new = YCBSegformerDataset(
        "splits/train_all.txt",
        phase="new",
        resize=resize,
    )
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
            logits = outputs.logits  # (B, C, h', w')

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
    print(f"{desc} mIoU (classes {eval_class_ids}): {miou:.4f}")
    return miou


# ---------------- Training loop ---------------- #

def train_incremental_lora(
    base_ckpt="checkpoints/segformer_ycb_base.pth",
    num_epochs=5,
    lr=1e-4,
    batch_size=2,
):
    train_new_loader, val_base_loader, val_new_loader = get_dataloaders(
        batch_size=batch_size
    )
    model = create_lora_model_from_base(base_ckpt, r=8, alpha=16).to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    eval_base_classes = [0] + BASE_CLASS_IDS
    eval_new_classes = [0] + NEW_CLASS_IDS

    # 1) BEFORE LoRA training
    print("==> Evaluation BEFORE custom-LoRA incremental training")
    evaluate(
        model, val_base_loader, eval_base_classes,
        desc="[Before LoRA] Base-set"
    )
    evaluate(
        model, val_new_loader, eval_new_classes,
        desc="[Before LoRA] New-set"
    )

    # 2) Train only LoRA + head on NEW classes
    print("\n==> Starting custom-LoRA incremental training on NEW classes...")
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
        print(f"Epoch {epoch}: LoRA train (new classes) loss = {avg_loss:.4f}")

        evaluate(
            model, val_new_loader, eval_new_classes,
            desc=f"[Epoch {epoch} LoRA] New-set"
        )

    # 3) AFTER LoRA: check forgetting
    print("\n==> Evaluation AFTER custom-LoRA incremental training")
    base_after = evaluate(
        model, val_base_loader, eval_base_classes,
        desc="[After LoRA] Base-set"
    )
    new_after = evaluate(
        model, val_new_loader, eval_new_classes,
        desc="[After LoRA] New-set"
    )

    torch.save(model.state_dict(), "checkpoints/segformer_ycb_incremental_lora_custom.pth")
    print("Saved custom-LoRA incremental model to checkpoints/segformer_ycb_incremental_lora_custom.pth")

    return base_after, new_after


if __name__ == "__main__":
    train_incremental_lora()
