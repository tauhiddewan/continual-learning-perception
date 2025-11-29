import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from dataset import YCBSegformerDataset
from config import NUM_CLASSES, IGNORE_INDEX, BASE_CLASS_IDS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataloaders(batch_size=32, resize=(384, 384)):
    train_ds = YCBSegformerDataset(
        "splits/train_all.txt",
        phase="base",
        resize=resize,
    )
    val_ds = YCBSegformerDataset(
        "splits/val_all.txt",
        phase="base",
        resize=resize,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader

def create_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,          # 22 (0..21)
        ignore_mismatched_sizes=True,    # adjust classifier head
    )
    # make sure model config uses same ignore_index as our labels
    model.config.ignore_index = IGNORE_INDEX
    return model

def compute_miou(pred, target, num_classes, ignore_index):
    # pred: (N, H, W), target: (N, H, W)
    pred = pred.view(-1)
    target = target.view(-1)

    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    ious = []
    eval_classes = [0] + BASE_CLASS_IDS  # background + base classes
    for cls in eval_classes:
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

def train(num_epochs=5, lr=6e-5, batch_size=2):
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    model = create_model().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()


            # outputs = model(pixel_values=imgs)
            # logits = outputs.logits  # (B, C, H, W)

            # loss = criterion(logits, labels)

            outputs = model(pixel_values=imgs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits


            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}: train loss = {avg_loss:.4f}")

        # validation
        model.eval()
        miou = 0.0
        n_batches = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(pixel_values=imgs)
                logits = outputs.logits  # (B, C, h', w')

                # Upsample logits to label size
                logits_up = F.interpolate(
                    logits,
                    size=labels.shape[-2:],  # (H, W) e.g. 384x384
                    mode="bilinear",
                    align_corners=False,
                )
                pred = logits_up.argmax(1)  # (B, H, W)

                miou += compute_miou(
                    pred.cpu(), labels.cpu(),
                    NUM_CLASSES, IGNORE_INDEX
                )
                n_batches += 1

        miou = miou / max(n_batches, 1)
        print(f"Epoch {epoch}: val mIoU (base classes) = {miou:.4f}")

    torch.save(model.state_dict(), "checkpoints/segformer_ycb_base.pth")
    print("Saved base model to checkpoints/segformer_ycb_base.pth")

if __name__ == "__main__":
    train()
