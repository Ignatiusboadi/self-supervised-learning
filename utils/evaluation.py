import torch
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm


def metrics_from_confusion(cm):
    """cm[true, pred] -> per-class precision/recall/F1 and aggregate scores."""
    cm = cm.double()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    support = cm.sum(1)
    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    present = support > 0                         # classes that occur in the labels
    return {
        "precision": precision, "recall": recall, "f1": f1, "support": support,
        "macro_f1": f1[present].mean().item(),
        "weighted_f1": ((f1 * support).sum() / support.sum()).item(),
        "balanced_accuracy": recall[present].mean().item(),
    }


def evaluate_model(model, loader, classes, device, checkpoint_path=None,
                   prefix="test", verbose=True):
    if checkpoint_path is not None:
        state_dict = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    C = len(classes)

    loss_sum, total = 0.0, 0
    cm = torch.zeros(C * C, dtype=torch.long, device=device)

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=f"Evaluating ({prefix})", leave=False, disable=not verbose):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            total += labels.size(0)
            cm += torch.bincount(labels * C + logits.argmax(dim=1), minlength=C * C)

    cm = cm.view(C, C).cpu()
    m = metrics_from_confusion(cm)
    correct = int(cm.diag().sum())

    metrics = {
        f"{prefix}_loss": loss_sum / total,
        f"{prefix}_accuracy": 100.0 * correct / total,
        "macro_f1": 100.0 * m["macro_f1"],
        "weighted_f1": 100.0 * m["weighted_f1"],
        "balanced_accuracy": 100.0 * m["balanced_accuracy"],
        "correct": correct,
        "total": total,
        "confusion_matrix": cm,
        "per_class": {
            name: {"precision": 100.0 * m["precision"][i].item(),
                   "recall": 100.0 * m["recall"][i].item(),
                   "f1": 100.0 * m["f1"][i].item(),
                   "support": int(m["support"][i].item())}
            for i, name in enumerate(classes)},
    }
    # kept for backwards compatibility: per-class accuracy == per-class recall
    metrics["per_class_accuracy"] = {n: v["recall"] for n, v in metrics["per_class"].items()}

    if verbose:
        print(f"{prefix.capitalize()} loss:        {metrics[f'{prefix}_loss']:.4f}")
        print(f"{prefix.capitalize()} accuracy:    {metrics[f'{prefix}_accuracy']:.2f}%")
        print(f"Macro F1:           {metrics['macro_f1']:.2f}%")
        print(f"Weighted F1:        {metrics['weighted_f1']:.2f}%")
        print(f"Balanced accuracy:  {metrics['balanced_accuracy']:.2f}%")
        print(f"\n{'class':>12s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'n':>7s}")
        for name, v in metrics["per_class"].items():
            print(f"{name:>12s} {v['precision']:7.2f} {v['recall']:7.2f} {v['f1']:7.2f} {v['support']:7d}")

    return metrics