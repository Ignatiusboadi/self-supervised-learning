
import torch
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm


def evaluate_model(model, test_loader, classes, device, checkpoint_path=None):
    if checkpoint_path is not None:
        state_dict = torch.load(Path(checkpoint_path), map_location=device, weights_only=True, )
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    criterion = nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct = 0
    total = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size

            for label, prediction in zip(labels, predictions):
                label_id = label.item()
                class_total[label_id] += 1
                if label_id == prediction.item():
                    class_correct[label_id] += 1

    metrics = {
        "test_loss": loss_sum / total,
        "test_accuracy": 100.0 * correct / total,
        "correct": correct,
        "total": total,
        "per_class_accuracy": {
            name: (100.0 * class_correct[i] / class_total[i] if class_total[i] else 0.0)
            for i, name in enumerate(classes)}, }

    print(f"Test loss:     {metrics['test_loss']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.2f}%")
    print(f"Correct:       {correct}/{total}")
    print("\nPer-class test accuracy:")

    for name, accuracy in metrics["per_class_accuracy"].items():
        print(f"{name:>12s}: {accuracy:.2f}%")

    return metrics
