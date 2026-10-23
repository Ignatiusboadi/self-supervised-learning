from dotenv import load_dotenv
import json
from pathlib import Path
import os
import torch

from data_pipeline.finetuning_data import get_finetuning_loaders
from training.finetuning import load_pretrained, fit_finetuning
from utils.evaluation import evaluate_model

root = "."
OUTPUT_ROOT = os.path.join(root, "outputs/finetuning_outputs")

BATCH_SIZE = 128
NUM_EPOCHS = 2 # 30
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
FREEZE_EPOCHS = 10
SEED = 42

STRATEGIES = ["full", "freeze_then_full", "linear_probe", "partial", ]
DATA_FRACTIONS = [1.0, 0.5]


def run_experiment(strategy, data_fraction, device, model_variant="resnet18", data="cifar10",
                   ssl_type="simclr"):
    DATA_DIR = os.path.join(root, "data", data)
    CHECKPOINT_PATH = os.path.join(root, "outputs/checkpoints", ssl_type, f"simclr_{model_variant}_encoder.pth")
    fraction_name = f"{int(data_fraction * 100)}pct"
    experiment_name = f"{strategy}_{fraction_name}"
    output_dir = Path(OUTPUT_ROOT) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Experiment: {experiment_name}")
    print("=" * 70)

    train_loader, test_loader, classes = get_finetuning_loaders(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE, data_fraction=data_fraction,
        seed=SEED, num_workers=0, )

    model = load_pretrained(checkpoint_path=CHECKPOINT_PATH, resnet_variant="resnet18", num_classes=len(classes), )

    model, history = fit_finetuning(
        model=model,
        train_loader=train_loader,
        device=device,
        output_dir=output_dir,
        strategy=strategy,
        num_epochs=NUM_EPOCHS,
        freeze_epochs=FREEZE_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    metrics = evaluate_model(model=model, test_loader=test_loader, classes=classes, device=device,
        checkpoint_path=output_dir / "best_model.pth", )

    results = {
        "strategy": strategy,
        "data_fraction": data_fraction,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "history": history,
        "metrics": metrics,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved experiment outputs to: {output_dir}")
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = []

    for data_fraction in DATA_FRACTIONS:
        for strategy in STRATEGIES[0:1]:
            result = run_experiment(
                strategy=strategy,
                data_fraction=data_fraction,
                device=device,
            )
            all_results.append(result)

    summary = [
        {
            "strategy": result["strategy"],
            "data_fraction": result["data_fraction"],
            "test_loss": result["metrics"]["test_loss"],
            "test_accuracy": result["metrics"]["test_accuracy"],
        }
        for result in all_results]

    summary_path = Path(OUTPUT_ROOT) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nExperiment summary:")
    for row in summary:
        print(f"{row['strategy']:>18s} | {row['data_fraction']:.0%} data | test accuracy: {row['test_accuracy']:.2f}%")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
