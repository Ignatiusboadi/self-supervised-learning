import json
from pathlib import Path

from data_pipeline.finetuning_data import get_finetuning_loaders
from training.finetuning import load_pretrained, fit_finetuning
from utils.distributed import Runtime
from utils.evaluation import evaluate_model
from utils.plotting import plot_fig
from utils.gpu_finetune_data import get_finetuning_loaders_gpu


def run_finetuning(args, runtime: Runtime):
    device = runtime.device
    root = Path(args.root_dir)
    base_output_dir = Path(args.output_dir or root / "outputs")
    plot_dir = base_output_dir / "plots"
    output_root = base_output_dir / "finetuning_outputs"

    plot_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_path or root / "data" / args.data)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = (
            root / "outputs" / "checkpoints" / args.ssl_type
            / f"simclr_{args.model}_encoder.pth"
        )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}"
        )

    num_epochs = args.epochs if args.epochs is not None else 30
    batch_size = args.batch_size if args.batch_size is not None else 128
    learning_rate = args.learning_rate if args.learning_rate is not None else 0.001
    weight_decay = args.weight_decay if args.weight_decay is not None else 5e-4
    optimizer_name = args.optimizer or "adamw"
    strategies = args.strategies or [args.strategy]

    all_results = []

    for strategy in strategies:
        for fraction in args.data_fraction:
            fraction_name = f"{int(fraction * 100)}pct"
            experiment_name = f"{args.model}_{strategy}_{fraction_name}"
            output_dir = output_root / experiment_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 70)
            print(f"Experiment: {experiment_name}")
            print(f"Checkpoint: {checkpoint_path}")
            print("=" * 70)

            if args.train_cache_path is None or args.test_cache_path is None:

                train_loader, test_loader, classes = get_finetuning_loaders(
                    data_dir=str(data_dir),
                    batch_size=batch_size,
                    data_fraction=fraction,
                    seed=args.seed,
                    num_workers=args.num_workers,
                )
            else:
                train_loader, test_loader, classes = get_finetuning_loaders_gpu(
                    data_dir, device=device, batch_size=batch_size, data_fraction=fraction,
                    seed=42, train_cache_path=args.train_cache_path, test_cache_path=args.test_cache_path,
                )

            model = load_pretrained(
                checkpoint_path=str(checkpoint_path),
                resnet_variant=args.model,
                num_classes=len(classes),
            )

            model, history = fit_finetuning(
                model=model,
                train_loader=train_loader,
                device=device,
                output_dir=output_dir,
                strategy=strategy,
                num_epochs=num_epochs,
                freeze_epochs=args.freeze_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                opt_name=optimizer_name,
            )

            plot_fig(history["train_loss"], "Epoch", "Cross-Entropy Loss",
                     "Fine-Tuning Training Loss",
                     str(plot_dir / f"{experiment_name}_train_loss.png"))

            plot_fig(history["train_accuracy"], "Epoch", "Training Accuracy (%)",
                     "Fine-Tuning Training Accuracy",
                     str(plot_dir / f"{experiment_name}_train_accuracy.png"))

            metrics = evaluate_model(
                model=model, test_loader=test_loader, classes=classes, device=device,
                checkpoint_path=output_dir / f"best_model.pth",
            )

            results = {
                "mode": "finetune",
                "model": args.model,
                "dataset": args.data,
                "checkpoint": str(checkpoint_path),
                "strategy": strategy,
                "data_fraction": fraction,
                "optimizer": optimizer_name,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "history": history,
                "metrics": metrics,
            }

            with open(output_dir / f"{experiment_name}_results.json", "w") as f:
                json.dump(results, f, indent=2)

            all_results.append(results)
            print(f"Saved results to: {output_dir}")

    summary = [
        {
            "model": result["model"],
            "strategy": result["strategy"],
            "data_fraction": result["data_fraction"],
            "test_loss": result["metrics"]["test_loss"],
            "test_accuracy": result["metrics"]["test_accuracy"],
        }
        for result in all_results
    ]

    summary_path = output_root / f"{experiment_name}_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nExperiment summary:")
    for row in summary:
        print(f"{row['model']:>10s} | {row['strategy']:>16s} | "
              f"{row['data_fraction']:.0%} data | "
              f"Accuracy: {row['test_accuracy']:.2f}%")

    print(f"Summary saved to: {summary_path}")
    return all_results
