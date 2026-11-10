import json
import torch.distributed as dist
from pathlib import Path

from data_pipeline.finetuning_data import get_finetuning_loaders
from data_pipeline.gpu_finetune_data import get_finetuning_loaders_gpu
from models.csr import train_csr, train_classifier, CSRClassifier, load_simclr_encoder
from evaluation.evaluation import evaluate_model


def run_csr(args, runtime):
    device = runtime.device
    root = Path(args.root_dir)

    base_output_dir = Path(args.output_dir or root / "outputs")
    plot_dir = base_output_dir / "plots"
    output_root = base_output_dir / "csr_outputs"

    plot_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_path or root / "data" / args.dataset)

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for CSR.")

    checkpoint_path = Path(args.checkpoint)

    batch_size = args.batch_size if args.batch_size is not None else 128
    csr_epochs = args.csr_epochs
    classifier_epochs = args.classifier_epochs

    if runtime.distributed:
        raise NotImplementedError(
            "Distributed CSR training is not implemented."
        )

    encoder, feature_dim = load_simclr_encoder(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    if args.train_cache_path is None or args.test_cache_path is None:
        train_loader, val_loader, test_loader, classes = get_finetuning_loaders(
            data_dir=str(data_dir),
            img_size=args.image_size,
            batch_size=batch_size,
            data_fraction=args.data_fraction[0],
            val_fraction=0.2,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    else:
        train_loader, val_loader, test_loader, classes = get_finetuning_loaders_gpu(
            data_dir,
            device=device,
            batch_size=batch_size,
            data_fraction=args.data_fraction[0],
            img_size=args.image_size,
            seed=args.seed,
            train_cache_path=args.train_cache_path,
            test_cache_path=args.test_cache_path,
            val_fraction=0.2,
            val_cache_path=args.val_cache_path,
            train_augment=False,
        )

    sae = train_csr(
        backbone=encoder,
        train_loader=train_loader,
        d=feature_dim,
        h=args.csr_hidden_dim or 4 * feature_dim,
        k=args.csr_topk,
        gamma=args.csr_gamma,
        epochs=csr_epochs,
        lr=args.csr_learning_rate,
        k_aux=args.csr_auxk,
        beta=args.csr_beta,
        device=device,
    )

    classifier = train_classifier(
        backbone=encoder,
        sae=sae,
        clf_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(classes),
        k=args.csr_topk,
        epochs=classifier_epochs,
        lr=args.classifier_learning_rate,
        device=device,
    )

    model = CSRClassifier(
        backbone=encoder,
        sae=sae,
        classifier=classifier,
        k=args.csr_topk,
    ).to(device)

    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        classes=classes,
        device=device,
    )

    output_dir = output_root / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "mode": "csr",
        "resnet_variant": args.resnet_variant,
        "dataset": args.dataset,
        "checkpoint": str(checkpoint_path),
        "data_fraction": args.data_fraction[0],
        "csr": {
            "hidden_dim": args.csr_hidden_dim or 4 * feature_dim,
            "topk": args.csr_topk,
            "gamma": args.csr_gamma,
            "auxk": args.csr_auxk,
            "beta": args.csr_beta,
            "epochs": csr_epochs,
            "learning_rate": args.csr_learning_rate,
        },
        "classifier": {
            "epochs": classifier_epochs,
            "learning_rate": args.classifier_learning_rate,
        },
        "metrics": metrics,
    }

    is_main_process = (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_rank() == 0
    )

    if is_main_process:
        results_path = output_dir / f"{args.experiment_name}_results.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        summary_path = output_root / f"{args.experiment_name}_summary.json"

        with open(summary_path, "w") as f:
            json.dump([{
                "resnet_variant": args.resnet_variant,
                "data_fraction": args.data_fraction[0],
                "test_loss": metrics["test_loss"],
                "test_accuracy": metrics["test_accuracy"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "confusion_matrix": metrics["confusion_matrix"],
                "metrics": metrics,
            }], f, indent=2)

        print("\nCSR test results:")
        print(f"Accuracy:          {metrics['test_accuracy']:.2f}%")
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.2f}%")
        print(f"Macro F1:          {metrics['macro_f1']:.2f}%")
        print(f"Results saved to: {results_path}")

    return results
