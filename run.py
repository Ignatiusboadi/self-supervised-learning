import argparse
import torch

from utils.distributed import setup_runtime, cleanup_runtime


def build_parser():
    parser = argparse.ArgumentParser(description="Self-supervised pretraining and downstream fine-tuning")

    parser.add_argument("--mode", choices=["pretrain", "finetune"], required=True,
        help="Training workflow to execute", )
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18",
        help="Backbone architecture", )
    parser.add_argument("--data", default="cifar10", help="Dataset name", )
    parser.add_argument("--data-path", type=str, default=None,
                        help="Dataset directory; defaults depend on the mode", )
    parser.add_argument("--output-dir", type=str, default=None,
        help="Root directory for checkpoints, plots, and results", )
    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto",)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw", "lars", "sam"], default=None, )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=float, default=0.2)
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Path to the pretrained encoder or model checkpoint", )
    parser.add_argument("--strategy", choices=["full", "freeze_then_full", "linear_probe", "partial"],
        default="full", )
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--data-fraction", type=float, nargs="+", default=[1.0],
        help="One or more training fractions, e.g. 1.0 0.5", )
    parser.add_argument("--strategies", nargs="+",
                        choices=["full", "freeze_then_full", "linear_probe", "partial"], default=None,
                        help="Run multiple fine-tuning strategies", )
    parser.add_argument("--ssl-type", default="simclr")

    return parser


def main():
    args = build_parser().parse_args()
    runtime = setup_runtime(args.device)

    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")

    if any(not 0 < fraction <= 1 for fraction in args.data_fraction):
        raise ValueError("--data-fraction values must be in (0, 1]")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    try:
        runtime.print(f"Mode: {args.mode}")
        runtime.print(f"Backbone: {args.model}")
        runtime.print(f"Device: {runtime.device}")
        runtime.print(f"Distributed: {runtime.distributed}, World size: {runtime.world_size}")

        if runtime.device.type == "cuda":
            runtime.print(f"GPU: {torch.cuda.get_device_name(runtime.device)}")

        if args.mode == "pretrain":
            from pretrain import run_pretraining
            run_pretraining(args, runtime)
        else:
            from finetune import run_finetuning
            run_finetuning(args, runtime)

    finally:
        cleanup_runtime(runtime)


if __name__ == "__main__":
    main()
