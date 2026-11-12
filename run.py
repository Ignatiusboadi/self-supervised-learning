import argparse
import torch

from utils.distributed import setup_runtime, cleanup_runtime


def build_parser():
    parser = argparse.ArgumentParser(description="Self-supervised pretraining and downstream fine-tuning")

    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--mode", choices=["pretrain", "finetune", "csr"], required=True,
        help="Training workflow to execute", )
    parser.add_argument("--ssl-type", default="simclr", choices=["simclr", "simclrv2"])
    parser.add_argument("--resnet-variant",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        default="resnet18", help="Backbone architecture", )
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name", )
    parser.add_argument("--data-path", type=str, default=None,
                        help="Dataset directory; defaults depend on the mode", )
    parser.add_argument("--image-size", type=int, default=None,
                        help="Image size, use dataset default if omitted")
    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default=None,
        help="Root directory for checkpoints, plots, and results", )
    parser.add_argument("--train-cache-path", type=str, default=None,
                        help="path for uint train cache", )
    parser.add_argument("--val-cache-path", type=str, default=None, help="path for uint val cache", )
    parser.add_argument("--test-cache-path", type=str, default=None, help="path for uint test cache", )
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
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Path to the pretrained encoder or model checkpoint", )
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--data-fraction", type=float, nargs="+", default=[1.0],
        help="One or more training fractions, e.g. 1.0 0.5", )
    parser.add_argument("--strategies", nargs="+",
                        choices=["full", "freeze_then_full", "linear_probe", "partial"], default=["full"],
                        help="Run one or multiple fine-tuning strategy(ies)", )
    parser.add_argument("--width-multiplier", type=int, default=1)
    parser.add_argument("--use-sk", action="store_true", )
    parser.add_argument("--small-image-stem", action="store_true", )
    parser.add_argument("--simclrv2-selector", type=int, default=0)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--feature-layer", type=str, default="encoder",
                        choices=["encoder", "projection_1", "projection_2", "projection"])
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--projection-hidden-dim", type=int, default=512)
    parser.add_argument("--projection-layers", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--from-scratch", action="store_true")

    parser.add_argument("--csr-hidden-dim", type=int, default=None)
    parser.add_argument("--csr-topk", type=int, default=32)
    parser.add_argument("--csr-gamma", type=float, default=1.0)
    parser.add_argument("--csr-auxk", type=int, default=512)
    parser.add_argument("--csr-beta", type=float, default=1 / 32)
    parser.add_argument("--csr-epochs", type=int, default=10)
    parser.add_argument("--csr-learning-rate", type=float, default=4e-5)
    parser.add_argument("--classifier-epochs", type=int, default=30)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-3)

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
        runtime.print(f"Backbone: {args.resnet_variant}")
        runtime.print(f"Device: {runtime.device}")
        runtime.print(f"Distributed: {runtime.distributed}, World size: {runtime.world_size}")

        if runtime.device.type == "cuda":
            runtime.print(f"GPU: {torch.cuda.get_device_name(runtime.device)}")

        if args.mode == "pretrain":
            from pretrain import run_pretraining
            run_pretraining(args, runtime)

        elif args.mode == "finetune":
            from finetune import run_finetuning
            run_finetuning(args, runtime)

        else:
            from csr_pipeline import run_csr
            run_csr(args, runtime)

    finally:
        cleanup_runtime(runtime)


if __name__ == "__main__":
    main()
