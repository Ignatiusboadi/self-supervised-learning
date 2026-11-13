# Self-Supervised Visual Representation Learning

A PyTorch research codebase implementing **SimCLR v1 and SimCLR v2** for self-supervised visual representation learning and downstream image classification. The project also investigates the effects of training-data availability, class imbalance, and label noise on learned representations. A sparse-feature learning pipeline is also explored.

## Overview

The repository implements a configurable SimCLR pipeline. The model variant is determined by the encoder and projection-head configuration:

- **SimCLR v1:** Standard ResNet encoder and a 2-layer projection head.
- **SimCLR v2:** ResNet encoder with Selective Kernel (SK) blocks, wider widths, and a 3-layer projection head.

These variants share the same training pipeline. The architecture is configured using `--use-sk` and `--projection-layers`.

The implementation supports ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152, alongside SGD, Adam, AdamW, LARS, and SAM optimizers.

## Repository Structure

```text
.
├── data_pipeline/   # Datasets, augmentations, and data loaders
├── evaluation/      # Classification metrics and evaluation
├── models/          # ResNet, SimCLR, classifier, and CSR models
├── training/        # Pretraining and fine-tuning loops
├── utils/           # Losses, optimizers, schedulers, and utilities
├── pretrain.py      # SimCLR pretraining
├── finetune.py      # Downstream classification
├── csr_pipeline.py  # Sparse-feature learning
└── run.py           # Main command-line interface
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows, activate the environment using:

```powershell
.\.venv\Scripts\Activate.ps1
```

The supplied requirements specify CUDA-enabled PyTorch packages. For CPU or Apple Silicon, install a compatible PyTorch/torchvision version using the [official installation guide](https://pytorch.org/get-started/locally/).

## Dataset Format

The data loaders use `torchvision.datasets.ImageFolder`. Arrange images into class-specific directories:

```text
data/cifar10/
├── train/
│   ├── class_1/
│   └── class_2/
├── val/
│   ├── class_1/
│   └── class_2/
└── test/
    ├── class_1/
    └── class_2/
```

The validation directory is optional; when absent, a validation split is created from the training data.

For SimCLR pretraining, `--data-path` should point directly to the directory containing the class folders (e.g. `data/cifar10/train`). Class labels are not used in the contrastive learning objective.

## Running Experiments

Use the main entry point to inspect available arguments:

```bash
python run.py --help
```

### SimCLR v1 Pretraining

```bash
python run.py \
    --mode pretrain \
    --experiment-name simclr_v1_resnet18_pretraining \
    --data-path ./data/cifar10/train \
    --resnet-variant resnet18 \
    --projection-layers 2 \
    --optimizer lars \
    --batch-size 1024 \
    --epochs 100
```

### SimCLR v2 Pretraining

```bash
python run.py \
    --mode pretrain \
    --experiment-name simclr_v2_resnet18_pretraining \
    --data-path ./data/cifar10/train \
    --resnet-variant resnet18 \
    --use-sk \
    --width-multiplier 2 \
    --projection-layers 3 \
    --optimizer lars \
    --batch-size 256 \
    --epochs 100
```

The `--use-sk` flag enables Selective Kernel blocks, while `--projection-layers` determines the projection-head depth. Model configurations are stored in checkpoints and are used to re-create model for downstream experiments.

### Downstream Fine-tuning

```bash
python run.py \
    --mode finetune \
    --experiment-name cifar10_finetune \
    --dataset cifar10 \
    --data-path ./data/cifar10 \
    --checkpoint ./outputs/checkpoints/simclr/simclr_v1_resnet18_pretraining_checkpoint.pth \
    --strategies linear_probe full \
    --epochs 100
```

Available strategies:

- `linear_probe`: Freeze the encoder and train the classification head.
- `full`: Train the encoder and classification head.
- `freeze_then_full`: Initially freeze the encoder, then unfreeze it.
- `partial`: Train the final ResNet stage and classification head.

### Sparse Feature Learning (CSR)

```bash
python run.py \
    --mode csr \
    --experiment-name cifar10_csr \
    --dataset cifar10 \
    --data-path ./data/cifar10 \
    --checkpoint ./outputs/checkpoints/simclr/simclr_v1_resnet18_pretraining_checkpoint.pth
```

The CSR pipeline extracts pretrained features, trains a Top-K sparse autoencoder, and uses the resulting representations for downstream classification. CSR does not currently support distributed training.

### Multi-GPU Training

Pretraining and fine-tuning support CUDA Distributed Data Parallel (DDP). For example, to use two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 run.py \
    --mode pretrain \
    --experiment-name simclrv1_ddp \
    --data-path ./data/cifar10/train \
    --device cuda \
    --batch-size 512 \
    --epochs 100
```

## Outputs

Checkpoints, training plots, and experiment summaries are saved under `outputs/` by default. Use `--output-dir` to specify another location.

## Current Scope

The project focuses on SimCLR pretraining, downstream transfer learning, optimization comparisons, and experiments involving data availability and data quality.

**Future work** includes MoCo and SimSiam reproduction with assessment on MRI and CT scans.

## References

- Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. (2020). [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709). *ICML 2020*.

- Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2020). [Sharpness-aware minimization for efficiently improving generalization](https://arxiv.org/abs/2010.01412). *arXiv preprint arXiv:2010.01412*.

- You, Y., Gitman, I., & Ginsburg, B. (2017). [Large batch training of convolutional networks](https://arxiv.org/abs/1708.03888). *arXiv preprint arXiv:1708.03888*.

- Chen, T., Kornblith, S., Swersky, K., Norouzi, M., & Hinton, G. E. (2020). [Big self-supervised models are strong semi-supervised learners](https://arxiv.org/abs/2006.10029). *Advances in Neural Information Processing Systems*, *33*, 22243–22255.

- Wen, T., Wang, Y., Zeng, Z., Peng, Z., Su, Y., Liu, X., Chen, B., Liu, H., Jegelka, S., & You, C. (2025). [Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation](https://arxiv.org/abs/2503.01776). *Proceedings of the 42nd International Conference on Machine Learning*, 267, 66520–66538.
