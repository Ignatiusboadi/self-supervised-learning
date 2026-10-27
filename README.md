# Self-Supervised Learning with SimCLR

A PyTorch research project exploring **self-supervised visual representation learning** using SimCLR and downstream image classification. The project investigates how model architecture, optimization strategies, training-data availability, class imbalance, and label noise affect representation quality and classification performance.

## Features

- **SimCLR pretraining** for learning visual representations from augmented image views.
- **ResNet-18 and ResNet-50** encoder architectures.
- **Linear probing and fine-tuning** of pretrained encoders.
- Multiple fine-tuning strategies: `linear_probe`, `full`, `freeze_then_full`, and `partial`.
- Optimization experiments using SGD, Adam, AdamW, LARS, and Sharpness-Aware Minimization (SAM).
- Experiments with different training-data fractions, class imbalance, and noisy labels.
- Training and validation plots, checkpoints, and evaluation summaries.
- PyTorch Distributed Data Parallel (DDP) workflows for multi-GPU training.

## Datasets

The project focuses on:

- **CIFAR-10**
- **CIFAR-100**

Dataset paths and training configurations can be specified through command-line arguments.

## Methodology

### SimCLR Pretraining

SimCLR learns representations by generating two augmented views of each image and training an encoder to produce similar representations for those views.

The model consists of an **encoder** for extracting image features and a **projection head** for contrastive learning. The pretrained encoder is subsequently used for downstream classification.

### Downstream Evaluation

The project supports several transfer-learning strategies:

- **Linear probing:** Freeze the encoder and train a linear classifier.
- **Full fine-tuning:** Train the encoder and classification head together.
- **Freeze-then-full:** Train the classification head with the encoder frozen, then unfreeze the encoder.
- **Partial fine-tuning:** Update a subset of model parameters.

### Optimization and Data Experiments

Experiments investigate the effects of optimizer choice, batch size, model architecture, training-data availability, class imbalance, and label noise on model performance.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Install dependencies using `requirements.txt` if it is present in the repository.

## Usage

The unified entry point is `run.py`.

View available options:

```bash
python run.py --help
```

### SimCLR Pretraining

```bash
python run.py \
    --mode pretrain \
    --model resnet18 \
    --optimizer lars \
    --batch-size 256 \
    --epochs 100
```

### Fine-Tuning

```bash
python run.py \
    --mode finetune \
    --model resnet18 \
    --checkpoint path/to/checkpoint.pth \
    --strategy full
```

### Linear Probing

```bash
python run.py \
    --mode finetune \
    --model resnet18 \
    --checkpoint path/to/checkpoint.pth \
    --strategy linear_probe
```
### Distributed Training (Multi-GPU)

The project supports distributed training workflows using PyTorch Distributed Data Parallel (DDP). Use `torchrun` to launch one training process per GPU.

**Single machine with multiple GPUs**

SimCLR pretraining:

```bash
torchrun --standalone --nproc_per_node=2 run.py \
    --mode pretrain \
    --model resnet50 \
    --optimizer lars \
    --batch-size 256 \
    --epochs 100
```

Fine-tuning:

```bash
torchrun --standalone --nproc_per_node=2 run.py \
    --mode finetune \
    --model resnet18 \
    --checkpoint path/to/checkpoint.pth \
    --strategy full \
    --epochs 50
```

**Parameters**

- `--nproc_per_node=4`: launches four processes, typically one per GPU.
- `--standalone`: configures distributed rendezvous for a single machine.
- `--batch-size`: check the implementation to determine whether this is the per-process batch size or the global batch size. In standard DDP data loading, it is usually the per-process batch size, making the effective batch size approximately `batch size × number of GPUs`.

Change the number of processes to match the GPUs available, for example, `--nproc_per_node=2` for two GPUs.

These commands illustrate the CLI; adjust dataset paths and training parameters to match your configuration.

## Evaluation

Depending on the experiment, evaluation includes:

- Accuracy and cross-entropy loss
- Balanced accuracy
- Macro-F1 and weighted F1
- Per-class precision and recall
- Confusion matrices

Validation data is used for model selection, while the held-out test set is reserved for final evaluation.

## Current Scope

The project focuses on SimCLR pretraining, downstream transfer learning, optimization comparisons, and experiments involving data availability and data quality.

**Future work** includes a closer SimCLR v2 reproduction and more systematic comparisons of model capacity, batch size, and representation quality.

## Reference

Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. (2020). [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709). *ICML 2020*.
