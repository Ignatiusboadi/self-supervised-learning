import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients, visualization as viz
from training.finetuning import load_pretrained
from data_pipeline.finetuning_data import get_finetuning_loaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Integrated Gradients visualizations for a fine-tuned SimCLR model."
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
        help="Root directory of the dataset containing train/val/test folders.")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True,
        help="Path to the pretrained SimCLR checkpoint.")
    parser.add_argument("--finetuned_checkpoint", type=str, required=True,
        help="Path to the fine-tuned classification checkpoint.")
    parser.add_argument("--output_dir", type=str,
        default="./outputs/integrated_gradients", help="Directory for saving attribution visualizations.")
    parser.add_argument("--num_images", type=int, default=5, help="Number of test images to explain.")
    parser.add_argument("--n_steps", type=int, default=50, help="Number of integration steps.")
    parser.add_argument("--internal_batch_size", type=int, default=1,
        help="Number of interpolation steps processed together.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=["true", "predicted"], default="predicted",
        help="Class whose output logit will be explained.")
    parser.add_argument("--mean", type=float, nargs=3, default=None,
                        help="Normalization mean for RGB channels.")
    parser.add_argument("--std", type=float, nargs=3, default=None,
                        help="Normalization standard deviation for RGB channels.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denormalize(image, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    if mean is None or std is None:
        return image.clamp(0, 1)
    mean = torch.tensor(mean, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=image.device).view(1, 3, 1, 1)
    return (image * std + mean).clamp(0, 1)


def select_images(dataset, num_images, seed):
    rng = random.Random(seed)
    targets = dataset.targets
    classes = dataset.classes
    class_indices = {
        class_idx: [i for i, label in enumerate(targets) if label == class_idx]
        for class_idx in range(len(classes))
    }
    available_classes = [class_idx for class_idx, indices in class_indices.items() if indices]
    if not available_classes:
        raise ValueError("No images found in the test dataset.")
    selected = []
    selected_classes = rng.sample(available_classes, min(num_images, len(available_classes)))
    for class_idx in selected_classes:
        selected.append(rng.choice(class_indices[class_idx]))
    remaining = list(set(range(len(dataset))) - set(selected))
    additional = min(num_images - len(selected), len(remaining))
    if additional > 0:
        selected.extend(rng.sample(remaining, additional))
    rng.shuffle(selected)
    return selected


def load_model(args, num_classes, device):
    model, config = load_pretrained(checkpoint_path=args.pretrained_checkpoint, num_classes=num_classes)
    checkpoint = torch.load(args.finetuned_checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def generate_attributions(model, dataset, args, device):
    ig = IntegratedGradients(model)
    indices = select_images(dataset, args.num_images, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Selected classes:", [dataset.classes[dataset.targets[i]] for i in indices])
    for count, idx in enumerate(indices, start=1):
        input_tensor, true_label = dataset[idx]
        input_tensor = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            predicted_class = logits.argmax(dim=1).item()
        if args.target == "true":
            target_class = int(true_label)
        else:
            target_class = predicted_class
        attributions, delta = ig.attribute(input_tensor, baselines=torch.zeros_like(input_tensor),
            target=target_class, n_steps=args.n_steps, internal_batch_size=args.internal_batch_size,
            return_convergence_delta=True,)
        image = denormalize(input_tensor, args.mean, args.std)
        img_np = np.transpose(image.squeeze(0).detach().cpu().numpy(),(1, 2, 0))
        attr_np = np.transpose(attributions.squeeze(0).detach().cpu().numpy(),(1, 2, 0))
        fig, axes = viz.visualize_image_attr_multiple(
            attr_np, img_np, methods=["original_image", "heat_map", "blended_heat_map"],
            signs=["all", "positive", "positive"], show_colorbar=True, cmap="Reds", )

        true_name = dataset.classes[true_label]
        predicted_name = dataset.classes[predicted_class]
        target_name = dataset.classes[target_class]
        fig.suptitle(f"True: {true_name} | Predicted: {predicted_name} | Explained: {target_name}")
        filename = f"{count:03d}_true-{true_label}_pred-{predicted_class}_target-{target_class}.png"
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[{count}/{len(indices)}] True: {true_name} | Predicted: {predicted_name} | "
            f"Target: {target_name} | Convergence delta: {delta.item():.6f}")
    print(f"\nVisualizations saved to: {output_dir.resolve()}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not Path(args.pretrained_checkpoint).is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_checkpoint}")
    if not Path(args.finetuned_checkpoint).is_file():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {args.finetuned_checkpoint}")
    if args.num_images < 1 or args.n_steps < 1:
        raise ValueError("num_images and n_steps must be positive.")
    if args.internal_batch_size < 1:
        raise ValueError("internal_batch_size must be positive.")
    if (args.mean is None) != (args.std is None):
        raise ValueError("Provide both --mean and --std, or neither.")
    if args.std is not None and any(s <= 0 for s in args.std):
        raise ValueError("All standard deviation values must be positive.")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_dir}")
    _, _, test_loader, _ = get_finetuning_loaders(str(dataset_dir))
    test_dataset = test_loader.dataset
    num_classes = len(test_dataset.classes)
    print(f"Classes: {test_dataset.classes}")
    print(f"Number of classes: {num_classes}")
    print(f"Test images: {len(test_dataset)}")
    model = load_model(args, num_classes, device)
    print("Fine-tuned model loaded successfully.")
    generate_attributions(model, test_dataset, args, device)


if __name__ == "__main__":
    main()


# python integrated_gradients.py \
#     --dataset_dir "./data/cifar10" \
#     --pretrained_checkpoint "./outputs/checkpoints/simclr/lars_pretrained_simclrv1_resnet18_checkpoint.pth" \
#     --finetuned_checkpoint "./outputs/finetune/cifar10/best_model.pth" \
#     --output_dir "./outputs/xai/cifar10/ig" \
#     --num_images 10
