import os
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path(r"C:\Users\iboad\PycharmProjects\self-supervised-learning\data\cifar10\train")
OUTPUT_DIR = Path(r"C:\Users\iboad\PycharmProjects\self-supervised-learning\data\imb_cifar")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", ]
NUM_CLASSES = len(CLASS_NAMES)
MAX_IMAGES_PER_CLASS = 5000

IMBALANCE_RATIOS = [10, 50, 100]

SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def create_imbalanced_dataset(source_dir: Path, imbalance_ratio: int, output_dir, seed: int = 42, ):
    rng = random.Random(seed)

    samples_per_class = [round(MAX_IMAGES_PER_CLASS * (1 / imbalance_ratio) ** (i / (NUM_CLASSES - 1)))
        for i in range(NUM_CLASSES)]

    dataset_dir = output_dir / f"imbalance_{imbalance_ratio}"

    if dataset_dir.exists():
        raise FileExistsError(f"{dataset_dir} already exists. Rename or remove it before rerunning.")

    print(f"\nCreating: {dataset_dir}")
    print(f"Target imbalance ratio: {imbalance_ratio}:1")

    for class_name, target_count in zip(CLASS_NAMES, samples_per_class):
        source_class_dir = source_dir / class_name
        target_class_dir = dataset_dir / class_name

        if not source_class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {source_class_dir}")

        images = sorted(p for p in source_class_dir.iterdir() if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS)

        if len(images) < target_count:
            raise ValueError(f"{class_name}: found {len(images)} images, but need {target_count}")

        selected_images = rng.sample(images, target_count)
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in selected_images:
            shutil.copy2(image_path, target_class_dir / image_path.name, )

        print(f"{class_name:12s}: {len(selected_images):5d} images")

    print(f"Finished: {dataset_dir}")


if __name__ == "__main__":
    for ratio in IMBALANCE_RATIOS:
        create_imbalanced_dataset(
            source_dir=SOURCE_DIR,
            output_dir=OUTPUT_DIR,
            imbalance_ratio=ratio,
            seed=SEED,
        )

    print("\nAll imbalanced datasets created successfully.")