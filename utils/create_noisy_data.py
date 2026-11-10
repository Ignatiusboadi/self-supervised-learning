import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
BASE_DIR = Path(BASE_DIR)
SOURCE_DIR = BASE_DIR / Path(r"data/cifar10")
OUTPUT_DIR = BASE_DIR / Path(r"data")

VAL_FRACTION = 0.20
NOISE_LEVELS = [0.20, 0.70]
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_class_images(directory):
    classes = sorted(d.name for d in directory.iterdir() if d.is_dir())
    samples = {}

    for class_name in classes:
        class_dir = directory / class_name
        samples[class_name] = sorted(p for p in class_dir.rglob("*") if p.is_file() \
                                     and p.suffix.lower() in IMAGE_EXTENSIONS)
    return classes, samples


def copy_samples(samples, destination):
    for class_name, paths in samples.items():
        class_dir = destination / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for src in tqdm(paths, desc=f"Copying {class_name}"):
            shutil.copy2(src, class_dir / src.name)


def split_train_val(train_dir, val_fraction=0.2, seed=42):
    classes, samples = get_class_images(train_dir)
    rng = random.Random(seed)
    train_samples = defaultdict(list)
    val_samples = defaultdict(list)

    for class_name in classes:
        paths = samples[class_name].copy()
        rng.shuffle(paths)
        n_val = round(len(paths) * val_fraction)
        val_samples[class_name] = paths[:n_val]
        train_samples[class_name] = paths[n_val:]

    return classes, train_samples, val_samples


def create_noisy_train(clean_samples, classes, destination, noise_level,
    seed=42):
    rng = random.Random(seed)
    noisy_samples = {class_name: paths.copy() for class_name, paths in clean_samples.items()}
    all_samples = [(class_name, path) for class_name, paths in clean_samples.items() for path in paths]
    n_noisy = round(len(all_samples) * noise_level)
    noisy_indices = set(rng.sample(range(len(all_samples)), n_noisy))
    for idx in noisy_indices:
        original_class, path = all_samples[idx]
        noisy_samples[original_class].remove(path)

    for idx in noisy_indices:
        original_class, path = all_samples[idx]
        possible_classes = [c for c in classes if c != original_class]
        new_class = rng.choice(possible_classes)
        noisy_samples[new_class].append(path)

    for class_name in classes:
        (destination / class_name).mkdir(parents=True, exist_ok=True)

    for class_name, paths in noisy_samples.items():
        for src in tqdm(paths, desc=f"Writing {class_name}"):
            shutil.copy2(src, destination / class_name / src.name)

    print(f"\nNoise level: {noise_level:.0%}")
    print(f"Total training images: {len(all_samples)}")
    print(f"Corrupted images: {n_noisy}")
    print(f"Actual noise: {n_noisy / len(all_samples):.2%}")


def prepare_dataset():
    source_train = SOURCE_DIR / "train"
    source_test = SOURCE_DIR / "test"
    classes, train_samples, val_samples = split_train_val(source_train, val_fraction=VAL_FRACTION, seed=SEED)
    val_dir = OUTPUT_DIR / "val"
    test_dir = OUTPUT_DIR / "test"

    print("\nCreating validation set...")
    copy_samples(val_samples, val_dir)
    print("\nCreating test set...")
    test_classes, test_samples = get_class_images(source_test)

    if test_classes != classes:
        raise ValueError("Train and test classes do not match.")

    copy_samples(test_samples, test_dir)
    for noise_level in NOISE_LEVELS:
        noise_percent = int(noise_level * 100)
        train_dir = OUTPUT_DIR / f"train_noise_{noise_percent}"
        print(f"\nCreating {noise_percent}% noisy training set...")
        create_noisy_train(clean_samples=train_samples, classes=classes, destination=train_dir,
            noise_level=noise_level, seed=SEED + noise_percent)


if __name__ == "__main__":
    prepare_dataset()
    print("\nDataset preparation completed!")
