import os
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from torchvision.datasets import ImageFolder


def cache_imagefolder(split_dir, cache_path, size=None, labeled=True, num_threads=32, overwrite=False):
    if os.path.exists(cache_path) and not overwrite:
        raise FileExistsError(f"{cache_path} exists. Pass overwrite=True to replace it.")

    ds = ImageFolder(str(split_dir))
    paths = [p for p, _ in ds.samples]

    def read(p):
        with Image.open(p) as im:
            im = im.convert("RGB")
            if size is not None:
                im = im.resize((size, size), Image.BILINEAR)
            return np.asarray(im)

    with ThreadPoolExecutor(num_threads) as ex:
        arrays = list(tqdm(ex.map(read, paths), total=len(paths), desc="Caching"))

    data = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
    obj = ({"data": data,
            "labels": torch.tensor(ds.targets, dtype=torch.long),
            "classes": ds.classes} if labeled else data)

    tmp = str(cache_path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, cache_path)
    return obj


# cache_imagefolder(r"/Users/ignatiusboadi/Documents/Pycharm_Projects/self_supervised_learning/data/cifar10/train",
#                   "cifar10_train_u8.pt", labeled=True, overwrite=True)
# cache_imagefolder(r"C:\Users\iboad\PycharmProjects\self-supervised-learning\data\imb_cifar\imbalance_100",
#                   "cifar10_train_imb100_u8.pt", labeled=True, overwrite=True)
# cache_imagefolder(r"C:\Users\iboad\PycharmProjects\self-supervised-learning\data\cifar10\test",
#                   "cifar10_test_u8.pt", labeled=False, overwrite=True)
cache_imagefolder(r"/Users/ignatiusboadi/Documents/Pycharm_Projects/self_supervised_learning/data/cifar10/train",
                  "cifar10_finetune_train_u8.pt", labeled=True, overwrite=True)
cache_imagefolder(r"/Users/ignatiusboadi/Documents/Pycharm_Projects/self_supervised_learning/data/cifar10/test",
                  "cifar10_finetune_test_u8.pt", labeled=True, overwrite=True)
