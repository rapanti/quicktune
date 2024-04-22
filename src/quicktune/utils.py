import os
import sys

import torch
from torchvision.datasets import ImageFolder


def get_dataset_metafeatures(path: str):
    print("metafeatures not given, infer from dataset")
    if not path.endswith("train"):
        path = os.path.join(path, "train")
    assert os.path.exists(path), f"Path {path} does not exist."
    try:
        dataset = ImageFolder(path)
        num_samples = len(dataset)
        num_channels = 3 if dataset[0][0].mode == "RGB" else 1
        num_classes = len(dataset.classes)
        image_resolution = dataset[0][0].size[1]
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    meta_features = (
        torch.FloatTensor(
            [num_samples, num_channels, num_classes, image_resolution]
        ).reshape(1, -1)
        / 10000
    )
    return meta_features