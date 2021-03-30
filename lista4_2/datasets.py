import os
import torch
import random
import imageio
import numpy as np
from skimage.transform import resize, rotate


class CellsDetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path="/content/",
        train=True,
    ):
        self.images_path = os.path.join(
            dataset_path, "train" if train else "val", "images"
        )
        self.labels_path = os.path.join(
            dataset_path, "train" if train else "val", "labels"
        )
        self.images = [f for f in os.listdir(self.images_path) if ".tif" in f]
        self.train = train

    def __len__(self):
        return len(self.images)

    def random_transform(self, image, label):
        if self.train:
            angle = random.randint(0, 180)
            image = rotate(image, angle, resize=False, mode="reflect")
            label = rotate(label, angle, resize=False)

        image = resize(image, (256, 256))
        scale = (
            255.0
            if image.dtype == np.uint8
            else (65535.0 if image.dtype == np.uint16 else 1.0)
        )
        image = image.astype(np.float) / scale

        label = (
            resize(label.astype(np.float), (256, 256), preserve_range=True) > 0
        ).astype(np.int8)

        y, x = np.where(label > 0)
        bbox = (
            np.array(
                [x.min(), y.min(), x.max() - x.min(), y.max() - y.min(), 256.0]
            )
            / 256.0
            if len(x) > 0 and len(y) > 0
            else np.array([0, 0, 0, 0, 0])
        )

        return image, bbox

    def __getitem__(self, index):
        _img = imageio.imread(
            os.path.join(self.images_path, self.images[index])
        )
        _lab = imageio.imread(
            os.path.join(self.labels_path, self.images[index])
        )

        if _img.ndim > 2:
            _img = _img[:, :, 0]

        image, bbox = self.random_transform(_img, _lab)

        return (
            torch.from_numpy(image.transpose(1, 0)).float(),
            torch.from_numpy(bbox).float(),
        )


train_dataset = CellsDetectionDataset(
    train=True, dataset_path="/media/fillo/_home/work/IN0996/exerc/lista4_2"
)
test_dataset = CellsDetectionDataset(
    train=False, dataset_path="/media/fillo/_home/work/IN0996/exerc/lista4_2"
)
