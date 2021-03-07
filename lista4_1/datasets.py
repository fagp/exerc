import os
import torch
import random
import imageio
import numpy as np
from skimage.transform import resize, rescale, rotate


class ComparingDataset(torch.utils.data.Dataset):
    def __init__(self, images_path="/content/", train=True):
        images_path = os.path.join(images_path, "train" if train else "val")
        self.images = [
            os.path.join(images_path, f)
            for f in os.listdir(images_path)
            if ".jpg" in f
        ]

    def __len__(self):
        return len(self.images)

    def random_transform(self, image):
        angle = random.randint(0, 180)
        img_1 = rotate(image, angle, resize=False)

        scale = 0.2 * random.random() + 0.9
        img_1 = rescale(img_1, (scale, scale, 1))

        min_size = tuple([min(dim, 256) for dim in img_1.shape])
        top_left_x, top_left_y, _ = (
            np.array(img_1.shape) // 2 - np.array(min_size) // 2
        )

        img_1 = img_1[
            top_left_x : top_left_x + min_size[0],
            top_left_y : top_left_y + min_size[1],
            :,
        ]
        img_1 = resize(img_1, (256, 256))
        img_1 = img_1.astype(np.float) / 255.0

        return img_1

    def __getitem__(self, index):
        _img = imageio.imread(self.images[index])
        if _img.ndim == 2:
            _img = _img[:, :, np.newaxis]
            _img = np.concatenate((_img, _img, _img), 2)

        image = self.random_transform(_img)
        image_pos = self.random_transform(_img)

        image_neg = imageio.imread(
            self.images[random.randint(0, len(self.images))]
        )
        if image_neg.ndim == 2:
            image_neg = image_neg[:, :, np.newaxis]
            image_neg = np.concatenate((image_neg, image_neg, image_neg), 2)

        image_neg = self.random_transform(image_neg)

        return (
            torch.from_numpy(image.transpose(2, 1, 0)).float(),
            torch.from_numpy(image_pos.transpose(2, 1, 0)).float(),
            torch.from_numpy(image_neg.transpose(2, 1, 0)).float(),
        )


training_triplet_dataset = ComparingDataset(train=True)
test_triplet_dataset = ComparingDataset(train=False)
