import random
import torchvision
from torch.utils import data


# Randomly scaled MNIST dataset implementation
class ScaledMNist(data.Dataset):
    def __init__(self, train=True):
        self.data = torchvision.datasets.MNIST(
            root="./data",
            train=train,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        random_resize_transform = torchvision.transforms.Resize(
            size=random.randint(20, 40)
        )

        return (random_resize_transform(image), label)


# randomly scaled MNIST variables
scaled_MNIST_train = ScaledMNist(train=True)
scaled_MNIST_test = ScaledMNist(train=False)

# Original MNIST dataset (fixed size)
MNIST_train = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
MNIST_test = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)