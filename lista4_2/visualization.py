import torch
import random
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_to_image(figure):
    plt.savefig("temporal.png")
    plt.close(figure)
    tuple_image = imageio.imread("temporal.png")
    return tuple_image


def visualize_detection(dataset, net, device):
    # index = random.randint(0, len(dataset) - 1)
    index = 0
    image, bbox = dataset[index]

    output = net(image.unsqueeze(0).to(device))

    figure = plt.figure()
    _, ax = plt.subplots()
    ax.imshow(image.cpu().numpy().transpose(1, 0) * 255, cmap="gray")
    bbox_numpy = bbox.cpu().numpy() * 256
    rect = patches.Rectangle(
        tuple(bbox_numpy[:2]),
        bbox_numpy[2],
        bbox_numpy[3],
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    ax.add_patch(rect)

    reg_bbox_numpy = output[0].detach().cpu().numpy() * 256
    rect1 = patches.Rectangle(
        tuple(reg_bbox_numpy[:2]),
        reg_bbox_numpy[2],
        reg_bbox_numpy[3],
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect1)

    vimage = plot_to_image(figure)
    plt.close("all")
    return torch.from_numpy(vimage[:, :, :3].transpose((2, 1, 0)))