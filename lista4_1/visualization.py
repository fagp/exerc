import torch
import random
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def visualize_10_space(dataset, net, device):
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    for i in range(10):
        positives, _, _ = dataset[i]
        positives = positives.unsqueeze(0)
        for _ in range(4):
            x, _, _ = dataset[i]
            positives = torch.cat((positives, x.unsqueeze(0)), 0)

        emb_1 = net(positives.to(device).float()).detach().cpu().numpy()

        circle1 = plt.Circle(np.mean(emb_1, 0), 1, fill=False)
        ax.scatter(list(emb_1[:, 0]), list(emb_1[:, 1]), c=colors[i])
        ax.add_patch(circle1)

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    vimage = np.fromstring(s, dtype="uint8").reshape(height, width, 4)

    return torch.from_numpy(vimage[:, :, :3].transpose((2, 1, 0)))


def visualize_space(dataset, net, device):
    positives, _, negatives = dataset[0]
    positives = positives.unsqueeze(0)
    negatives = negatives.unsqueeze(0)

    for _ in range(4):
        x, _, y = dataset[0]
        positives = torch.cat((positives, x.unsqueeze(0)), 0)
        negatives = torch.cat((negatives, y.unsqueeze(0)), 0)

    emb_1 = net(positives.to(device).float()).detach().cpu().numpy()
    emb_2 = net(negatives.to(device).float()).detach().cpu().numpy()

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    circle1 = plt.Circle(np.mean(emb_1, 0), 1, fill=False)
    ax.scatter(list(emb_1[:, 0]), list(emb_1[:, 1]), c="red")
    ax.scatter(list(emb_2[:, 0]), list(emb_2[:, 1]), c="black")
    ax.add_patch(circle1)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    vimage = np.fromstring(s, dtype="uint8").reshape(height, width, 4)

    return torch.from_numpy(vimage[:, :, :3].transpose((2, 1, 0)))


def plot_to_image(figure):
    plt.savefig("temporal.png")
    plt.close(figure)
    tuple_image = imageio.imread("temporal.png")
    return tuple_image


def visualize_tuple(dataset, net, device):
    index = random.randint(0, len(dataset))
    anchor, positives, negatives = dataset[index]

    emb_0 = net(anchor.unsqueeze(0).to(device).float()).detach()
    emb_1 = net(positives.unsqueeze(0).to(device).float()).detach()
    emb_2 = net(negatives.unsqueeze(0).to(device).float()).detach()

    pos_dist = torch.nn.functional.pairwise_distance(emb_0, emb_1)
    neg_dist = torch.nn.functional.pairwise_distance(emb_0, emb_2)

    figure = plt.figure()
    plt.subplot(1, 3, 1, title="Anchor")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(anchor.numpy().transpose(2, 1, 0) * 255)

    plt.subplot(1, 3, 2, title="Pos: {:1.4f}".format(pos_dist.item()))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(positives.numpy().transpose(2, 1, 0) * 255)

    plt.subplot(1, 3, 3, title="Neg: {:1.4f}".format(neg_dist.item()))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(negatives.numpy().transpose(2, 1, 0) * 255)

    vimage = plot_to_image(figure)

    return torch.from_numpy(vimage[:, :, :3].transpose((2, 1, 0)))
