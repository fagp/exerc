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
        ax.scatter(list(emb_1[:, 0]), list(emb_1[:, 1]), c=colors[i])

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
    ax.scatter(list(emb_1[:, 0]), list(emb_1[:, 1]), c="red")
    ax.scatter(list(emb_2[:, 0]), list(emb_2[:, 1]), c="black")
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


def visualize_retrieval(test_dataset, model, device):
    model.eval()
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)
    for i, (positives, anchor, negatives) in enumerate(test_dataset):
        anchor = anchor.unsqueeze(0)
        positives = positives.unsqueeze(0)
        negatives = negatives.unsqueeze(0)

        for _ in range(4):
            x, _, y = test_dataset[i]
            anchor = torch.cat((anchor, anchor[0].unsqueeze(0)), 0)
            positives = torch.cat((positives, x.unsqueeze(0)), 0)
            negatives = torch.cat((negatives, y.unsqueeze(0)), 0)

        emb_0, emb_1, emb_2 = model(
            wrap_input(anchor).to(device),
            wrap_input(positives).to(device),
            wrap_input(negatives).to(device),
        )
        pos_dist = torch.nn.functional.pairwise_distance(emb_0, emb_1)
        neg_dist = torch.nn.functional.pairwise_distance(emb_0, emb_2)

        dist = unwrap_output(torch.cat((pos_dist, neg_dist), 0))
        retrieval = np.argsort(dist)
        exemplos = torch.cat((positives, negatives), 0).cpu().numpy()

        figure = plt.figure()
        plt.subplot(i, 11, 1, title="Anchor")
        plt.imshow(anchor[0].numpy().transpose(2, 1, 0) * 255)
        for k in range(10):
            plt.subplot(i, 11, k + 2)
            plt.imshow(exemplos.transpose(2, 1, 0) * 255)

    return torch.from_numpy(vimage[:, :, :3].transpose((2, 1, 0)))
