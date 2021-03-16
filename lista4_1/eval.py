import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


def wrap_input(image):
    return image.to(torch.device("cpu"))


def unwrap_output(pred):
    return pred.detach().cpu().numpy()


def compute_test_AvgP(test_dataset, model):
    model.eval()
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)
    np.random.seed(0)
    score = np.empty((1,))
    target = np.empty((1,))
    for i, (positives, anchor, negatives) in enumerate(test_dataset):
        anchor = anchor.unsqueeze(0)
        positives = positives.unsqueeze(0)
        negatives = negatives.unsqueeze(0)

        for _ in range(2):
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
        score = np.concatenate((score, 1 - dist), 0)
        target = np.concatenate((target, np.array([1, 1, 1, 0, 0, 0])), 0)

    average_precision = average_precision_score(target[1:], score[1:])

    return average_precision