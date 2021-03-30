import torch
from sklearn.metrics import accuracy_score


def iou(boxA, boxB):
    boxA = boxA[:, :4]
    boxB = boxB[:, :4]
    boxA[:, 2:] = boxA[:, 2:] + boxA[:, :2]
    boxB[:, 2:] = boxB[:, 2:] + boxB[:, :2]

    join_box = torch.cat((boxA.unsqueeze(0), boxB.unsqueeze(0)), 0)
    join_box_max = torch.max(join_box, 0).values
    join_box_min = torch.min(join_box, 0).values

    xA = join_box_max[:, 0]
    yA = join_box_max[:, 1]
    xB = join_box_min[:, 2]
    yB = join_box_min[:, 3]

    interArea = torch.nn.functional.relu(
        xB - xA + 1
    ) * torch.nn.functional.relu(yB - yA + 1)

    boxAArea = (boxA[:, 2] - boxA[:, 0] + 1) * (boxA[:, 3] - boxA[:, 1] + 1)
    boxBArea = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)

    iou_val = interArea / (boxAArea + boxBArea - interArea).float()
    return iou_val


def accuracy(boxA, boxB):
    boxA = (boxA[:, 4] > 0.5).int()
    boxB = (boxB[:, 4] > 0.5).int()

    return accuracy_score(
        boxA.cpu().numpy().tolist(), boxB.cpu().numpy().tolist()
    )
