import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


def wrap_input(image):
    return image.unsqueeze(0).to(torch.device("cpu"))


def unwrap_output(pred):
    return pred.detach().cpu().numpy().transpose((2, 1, 0))


def compute_test_ssim(test_dataset, model):
    ssim_val = []
    model.eval()
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)
    for image, target in test_dataset:
        pred = model(wrap_input(image).to(device))
        pred = unwrap_output(pred[0])
        target = unwrap_output(target)
        ssim_val.append(
            ssim(
                target,
                pred,
                data_range=pred.max() - pred.min(),
                multichannel=True,
            )
        )

    return np.array(ssim_val).mean()