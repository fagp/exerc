import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


def wrap_input(image):
    return image.unsqueeze(0).to(torch.device("cpu"))


def unwrap_output(pred):
    return pred[0].detach().cpu().numpy()


def compute_test_ssim(test_dataset, model):
    ssim = []
    model.eval()
    for image, target in test_dataset:
        pred = model(wrap_input(image))
        pred = unwrap_output(pred)
        ssim.append(ssim(target, pred, data_range=pred.max() - pred.min()))

    return np.array(ssim).mean()
