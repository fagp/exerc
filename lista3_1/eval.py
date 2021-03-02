import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

def wrap_input(image):
  return image.unsqueeze(0).to(torch.device('cpu'))

def unwrap_output(pred):
  pred = F.softmax(pred,1)
  return pred[0].detach().cpu().argmax().numpy().tolist()
  
def compute_test_accuracy(test_dataset, model):
  pred_MNIST = []
  target_MNIST =[]
  
  model.eval()
  for image, target in test_dataset:
    pred = model(wrap_input(image))
    target_MNIST.append(target)
    pred_MNIST.append(unwrap_output(pred))
  
  return accuracy_score(target_MNIST,pred_MNIST)
