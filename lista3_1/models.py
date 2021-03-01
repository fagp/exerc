import torch
import torch.nn as nn
from torch.utils import data
import pytorch_lightning as pl


# pytorch lightning module
class LeNetPL(pl.LightningModule):
    def __init__(self, model, num_classes=10):
        super().__init__()
        self.lenet = model(num_classes)
        self.num_classes = num_classes

    def forward(self, input):
        try:
            y = self.lenet(input.float())
        except Exception:
            raise ValueError(
                "This network doesn't allow a dataset with different image size"
            )
        return y

    def loss_function(self, output, target):
        one_hot = nn.functional.one_hot(target.long(), self.num_classes).to(
            self.device
        )
        loss = nn.functional.binary_cross_entropy_with_logits(
            output, one_hot.float()
        )

        return loss

    def training_step(self, batch, batch_ind):
        x, y = batch
        z = self.forward(x)
        loss = self.loss_function(z, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# calling trainning
def train_model(model, dataset, batch_size=1, epochs=10):
    net = LeNetPL(model)
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            gpus=1, max_epochs=epochs, progress_bar_refresh_rate=40
        )
    else:
        trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=40)

    trainer.fit(net, data.DataLoader(dataset, batch_size=batch_size))
    return trainer