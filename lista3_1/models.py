import torch
import torch.nn as nn
from torch.utils import data
import pytorch_lightning as pl


# pytorch lightning module
class LeNetPL(pl.LightningModule):
    def __init__(self, model, num_classes=10, lr=1e-3):
        super().__init__()
        self.lenet = model(num_classes)
        self.num_classes = num_classes
        self.accuracy = pl.metrics.Accuracy()
        self.iteration = 0
        self.lr = lr

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
        self.iteration += 1
        x, y = batch
        z = self.forward(x)
        loss = self.loss_function(z, y)

        self.logger.experiment.add_scalar(
            "training_loss", loss, self.iteration
        )
        self.log(
            "training_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        accuracy = self.accuracy(torch.nn.functional.softmax(z, 1), y)
        self.logger.experiment.add_scalar(
            "train_accuracy",
            accuracy,
            self.iteration,
        )
        self.log(
            "train_accuracy",
            accuracy,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# calling trainning
def train_model(model, dataset, batch_size=1, epochs=10, lr=1e-3):
    net = LeNetPL(model, lr=lr)
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            default_root_dir="/content/gdrive/MyDrive/Colab Notebooks/resultados_lista3.1/",
            gpus=1,
            max_epochs=epochs,
            progress_bar_refresh_rate=40,
        )
    else:
        trainer = pl.Trainer(
            default_root_dir="/content/gdrive/MyDrive/Colab Notebooks/resultados_lista3.1/",
            max_epochs=epochs,
            progress_bar_refresh_rate=40,
        )

    trainer.fit(
        net, data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    )
    return net
