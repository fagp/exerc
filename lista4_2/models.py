import torch
import pytorch_lightning as pl
from .visualization import visualize_detection
from .datasets import train_dataset, test_dataset
from .eval import iou, accuracy
from copy import deepcopy


class LocationPL(pl.LightningModule):
    def __init__(self, model, input_channels=1, lr=1e-5):
        super().__init__()
        self.plot_iteration = 0
        self.iteration = 0
        self.net = model(input_channels)
        self.lr = lr

    def forward(self, x):
        return self.net(x.float())

    def loss_function(self, output, target):
        loss = torch.nn.functional.mse_loss(output, target)
        max_output, _ = torch.max(output, dim=1)
        max_target, _ = torch.ones_like(max_output)
        min_output, _ = torch.min(output, dim=1)
        min_target, _ = torch.zeros_like(min_output)
        return (
            loss
            + torch.abs(max_output - max_target).max()
            + torch.abs(min_output - min_target).max()
        )

    def training_step(self, batch, batch_ind):
        self.iteration += 1
        image, bbox = batch

        output = self.forward(image)
        loss = self.loss_function(output, bbox)
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

        target_bbox = deepcopy(bbox)
        regressed_bbox = deepcopy(output.detach())
        iou_val = iou(target_bbox, regressed_bbox).mean()
        self.logger.experiment.add_scalar("iou", iou_val, self.iteration)
        self.log(
            "iou",
            iou_val,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        acc_val = accuracy(target_bbox, regressed_bbox)
        self.logger.experiment.add_scalar("accuracy", acc_val, self.iteration)
        self.log(
            "accuracy",
            acc_val,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        if (batch_ind % 10) == 0:
            self.plot_iteration += 1
            self.logger.experiment.add_image(
                "Cell Detection",
                visualize_detection(test_dataset, self.net, self.device),
                self.plot_iteration,
                dataformats="CWH",
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train_model(model, batch_size=10, batch_acc=1, epochs=10, lr=1e-3):
    net = LocationPL(lr=lr, model=model)
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            # default_root_dir="/content/gdrive/MyDrive/Colab Notebooks/resultados_lista4.2/",
            gpus=1,
            max_epochs=epochs,
            progress_bar_refresh_rate=10,
            accumulate_grad_batches=batch_acc,
        )
    else:
        trainer = pl.Trainer(
            # default_root_dir="/content/gdrive/MyDrive/Colab Notebooks/resultados_lista4.2/",
            max_epochs=epochs,
            progress_bar_refresh_rate=10,
            accumulate_grad_batches=batch_acc,
        )

    trainer.fit(
        net,
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        ),
    )
    return net