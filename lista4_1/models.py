import torch
import pytorch_lightning as pl
from .datasets import test_triplet_dataset, training_triplet_dataset
from .visualization import visualize_space


class ConvNet(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(ConvNet, self).__init__()
        self.in_channels = input_channels

        filters = [64, 128, 256, 512]

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=filters[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=filters[0],
                out_channels=filters[1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=filters[1],
                out_channels=filters[2],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=filters[2],
                out_channels=filters[3],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
        )

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        x = self.features(x)
        z = self.avg_pool(x).view(-1, 512)
        embedding = self.fc(z)

        return embedding


class TripletConvNetPL(pl.LightningModule):
    def __init__(self, input_channels=3, lr=1e-3, criterion=None):
        super().__init__()
        self.net = ConvNet(input_channels)
        self.plot_iteration = 0
        self.lr = lr
        self.criterion = criterion

    def forward(self, x, y, z):
        return self.net(x), self.net(y), self.net(z)

    def triplet_loss_function(self, anchor, positive, negative):
        return self.criterion(anchor, positive, negative)

    def training_step(self, batch, batch_ind):
        anchor, positive, negative = batch

        emb_anchor, emb_positive, emb_negative = self.forward(
            anchor, positive, negative
        )
        loss = self.triplet_loss_function(
            emb_anchor, emb_positive, emb_negative
        )

        if (batch_ind % 40) == 0:
            self.plot_iteration += 1

            self.logger.experiment.add_image(
                "Embedded space",
                visualize_space(
                    training_triplet_dataset, self.net, self.device
                ),
                self.plot_iteration,
                dataformats="CWH",
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# calling training
def train_model(criterion, batch_size=10, epochs=10, lr=1e-3):
    net = TripletConvNetPL(lr=lr, criterion=criterion)
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            gpus=1, max_epochs=epochs, progress_bar_refresh_rate=10
        )
    else:
        trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=10)

    trainer.fit(
        net,
        torch.utils.data.DataLoader(
            training_triplet_dataset, batch_size=batch_size, shuffle=True
        ),
    )
    return net