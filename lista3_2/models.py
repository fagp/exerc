import torch
import torch.nn as nn
from torch.utils import data
import pytorch_lightning as pl
from skimage.metrics import structural_similarity as ssim

# Unet architecture
class Unet(nn.Module):
    def __init__(self, input_channels=3, num_classes=2):
        super(Unet, self).__init__()
        self.in_channels = input_channels
        self.n_classes = num_classes
        filters = [64, 128, 256, 512]

        self.down1 = unetDown(self.in_channels, filters[0])
        self.down2 = unetDown(filters[0], filters[1])
        self.down3 = unetDown(filters[1], filters[2])
        self.center = unetConv2(filters[2], filters[3])
        self.up3 = unetUp(filters[3] + filters[2], filters[2])
        self.up2 = unetUp(filters[2] + filters[1], filters[1])
        self.up1 = unetUp(filters[1] + filters[0], filters[0])
        self.final = nn.Conv2d(filters[0], self.n_classes, 1)

    def forward(self, inputs):
        x, befdown1 = self.down1(inputs)
        x, befdown2 = self.down2(x)
        x, befdown3 = self.down3(x)
        x = self.center(x)
        x = self.up3(befdown3, x)
        x = self.up2(befdown2, x)
        x = self.up1(befdown1, x)

        return self.final(x)


# Unet block
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetConv2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


# Unet encoder block
class unetDown(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs1 = self.down(outputs)
        return outputs1, outputs


# Unet decoder block
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        return self.conv(torch.cat([inputs1, outputs2], 1))


class UnetPL(pl.LightningModule):
    def __init__(self, input_channels=3, output_channels=3, lr=1e-3):
        super().__init__()
        self.unet = Unet(input_channels, output_channels)
        self.iteration = 0
        self.plot_iteration = 0
        self.lr = lr

    def forward(self, input):
        assert input.shape[1:] == (
            3,
            256,
            256,
        ), "A rede espera tensores de tamanho ({}, 3, 256, 256), mas foi recebido ({}, {}, {}, {})".format(
            input.shape[0],
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        )
        return self.unet(input.float())

    def loss_function(self, output, target):
        loss = nn.functional.mse_loss(output, target.float())
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

        ssim_val = ssim(
            y[0].cpu().numpy().transpose((2, 1, 0)),
            z[0].detach().cpu().numpy().transpose((2, 1, 0)),
            multichannel=True,
            data_range=z[0].detach().cpu().numpy().max()
            - z[0].detach().cpu().numpy().min(),
        )
        self.logger.experiment.add_scalar(
            "training_ssim", ssim_val, self.iteration
        )
        self.log(
            "training_ssim",
            ssim_val,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        if (batch_ind % 40) == 0:
            self.plot_iteration += 1
            self.logger.experiment.add_image(
                "Noisy image",
                x[0] * 255,
                self.plot_iteration,
                dataformats="CWH",
            )
            self.logger.experiment.add_image(
                "Ground Truth",
                y[0] * 255,
                self.plot_iteration,
                dataformats="CWH",
            )
            self.logger.experiment.add_image(
                "Reconstruction",
                z[0] * 255,
                self.plot_iteration,
                dataformats="CWH",
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# calling trainning
def train_model(dataset, batch_size=10, epochs=10, lr=1e-3):
    net = UnetPL(lr=lr)
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            default_root_dir="/content/gdrive/MyDrive/Colab Notebooks/resultados_lista3.2/",
            gpus=1,
            max_epochs=epochs,
            progress_bar_refresh_rate=10,
        )
    else:
        trainer = pl.Trainer(
            default_root_dir="/content/gdrive/MyDrive/Colab Notebooks/resultados_lista3.2/",
            max_epochs=epochs,
            progress_bar_refresh_rate=10,
        )

    trainer.fit(
        net,
        data.DataLoader(dataset, batch_size=batch_size, shuffle=True),
    )
    return net
