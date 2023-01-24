import segmentation_models_pytorch as smp
from .base import BaseModule
import torch

class UNet(BaseModule):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        self.unet1 = smp.Unet(
            encoder_name=self.hparams.encoder,
            encoder_weights=self.hparams.pretrained,
            in_channels=self.hparams.in_channels1,
            classes=1,
        )
        self.unet2 = smp.Unet(
            encoder_name=self.hparams.encoder,
            encoder_weights=self.hparams.pretrained,
            in_channels=self.hparams.in_channels2,
            classes=1,
        )
        self.unet3 = smp.Unet(
            encoder_name=self.hparams.encoder,
            encoder_weights=self.hparams.pretrained,
            in_channels=self.hparams.in_channels3,
            classes=1,
        )       
        #self.comv_out = torch.nn.Conv2d(3,1,1)
        self.comv_out = torch.nn.Conv2d(3,1,3, padding=1)

    def forward(self, x):
        s1s, s2s, s3s = x
        x1 = self.unet1(s1s.squeeze(1))
        x2 = self.unet2(s2s.squeeze(1))
        x3 = self.unet3(s3s.squeeze(1))
        x_ = torch.cat((x1,x2,x3), dim =1)

        out = self.comv_out(x_)
        return torch.sigmoid(out).squeeze(1)