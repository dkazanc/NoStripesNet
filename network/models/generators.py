"""The generators in this module have been inspired by those described in
    Isola, P., Zhu, J.Y., Zhou, T. and Efros, A.A., 2016. Image-to-image
    translation with conditional adversarial networks. In Proceedings of the
    IEEE conference on computer vision and pattern recognition (pp. 1125-1134).
And the following GitHub repositories:
    - https://github.com/phillipi/pix2pix
    - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import torch
import torch.nn as nn


class BaseUNet(nn.Module):
    """Basic Generator used by most GAN models.
    Takes one 402 x 362 image as input, and outputs one 402 x 362 image.
    The architecture is inspired by the U-Net described in Isola et al. 2016.
    It consists of a 7-layer decoder, a 7-layer encoder, and skip connections
    between the two.
    Each decoder layer has the following structure:
        1. 2D Convolution
        2. Batch Normalization
        3. Leaky ReLU
    The first decoder layer has no batch normalization, and the last decoder
    layer has no batch normalization and a non-leaky ReLU.
    Each encoder layer has the following structure:
        1. 2D Tranposed Convolution
        2. Batch Normalization
        3. (Non-Leaky) ReLU
    The first 3 encoder layers also contain dropout with p=0.5 between (2)
    and (3).
    The final activation function is Tanh.
    """
    def __init__(self):
        super(BaseUNet, self).__init__()

        filters = 64

        # Input (1, 402, 362) -> Output (64, 200, 180)
        self.down1 = nn.Conv2d(1, filters,
                               kernel_size=(4, 4), stride=(2, 2),
                               padding=(0, 0), bias=True)

        # Input (64, 200, 180) -> Output (128, 99, 89)
        self.down2 = self.down(filters, filters * 2)

        # Input (128, 99, 89) -> Output (256, 48, 43)
        self.down3 = self.down(filters * 2, filters * 4)

        # Input (256, 48, 43) -> Output (512, 23, 20)
        self.down4 = self.down(filters * 4, filters * 8)

        # Input (512, 23, 20) -> Output (512, 10, 9)
        self.down5 = self.down(filters * 8, filters * 8)

        # Input (512, 10, 9) -> Output (512, 4, 4)
        self.down6 = self.down(filters * 8, filters * 8, p=(0, 1))

        # Input (512, 4, 4) -> Output (512, 1, 1)
        self.down7 = self.down(filters * 8, filters * 8, batchNorm=False)

        # Input (512, 1, 1) -> Output (512, 4, 4)
        self.up1 = self.up(filters*8, filters*8, dropout=True)

        # Input (1024, 4, 4) -> Output (512, 10, 9)
        # Input channels double due to skip connections
        self.up2 = self.up(filters*8 * 2, filters*8, dropout=True, k=(4, 3))

        # Input (1024, 10, 9) -> Output (512, 23, 20)
        self.up3 = self.up(filters*8 * 2, filters*8, dropout=True, k=(5, 4))

        # Input (1024, 23, 20) -> Output (256, 48, 43)
        self.up4 = self.up(filters*8 * 2, filters*4, k=(4, 5))

        # Input (512, 48, 43) -> Output (128, 99, 89)
        self.up5 = self.up(filters*4 * 2, filters * 2, k=(5, 5))

        # Input (256, 99, 89) -> Output (64, 200, 180)
        self.up6 = self.up(filters*2 * 2, filters)

        # Input (128, 200, 180) -> Output (1, 402, 362)
        self.up7 = self.up(filters * 2, 1, batchNorm=False)

        # Input (1, 402, 362) -> (1, 402, 362)
        self.final = nn.Sequential(
            nn.Tanh()
        )

    @staticmethod
    def down(in_c, out_c, batchNorm=True, k=(4, 4), s=(2, 2), p=(0, 0)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001,
                                       track_running_stats=False)
        else:
            batchNorm = nn.Identity()
        return nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_c, out_c,
                      kernel_size=k, stride=s,
                      padding=p, bias=True),
            batchNorm
        )

    @staticmethod
    def up(in_c, out_c, batchNorm=True, dropout=False, k=(4, 4), s=(2, 2),
           p=(0, 0)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001,
                                       track_running_stats=False)
        else:
            batchNorm = nn.Identity()
        if dropout:
            dropout = nn.Dropout(0.5)
        else:
            dropout = nn.Identity()
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_c, out_c,
                               kernel_size=k, stride=s,
                               padding=p, bias=True),
            batchNorm,
            dropout
        )

    def forward(self, x):
        # Downsampling
        down1_out = self.down1(x)
        down2_out = self.down2(down1_out)
        down3_out = self.down3(down2_out)
        down4_out = self.down4(down3_out)
        down5_out = self.down5(down4_out)
        down6_out = self.down6(down5_out)
        down7_out = self.down7(down6_out)

        # Bottom layer
        up1_out = self.up1(down7_out)

        # Upsampling
        up2_in = torch.cat((up1_out, down6_out), dim=1)
        up2_out = self.up2(up2_in)

        up3_in = torch.cat((up2_out, down5_out), dim=1)
        up3_out = self.up3(up3_in)

        up4_in = torch.cat((up3_out, down4_out), dim=1)
        up4_out = self.up4(up4_in)

        up5_in = torch.cat((up4_out, down3_out), dim=1)
        up5_out = self.up5(up5_in)

        up6_in = torch.cat((up5_out, down2_out), dim=1)
        up6_out = self.up6(up6_in)

        up7_in = torch.cat((up6_out, down1_out), dim=1)
        up7_out = self.up7(up7_in)

        final_out = self.final(up7_out)
        return final_out


class PatchUNet(nn.Module):
    """UNet to generate sinogram patches of size (1801, 256).
    It consists of an 8-layer decoder and 8-layer encoder, with skip
    connections inbetween the layers.
    It has a similar structure to BaseUNet, but with padding=(1, 1) for every
    layer.
    """
    def __init__(self):
        super(PatchUNet, self).__init__()
        filters = 64

        # Input (1, 1801, 256) -> Output (64, 900, 128)
        self.down1 = self.down(1, filters, batchNorm=False)

        # Input (64, 900, 128) -> Output (128, 450, 64)
        self.down2 = self.down(filters, filters*2)

        # Input (128, 450, 64) -> Output (256, 225, 32)
        self.down3 = self.down(filters*2, filters*4)

        # Input (256, 225, 32) -> Output (512, 112, 16)
        self.down4 = self.down(filters*4, filters*8)

        # Input (512, 112, 16) -> Output (512, 56, 8)
        self.down5 = self.down(filters*8, filters*8)

        # Input (512, 56, 8) -> Output (512, 28, 4)
        self.down6 = self.down(filters*8, filters*8)

        # Input (512, 28, 4) -> Output (512, 14, 2)
        self.down7 = self.down(filters*8, filters*8)

        # Input (512, 14, 2) -> Output (512, 7, 1)
        self.down8 = nn.Sequential(
            nn.Conv2d(filters*8, filters*8, kernel_size=(4, 4), stride=(2, 2),
                      padding=(1, 1), bias=True),
            nn.ReLU(inplace=True)
        )

        ####### UP #######

        # Input (512, 7, 1) -> Output (512, 14, 2)
        self.up1 = self.up(filters*8, filters*8, dropout=True)

        # Input (1024, 14, 2) -> Output (512, 28, 4)
        self.up2 = self.up(filters*8 * 2, filters*8, dropout=True)

        # Input (1024, 28, 4) -> Output (512, 56, 8)
        self.up3 = self.up(filters*8 * 2, filters*8, dropout=True)

        # Input (1024, 56, 8) -> Output (512, 112, 16)
        self.up4 = self.up(filters*8 * 2, filters*8)

        # Input (1024, 112, 16) -> Output (256, 225, 32)
        self.up5 = self.up(filters*8 * 2, filters*4, out_pad=(1, 0))

        # Input (512, 225, 32) -> Output (128, 450, 64)
        self.up6 = self.up(filters*4 * 2, filters*2)

        # Input (256, 450, 64) -> Output (64, 900, 128)
        self.up7 = self.up(filters*2 * 2, filters)

        # Input (128, 900, 128) -> Output (1, 1801, 256)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(filters * 2, 1, (4, 4), (2, 2), (1, 1),
                               output_padding=(1, 0)),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        down1_out = self.down1(x)
        down2_out = self.down2(down1_out)
        down3_out = self.down3(down2_out)
        down4_out = self.down4(down3_out)
        down5_out = self.down5(down4_out)
        down6_out = self.down6(down5_out)
        down7_out = self.down7(down6_out)
        down8_out = self.down8(down7_out)

        # Decoder
        up1_out = self.up1(down8_out)

        # Skip connections are concatenated
        up2_in = torch.cat((up1_out, down7_out), dim=1)
        up2_out = self.up2(up2_in)

        up3_in = torch.cat((up2_out, down6_out), dim=1)
        up3_out = self.up3(up3_in)

        up4_in = torch.cat((up3_out, down5_out), dim=1)
        up4_out = self.up4(up4_in)

        up5_in = torch.cat((up4_out, down4_out), dim=1)
        up5_out = self.up5(up5_in)

        up6_in = torch.cat((up5_out, down3_out), dim=1)
        up6_out = self.up6(up6_in)

        up7_in = torch.cat((up6_out, down2_out), dim=1)
        up7_out = self.up7(up7_in)

        up8_in = torch.cat((up7_out, down1_out), dim=1)
        up8_out = self.up8(up8_in)

        return up8_out

    @staticmethod
    def down(in_c, out_c, batchNorm=True, k=(4, 4), s=(2, 2), p=(1, 1)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001,
                                       track_running_stats=False)
        else:
            batchNorm = nn.Identity()
        return nn.Sequential(
            nn.Conv2d(in_c, out_c,
                      kernel_size=k, stride=s,
                      padding=p, bias=True),
            batchNorm,
            nn.LeakyReLU(0.2, inplace=True)
        )

    @staticmethod
    def up(in_c, out_c, batchNorm=True, dropout=False, k=(4, 4), s=(2, 2),
           p=(1, 1), out_pad=(0, 0)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001,
                                       track_running_stats=False)
        else:
            batchNorm = nn.Identity()
        if dropout:
            dropout = nn.Dropout(0.5)
        else:
            dropout = nn.Identity()
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c,
                               kernel_size=k, stride=s, output_padding=out_pad,
                               padding=p, bias=True),
            batchNorm,
            dropout,
            nn.ReLU(inplace=True),
        )


class WindowUNet(nn.Module):
    """Generator used by GANs training on windowed sinograms.
    Assumes window width = 25, and cannot scale to different widths.
    Takes one 402 x 25 image as input, and outputs one 402 x 25 image.
    The architecture is inspired by the U-Net described in Isola et al. 2016.
    It consists of an 8-layer decoder, an 8-layer encoder, and skip connections
    between the two.
    Each decoder layer has the following structure:
        1. 2D Convolution
        2. Batch Normalization
        3. Leaky ReLU
    The first decoder layer has no batch normalization, and the last decoder
    layer has no batch normalization and a non-leaky ReLU.
    Each encoder layer has the following structure:
        1. 2D Tranposed Convolution
        2. Batch Normalization
        3. (Non-Leaky) ReLU
    The first 3 encoder layers also contain dropout with p=0.5 between (2)
    and (3).
    The final activation function is Tanh.
    Training on windowed sinograms did not give great results, and so this
    class might be deprecated soon.
    """
    def __init__(self):
        super(WindowUNet, self).__init__()

        filters = 32

        # Input (1, 402, 25) -> Output (32, 402, 22)
        self.down1 = nn.Conv2d(1, filters,
                               kernel_size=(1, 4), stride=(1, 1),
                               padding=(0, 0))

        # Input (32, 402, 22) -> Output (64, 200, 19)
        self.down2 = self.down(filters, filters*2)

        # Input (64, 200, 19) -> Output (128, 99, 16)
        self.down3 = self.down(filters*2, filters*4)

        # Input (128, 99, 16) -> Output (256, 48, 13)
        self.down4 = self.down(filters*4, filters*8)

        # Input (256, 48, 13) -> Output (512, 23, 10)
        self.down5 = self.down(filters*8, filters*16)

        # Input (512, 23, 10) -> Output (512, 10, 7)
        self.down6 = self.down(filters*16, filters*16)

        # Input (512, 10, 7) -> Output (512, 4, 4)
        self.down7 = self.down(filters*16, filters*16)

        # Input (512, 4, 4) -> Output (512, 1, 1)
        self.down8 = self.down(filters*16, filters*16, batchNorm=False)

        # Input (512, 1, 1) -> Output (512, 4, 4)
        self.up1 = self.up(filters*16, filters*16, dropout=True)

        # Input (1024, 4, 4) -> Output (512, 10, 7)
        # Input channels double due to skip connections
        self.up2 = self.up(filters*16 * 2, filters*16, dropout=True)

        # Input (1024, 10, 7) -> Output (512, 23, 10)
        self.up3 = self.up(filters*16 * 2, filters*16, dropout=True, k=(5, 4))

        # Input (1024, 23, 10) -> Output (256, 48, 13)
        self.up4 = self.up(filters*16 * 2, filters*8)

        # Input (512, 48, 13) -> Output (128, 99, 16)
        self.up5 = self.up(filters*8 * 2, filters * 4, k=(5, 4))

        # Input (256, 99, 16) -> Output (64, 200, 19)
        self.up6 = self.up(filters*4 * 2, filters * 2)

        # Input (128, 402, 19) -> Output (32, 402, 22)
        self.up7 = self.up(filters*2 * 2, filters)

        # Input (64, 402, 25) -> Output (1, 402, 25)
        self.up8 = self.up(filters * 2, 1, k=(1, 4), s=(1, 1), batchNorm=False)

        # Input (1, 402, 25) -> (1, 402, 25)
        self.final = nn.Sequential(
            nn.Tanh()
        )

    @staticmethod
    def down(in_c, out_c, batchNorm=True, k=(4, 4), s=(2, 1), p=(0, 0)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001)
        else:
            batchNorm = nn.Identity()
        return nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p),
            batchNorm,
        )

    @staticmethod
    def up(in_c, out_c, batchNorm=True, dropout=False, k=(4, 4), s=(2, 1),
           p=(0, 0)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001,
                                       track_running_stats=False)
        else:
            batchNorm = nn.Identity()
        if dropout:
            dropout = nn.Dropout(0.5)
        else:
            dropout = nn.Identity()
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_c, out_c,
                               kernel_size=k, stride=s, padding=p),
            batchNorm,
            dropout
        )

    def forward(self, x):
        # Width of sinogram may not always be a multiple of 25,
        # so we must pad any windows with width < 25
        if x.shape[-1] < 25:
            down0_out = nn.ReplicationPad2d((0, 25 - x.shape[-1], 0, 0))(x)
        else:
            down0_out = nn.Identity()(x)
        # Downsampling
        down1_out = self.down1(down0_out)
        down2_out = self.down2(down1_out)
        down3_out = self.down3(down2_out)
        down4_out = self.down4(down3_out)
        down5_out = self.down5(down4_out)
        down6_out = self.down6(down5_out)
        down7_out = self.down7(down6_out)
        down8_out = self.down8(down7_out)

        # Bottom layer
        up1_out = self.up1(down8_out)

        # Upsampling
        up2_in = torch.cat((up1_out, down7_out), dim=1)
        up2_out = self.up2(up2_in)

        up3_in = torch.cat((up2_out, down6_out), dim=1)
        up3_out = self.up3(up3_in)

        up4_in = torch.cat((up3_out, down5_out), dim=1)
        up4_out = self.up4(up4_in)

        up5_in = torch.cat((up4_out, down4_out), dim=1)
        up5_out = self.up5(up5_in)

        up6_in = torch.cat((up5_out, down3_out), dim=1)
        up6_out = self.up6(up6_in)

        up7_in = torch.cat((up6_out, down2_out), dim=1)
        up7_out = self.up7(up7_in)

        up8_in = torch.cat((up7_out, down1_out), dim=1)
        up8_out = self.up8(up8_in)

        final_out = self.final(up8_out)

        return final_out
