"""The discriminators in this module have been inspired by those described in
    Isola, P., Zhu, J.Y., Zhou, T. and Efros, A.A., 2016. Image-to-image
    translation with conditional adversarial networks. In Proceedings of the
    IEEE conference on computer vision and pattern recognition (pp. 1125-1134).
And the following GitHub repositories:
    - https://github.com/phillipi/pix2pix
    - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import torch.nn as nn


class BaseDiscriminator(nn.Module):
    """Basic Discriminator used by most GAN models.
    Takes a pair of 402 x 362 images as input and outputs a singleton float.
    Consists of 7 decoder layers, each with the following pattern:
        1. 2D Convolution
        2. Batch Normalization
        3. Leaky ReLU
    The first layer has no batch normalization, and the last layer has neither
    batch normalization nor ReLU.
    There is no activation function in the last layer, as this is handled by
    PyTorch's BCELossWithLogits() function.
    Additionally, if training an LSGAN, no final activation is required.
    """
    def __init__(self):
        super(BaseDiscriminator, self).__init__()

        filters = 64
        self.net = nn.Sequential(
            # Input (2, 402, 362) -> Output (64, 200, 180)
            self.down(2, filters, batchNorm=False),

            # Input (64, 200, 180) -> Output (128, 99, 89)
            self.down(filters, filters*2),

            # Input (128, 99, 89) -> Output (256, 48, 43)
            self.down(filters*2, filters*4),

            # Input (256, 48, 43) -> Output (512, 23, 20)
            self.down(filters*4, filters*8),

            # Input (512, 23, 20) -> Output (512, 10, 9)
            self.down(filters*8, filters*8),

            # Input (512, 10, 9) -> Output (512, 4, 4)
            self.down(filters*8, filters*8, p=(0, 1)),

            # Input (512, 4, 4) -> Output (1, 1, 1)
            nn.Conv2d(filters*8, 1,
                      kernel_size=(4, 4), stride=(2, 2),
                      padding=(0, 0), bias=True),
        )

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def down(in_c, out_c, batchNorm=True, k=(4, 4), s=(2, 2), p=(0, 0)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001,
                                       track_running_stats=False)
        else:
            batchNorm = nn.Identity()
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p,
                      bias=True),
            batchNorm,
            nn.LeakyReLU(0.2, inplace=True)
        )


class PatchDiscriminator(nn.Module):
    """Discriminator that runs on patches of size (1801, 256).
    For use when mode == 'patch'.
    Has a similar structure to BaseDiscriminator, but with (1, 1) padding in
    every layer.
    """
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        filters = 64
        self.down = BaseDiscriminator.down

        self.model = nn.Sequential(
            self.down(2, filters, p=(1, 1), batchNorm=False),
            self.down(filters, filters*2, p=(1, 1)),
            self.down(filters*2, filters*4, p=(1, 1)),
            self.down(filters*4, filters*8, p=(1, 1)),
            self.down(filters*8, filters*8, p=(1, 1)),
            self.down(filters*8, filters*8, p=(1, 1)),
            self.down(filters*8, filters*8, p=(1, 1)),
            nn.Conv2d(filters*8, 1, kernel_size=(4, 4), stride=(2, 2),
                      padding=(1, 1), bias=True)
        )

    def forward(self, x):
        return self.model(x)


class WindowDiscriminator(nn.Module):
    """Discriminator used by GANs training on windowed sinograms.
    Assumes window width = 25, and cannot scale to different widths.
    Takes a pair of 402 x 25 images as input and outputs a singleton float.
    Consists of 8 decoder layers, each with the following pattern:
        1. 2D Convolution
        2. Batch Normalization
        3. Leaky ReLU
    The first layer has no batch normalization, and the last layer has neither
    batch normalization nor ReLU.
    There is no activation function in the last layer, as this is handled by
    PyTorch's BCELossWithLogits() function.
    Additionally, if training an LSGAN, no final activation is required.
    Training on windowed sinograms did not give great results, and so this
    class might be deprecated soon.
    """
    def __init__(self):
        super(WindowDiscriminator, self).__init__()

        filters = 16
        epsilon = 0.001
        self.net = nn.Sequential(
            # Input (2, 402, 25) -> Output (16, 402, 22)
            nn.Conv2d(2, filters,
                      kernel_size=(1, 4), stride=(1, 1),
                      padding=(0, 0), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (16, 402, 22) -> Output (32, 200, 19)
            nn.Conv2d(filters, filters*2,
                      kernel_size=(4, 4), stride=(2, 1),
                      padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters*2, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (32, 200, 19) -> Output (64, 99, 19)
            nn.Conv2d(filters*2, filters*4,
                      kernel_size=(4, 4), stride=(2, 1),
                      padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters*4, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (64, 99, 19) -> Output (128, 48, 16)
            nn.Conv2d(filters*4, filters*8,
                      kernel_size=(4, 4), stride=(2, 1),
                      padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters*8, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (128, 48, 16) -> Output (256, 23, 13)
            nn.Conv2d(filters*8, filters*16,
                      kernel_size=(4, 4), stride=(2, 1),
                      padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters*16, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (256, 23, 13) -> Output (512, 10, 7)
            nn.Conv2d(filters*16, filters*32,
                      kernel_size=(4, 4), stride=(2, 1),
                      padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters*32, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (512, 10, 7) -> Output (512, 4, 4)
            nn.Conv2d(filters*32, filters*32,
                      kernel_size=(4, 4), stride=(2, 1),
                      padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters*32, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (512, 4, 4) -> Output (1, 1, 1)
            nn.Conv2d(filters*32, 1,
                      kernel_size=(4, 4), stride=(2, 1),
                      padding=(0, 0), bias=False),
        )

    def forward(self, x):
        return self.net(x)


class OldDiscriminator(nn.Module):
    """Discriminator used in previous versions of the project.
    Now deprecated and should not be used.
    """
    def __init__(self):
        super(OldDiscriminator, self).__init__()

        filters = 16
        epsilon = 0.001
        self.net = nn.Sequential(
            # Input (2, 402, 362) -> Output (16, 201, 181)
            nn.Conv2d(2, filters, (4, 4), stride=(2, 2), padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (16, 201, 181) -> Output (32, 100, 90)
            nn.Conv2d(filters, filters*2, (4, 4), stride=(2, 2), padding=1,
                      bias=False),
            nn.BatchNorm2d(filters*2, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (32, 100, 90) -> Output (64, 50, 45)
            nn.Conv2d(filters*2, filters*4, (4, 4), stride=(2, 2), padding=1,
                      bias=False),
            nn.BatchNorm2d(filters*4, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (64, 50, 45) -> Output (128, 25, 22)
            nn.Conv2d(filters*4, filters*8, (4, 4), stride=(2, 2), padding=1,
                      bias=False),
            nn.BatchNorm2d(filters*8, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (128, 25, 22) -> Output (256, 12, 11)
            nn.Conv2d(filters*8, filters*16, (4, 4), stride=(2, 2), padding=1,
                      bias=False),
            nn.BatchNorm2d(filters*16, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (256, 12, 11) -> Output (512, 6, 5)
            nn.Conv2d(filters*16, filters*32, (4, 4), stride=(2, 2), padding=1,
                      bias=False),
            nn.BatchNorm2d(filters*32, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (512, 6, 5) -> Output (1, 1, 1)
            nn.Conv2d(filters*32, 1, (6, 5), bias=False),
        )

    def forward(self, x):
        return self.net(x)
