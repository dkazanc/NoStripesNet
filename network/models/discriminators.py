import torch.nn as nn


class SinoDiscriminator(nn.Module):
    def __init__(self):
        super(SinoDiscriminator, self).__init__()

        filters = 16
        epsilon = 0.001
        self.net = nn.Sequential(
            # Input (2, 402, 362) -> Output (16, 201, 181)
            nn.Conv2d(2, filters, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (16, 201, 181) -> Output (32, 100, 90)
            nn.Conv2d(filters, filters*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filters*2, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (32, 100, 90) -> Output (64, 50, 45)
            nn.Conv2d(filters*2, filters*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filters*4, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (64, 50, 45) -> Output (128, 25, 22)
            nn.Conv2d(filters*4, filters*8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filters*8, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (128, 25, 22) -> Output (256, 12, 11)
            nn.Conv2d(filters*8, filters*16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filters*16, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (256, 12, 11) -> Output (512, 6, 5)
            nn.Conv2d(filters*16, filters*32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filters*32, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (512, 6, 5) -> Output (1, 1, 1)
            nn.Conv2d(filters*32, 1, (6, 5), bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class PairedWindowDiscriminator(nn.Module):
    def __init__(self):
        super(PairedWindowDiscriminator, self).__init__()

        filters = 16
        epsilon = 0.001
        self.net = nn.Sequential(
            # Input (2, 402, 25) -> Output (16, 402, 22)
            nn.Conv2d(2, filters, (1, 4), stride=(1, 1), padding=(0, 0), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (16, 402, 22) -> Output (32, 200, 19)
            nn.Conv2d(filters, filters*2, (4, 4), stride=(2, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters*2, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (32, 200, 19) -> Output (64, 99, 19)
            nn.Conv2d(filters*2, filters*4, (4, 4), stride=(2, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters*4, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (64, 99, 19) -> Output (128, 48, 16)
            nn.Conv2d(filters*4, filters*8, (4, 4), stride=(2, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters*8, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (128, 48, 16) -> Output (256, 23, 13)
            nn.Conv2d(filters*8, filters*16, (4, 4), stride=(2, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters*16, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (256, 23, 13) -> Output (512, 10, 7)
            nn.Conv2d(filters*16, filters*32, (4, 4), stride=(2, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters*32, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (512, 10, 7) -> Output (512, 4, 4)
            nn.Conv2d(filters*32, filters*32, (4, 4), stride=(2, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters*32, eps=epsilon),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (512, 4, 4) -> Output (1, 1, 1)
            nn.Conv2d(filters*32, 1, (4, 4), stride=(2, 1), padding=(0, 0), bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
