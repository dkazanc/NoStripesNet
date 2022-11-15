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
        )

    def forward(self, x):
        return self.net(x)


class PairedFullDiscriminator(nn.Module):
    def __init__(self):
        super(PairedFullDiscriminator, self).__init__()

        filters = 64
        self.net = nn.Sequential(
            # Input (2, 402, 362) -> Output (16, 200, 180)
            self.down(2, filters, batchNorm=False),

            # Input (16, 200, 180) -> Output (32, 99, 89)
            self.down(filters, filters*2),

            # Input (32, 99, 89) -> Output (64, 48, 43)
            self.down(filters*2, filters*4),

            # Input (64, 48, 43) -> Output (128, 23, 20)
            self.down(filters*4, filters*8),

            # Input (128, 23, 20) -> Output (256, 10, 9)
            self.down(filters*8, filters*8),

            # Input (256, 10, 9) -> Output (512, 4, 4)
            self.down(filters*8, filters*8, p=(0, 1)),

            # Input (512, 4, 4) -> Output (1, 1, 1)
            nn.Conv2d(filters*8, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False),
        )

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def down(in_c, out_c, batchNorm=True, k=(4, 4), s=(2, 2), p=(0, 0)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001)
        else:
            batchNorm = nn.Identity()
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            batchNorm,
            nn.LeakyReLU(0.2, inplace=True),
        )
