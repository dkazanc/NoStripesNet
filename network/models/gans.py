import torch
import torch.nn as nn
import torch.optim as optim


# Weights initialization function
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class BaseGAN:
    def __init__(self, gen, disc=None, mode='train', learning_rate=0.01, betas=(0.5, 0.999), lambdaL1=100.0):
        self.gen = gen
        self.disc = disc
        self.setMode(mode)
        self.lossD_values = []
        self.lossG_values = []

        # if training, create discriminator and set up loss & optimizer functions
        if self.mode == 'train':
            self.lossGAN = nn.BCEWithLogitsLoss()
            self.lossL1 = nn.L1Loss()
            self.lambdaL1 = lambdaL1
            self.optimizerG = optim.Adam(self.gen.parameters(), lr=learning_rate, betas=betas)
            self.optimizerD = optim.Adam(self.disc.parameters(), lr=learning_rate, betas=betas)
            self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=1, verbose=True)
            self.schedulerD = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=2, verbose=True)

    def setMode(self, mode, disc=None):
        if mode == 'train':
            self.gen.train()
            self.set_requires_grad(self.gen, True)
            if self.disc is not None:
                self.disc.train()
                self.set_requires_grad(self.disc, True)
            else:
                if disc is not None:
                    self.disc = disc
                    self.set_requires_grad(self.disc, True)
                else:
                    raise TypeError("If training (i.e. mode == 'train'), discriminator should be passed as argument.")
        elif mode == 'validate':
            self.gen.eval()
            self.set_requires_grad(self.gen, True)
            self.disc.eval()  # if this raises an error something really wrong has happened
            self.set_requires_grad(self.disc, True)
        elif mode == 'test':
            self.gen.eval()
            self.set_requires_grad(self.gen, False)
            if self.disc is not None:
                self.disc.eval()
                self.set_requires_grad(self.disc, False)
        else:
            raise ValueError(f"mode should be one of 'train', 'validate' or 'test'. Instead got '{mode}'.")
        self.mode = mode

    def preprocess(self, a, b):
        self.realA = a
        self.realB = b

    def forward(self):
        """Run forward pass, i.e. generate batch of fake images"""
        self.fakeB = self.gen(self.realA)

    def backwardD(self):
        """Run backward pass for discriminator"""
        # Step 1 - calculate discriminator loss on fake inputs
        fakeAB = torch.cat((self.realA, self.fakeB), dim=1)
        outFake = self.disc(fakeAB.detach())
        labels = torch.zeros_like(outFake)
        self.lossD_fake = self.lossGAN(outFake, labels)

        # Step 2 - calculate discriminator loss on real inputs
        realAB = torch.cat((self.realA, self.realB), dim=1)
        outReal = self.disc(realAB)
        labels = torch.ones_like(outReal)
        self.lossD_real = self.lossGAN(outReal, labels)

        # Step 3 - Combine losses and call backwards pass
        self.lossD = (self.lossD_fake + self.lossD_real) * 0.5
        self.lossD.backward()

    def backwardG(self):
        """Run backward pass for generator"""
        # Step 1 - Caluclate GAN loss for fake images, i.e. disc incorrect predictions
        fakeAB = torch.cat((self.realA, self.fakeB), dim=1)
        outFake = self.disc(fakeAB)
        labels = torch.ones_like(outFake)
        self.lossG_GAN = self.lossGAN(outFake, labels)

        # Step 2 - Calculate L1 loss for fake images, i.e. how similar fake were to real
        self.lossG_L1 = self.lossL1(self.fakeB, self.realB) * self.lambdaL1

        # Step 3 - Combine losses and call backwards pass
        self.lossG = self.lossG_GAN + self.lossG_L1
        self.lossG.backward()

    def run_passes(self):
        """Run forwards and backwards passes"""
        # Run forward pass
        self.forward()

        # If testing, only forward pass needs to be ran
        if self.mode != 'test':
            # Run backward pass for discriminator
            self.set_requires_grad(self.disc, True)
            self.optimizerD.zero_grad()
            self.backwardD()
            self.optimizerD.step()

            # Run backward pass for generator
            self.set_requires_grad(self.disc, False)  # stop gradient calculation for discriminator
            self.optimizerG.zero_grad()
            self.backwardG()
            self.optimizerG.step()

            # Add losses to list (mainly for plotting purposes)
            self.lossD_values.append(self.lossD.item())
            self.lossG_values.append(self.lossG.item())


    @staticmethod
    def set_requires_grad(network, grad):
        for param in network.parameters():
            param.requires_grad = grad



class WindowGAN(BaseGAN):
    def __init__(self, width, gen, disc=None, mode='train', learning_rate=0.01, betas=(0.5, 0.999), lambdaL1=100.0):
        super().__init__(gen, disc, mode=mode, learning_rate=learning_rate, betas=betas, lambdaL1=lambdaL1)
        self.windowWidth = width
        self.realAs, self.realBs = [], []

    def preprocess(self, a, b):
        self.realAs, self.realBs = [], []
        # We assume the network looks at one window at a time (i.e. channels = 1)
        # Therefore we can return data as is (it is already a list of windows), no need to change channels
        if not (isinstance(a, list) and isinstance(b, list)):
            raise TypeError(f"Inputs must be of type {list}. Instead got a: {type(a)} and b: {type(b)}")
        # Get all widths to be the same
        for window in a:
            self.realAs.append(nn.ReplicationPad2d((0, self.windowWidth - window.shape[-1], 0, 0))(window))
        for window in b:
            self.realBs.append(nn.ReplicationPad2d((0, self.windowWidth - window.shape[-1], 0, 0))(window))

    def run_passes(self):
        """Run forwards and backwards passes.
        Loop through list of windows and run forwards & backwards pass for each window"""
        self.fakeBs = []
        for i in range(len(self.realAs)):
            self.realA = self.realAs[i]
            self.realB = self.realBs[i]

            super().run_passes()

            self.fakeBs.append(self.fakeB)

    def expandWidth(self):
        a_copy = self.realA.clone()
        b_copy = self.realB.clone()
        if self.realA.shape[-1] < self.windowWidth:
            torch.nn.ReplicationPad2d((0, self.windowWidth - a_copy.shape[-1], 0, 0))(a_copy)
        if self.realB.shape[-1] < self.windowWidth:
            torch.nn.ReplicationPad2d((0, self.windowWidth - b_copy.shape[-1], 0, 0))(b_copy)
        self.realA = a_copy
        self.realB = b_copy
