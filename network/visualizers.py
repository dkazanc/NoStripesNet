import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
from torchvision import utils
from tomobar.methodsDIR import RecToolsDIR

from datasets import *
from models.gans import *


def getRectools2D(size, device='cpu'):
    total_angles = int(0.5 * np.pi * size)
    angles = np.linspace(0, 179.9, total_angles, dtype='float32')
    angles_rad = angles * (np.pi / 180.0)
    p = int(np.sqrt(2) * size)
    rectools = RecToolsDIR(DetectorsDimH=p,
                           DetectorsDimV=None,
                           CenterRotOffset=0.0,
                           AnglesVec=angles_rad,
                           ObjSize=size,
                           device_projector=device)
    return rectools


def batch_reconstruct(batch, size, device='cpu'):
    rectools = getRectools2D(size, device)
    p = int(np.sqrt(2) * size)
    assert (batch.shape[1] == 1)
    new_batch = torch.zeros((batch.shape[0], 1, size, size))
    for i in range(batch.shape[0]):  # is looping best idea? possible to vectorise?
        sinogram = batch[i].squeeze().numpy()
        sinogram = np.delete(sinogram, np.s_[p:], axis=-1)  # crop sinogram to correct size
        new_batch[i] = torch.from_numpy(rectools.FBP(sinogram)).unsqueeze(0)
    return new_batch


class BaseGANVisualizer:
    def __init__(self, model, dataset, size):
        self.model = model
        self.gen = self.model.gen
        self.disc = self.model.disc
        self.dataset = dataset
        self.size = size

    def plot_losses(self):
        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.model.lossG_values, label="G")
        plt.plot(self.model.lossD_values, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_one(self):
        item = np.random.randint(0, self.model.realA.shape[0])
        clean = self.model.realB.detach()[item]
        stripe = self.model.realA.detach()[item]
        fake = self.model.fakeB.detach()[item]
        images = [clean, stripe, fake]
        images += batch_reconstruct(torch.stack(images, dim=0), self.size)
        for i, img in enumerate(images):
            plt.subplot(2, 3, i + 1)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.axis('off')
        plt.show()

    def plot_real_vs_fake_batch(self):
        """Function to plot a batch of real inputs, target outputs and generated outputs.
        Before running this function, at least one train or test pass must have been made."""
        # Plot the target outputs (i.e. clean sinograms)
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 3, 1)
        plt.axis("off")
        plt.title("Target Outputs")
        plt.imshow(np.transpose(utils.make_grid(self.model.realB.detach(), padding=5, normalize=True, nrow=4).cpu(),
                                (1, 2, 0)), cmap='gray')
        # Plot the real inputs (i.e. centre sinograms)
        plt.subplot(1, 3, 2)
        plt.axis("off")
        plt.title("Real Inputs")
        plt.imshow(np.transpose(utils.make_grid(self.model.realA.detach(), padding=5, normalize=True, nrow=4).cpu(),
                                (1, 2, 0)), cmap='gray')
        # Plot the fake outputs (i.e. generated sinograms)
        plt.subplot(1, 3, 3)
        plt.axis("off")
        plt.title("Generated Outputs")
        plt.imshow(np.transpose(utils.make_grid(self.model.fakeB.detach(), padding=5, normalize=True, nrow=4).cpu(),
                                (1, 2, 0)), cmap='gray')
        plt.show()

    def plot_real_vs_fake_recon(self):
        """Function to plot a batch of *reconstructed* real inputs, target outputs and generated outputs.
        Before running this function, at least one train or test pass must have been made."""
        # Reconstruct all
        input_recon = batch_reconstruct(self.model.realA.detach(), self.size)
        target_recon = batch_reconstruct(self.model.realB.detach(), self.size)
        fake_recon = batch_reconstruct(self.model.fakeB.detach(), self.size)
        # Plot clean vs centre vs generated reconstructions
        plt.figure(figsize=(8, 8))
        plt.subplot(131)
        plt.axis("off")
        plt.title("Targets")
        plt.imshow(np.transpose(utils.make_grid(target_recon, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
                   cmap='gray')
        plt.subplot(132)
        plt.axis("off")
        plt.title("Inputs")
        plt.imshow(np.transpose(utils.make_grid(input_recon, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
                   cmap='gray')
        plt.subplot(133)
        plt.axis("off")
        plt.title("Generated")
        plt.imshow(np.transpose(utils.make_grid(fake_recon, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
                   cmap='gray')
        plt.show()

    def plot_disc_predictions(self):
        """Function to plot a series of real and fake images, with the label predicted by the discriminator"""
        real_outputs = torch.cat((self.model.realA[:5], self.model.realB[:5]), dim=1)
        fake_outputs = torch.cat((self.model.realA[:5], self.model.fakeB[:5]), dim=1)
        disc_inputs = torch.cat((real_outputs, fake_outputs), dim=0)
        disc_outputs = self.model.disc(disc_inputs).detach()
        disc_inputs.detach_()
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(disc_inputs[i][1], cmap='gray')
            plt.axis('off')
            plt.title(f"Pred: {'Real' if disc_outputs[i] else 'Fake'}")
        plt.show()


class PairedWindowGANVisualizer(BaseGANVisualizer):
    def __init__(self, model, dataset, size):
        super().__init__(model, dataset, size)

    def plot_real_vs_fake_batch(self):
        self.model.realA = self.dataset.combineWindows(self.model.realAs)
        self.model.realB = self.dataset.combineWindows(self.model.realBs)
        self.model.fakeB = self.dataset.combineWindows(self.model.fakeBs)
        super().plot_real_vs_fake_batch()

    def plot_real_vs_fake_recon(self):
        self.model.realA = self.dataset.combineWindows(self.model.realAs)
        self.model.realB = self.dataset.combineWindows(self.model.realBs)
        self.model.fakeB = self.dataset.combineWindows(self.model.fakeBs)
        super().plot_real_vs_fake_recon()

    def plot_one(self):
        self.model.realA = self.dataset.combineWindows(self.model.realAs)
        self.model.realB = self.dataset.combineWindows(self.model.realBs)
        self.model.fakeB = self.dataset.combineWindows(self.model.fakeBs)
        item = np.random.randint(0, self.model.realA.shape[0])
        clean = self.model.realB.detach()[item]
        stripe = self.model.realA.detach()[item]
        fake = self.model.fakeB.detach()[item]
        images = [clean, stripe, fake]
        images += batch_reconstruct(torch.stack(images, dim=0), self.size)
        for i, img in enumerate(images):
            plt.subplot(2, 3, i + 1)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.axis('off')
        plt.show()


class MaskedVisualizer(BaseGANVisualizer):
    def plot_one(self):
        item = np.random.randint(0, self.model.realA.shape[0])
        clean = self.model.realB.detach()[item]
        stripe = self.model.realA.detach()[item]
        mask = self.model.mask.detach()[item]
        gen_in = stripe.clone()
        gen_in[mask] = 0
        gen_out = self.model.fakeB.detach()[item]
        images = [clean, stripe, mask, gen_in, gen_out]
        images += batch_reconstruct(torch.stack(images, dim=0), self.size)

        for i, img in enumerate(images):
            plt.subplot(2, 5, i+1)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.axis('off')
        plt.show()


class MetricVisualizer:
    def __init__(self, size, subplot_size, figsize=(20, 10)):
        self.size = size
        self.rectools = getRectools2D(size)
        self.subplot_size = subplot_size
        self.i = 1
        self.fig = plt.figure(figsize=figsize)

    def plot_box(self, data, titles=None):
        plt.close(self.fig)
        if titles is None:
            titles = []
        new_fig = plt.figure(figsize=(20, 10))
        ax = new_fig.add_subplot()
        bp = ax.boxplot(data, meanline=True, showmeans=True, vert=False)
        ax.set_yticklabels(titles)
        for median in bp['medians']:
            coords = median.get_xydata()[0]
            ax.annotate(round(median.get_xdata()[0], 4), coords, xytext=(coords[0]-0.01, coords[1]-0.2))
            median.set(color='red')
        for mean in bp['means']:
            coords = mean.get_xydata()[0]
            ax.annotate(round(mean.get_xdata()[0], 4), coords, xytext=(coords[0] - 0.01, coords[1] - 0.2))
            mean.set(color='blue', label=mean.get_xdata()[0])
        legend_elements = [Line2D([0], [0], color='red', label='Median'),
                           Line2D([0], [0], color='blue', linestyle='--', label='Mean')]
        ax.legend(handles=legend_elements, loc='upper left', fontsize='large')

    def plot_img(self, image, title='', cmap='gray'):
        plt.subplot(*self.subplot_size, self.i)
        self.i += 1
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')

    def plot_recon(self, sino, title='', cmap='gray'):
        self.plot_img(self.rectools.FBP(sino), title, cmap)
