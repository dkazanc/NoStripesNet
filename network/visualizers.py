import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import utils as tv_utils

from utils.tomography import reconstruct
from utils.misc import toNumpy


def batch_reconstruct(batch):
    """Function that takes in a batch of sinograms of shape (B, C, H, W)
    and returns the reconstruction of every item in that batch as tensors of
    shape (B, C, H, W)
        where B is batch size, C is no. channels, H is height, and W is width.
    Parameters:
        batch : torch.Tensor
            Batch of sinograms on which to calculate metrics
    """
    if batch.shape[1] != 1:
        raise NotImplementedError("Functionality for images with more than "
                                  "1 channel is not implemented.")
    batch = toNumpy(batch)  # new shape: (B, H, W)
    # Loop through each sinogram in batch, reconstruct it,
    # then append it to new array
    recons = []
    for sino in batch:
        recons.append(reconstruct(sino))
    recons = np.array(recons)
    if recons.ndim == 3:
        recons = recons[:, None, :, :]
    return torch.from_numpy(recons)


class BaseGANVisualizer:
    """Visualizer class to plot images during training/testing."""

    def __init__(self, model, dataset, block=True):
        """Parameters:
            model : BaseGAN
                The GAN model from which to plot data.
            dataset : torch.utils.data.Dataset
                The dataset from which data is being retrieved.
            block : bool
                Whether or not plots should pause execution of code.
                Default is True.
        """
        self.model = model
        self.gen = self.model.gen
        self.disc = self.model.disc
        self.dataset = dataset
        self.recon_size = None
        self.block = block

    def plot_losses(self):
        """Plot a graph of Generator and Discriminator losses."""
        # Plot losses
        fig, ax1 = plt.subplots()
        plt.figure(num=1, figsize=(15, 8))
        plt.title("Generator and Discriminator Loss During Training")
        # Generator
        l1 = ax1.plot(self.model.lossG_values, 'b-', label="G")
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Generator Loss")
        ax1.set_ylim([0, None])
        # Discriminator
        ax2 = ax1.twinx()
        l2 = ax2.plot(self.model.lossD_values, 'y-', label="D")
        ax2.set_ylabel("Discriminator Loss")
        ax2.set_ylim([0, None])
        # Legend
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc=0)
        if not self.block:
            classname = self.model.__class__.__name__
            num_losses = len(self.model.lossD_values)
            savename = f"./images/{classname}_{num_losses}_losses.png"
            plt.savefig(savename)
        plt.show(block=self.block)

    def plot_one(self):
        """Plot an image showing the following sinograms and their
        reconstructions:
            Target (i.e. clean sinogram with no stripes)
            Input (i.e. sinogram with stripes, input to Generator)
            Output (i.e. output from Generator, hopefully with no stripes)
        """
        item = np.random.randint(0, self.model.realA.shape[0])
        clean = self.model.realB.detach().cpu()[item]
        stripe = self.model.realA.detach().cpu()[item]
        fake = self.model.fakeB.detach().cpu()[item]
        images = [clean, stripe, fake]
        if self.recon_size is None:
            self.recon_size = round(self.model.realA.shape[-1] / np.sqrt(2))
        images += batch_reconstruct(torch.stack(images, dim=0))
        titles = ['Target', 'Input', 'Output']
        for i, img in enumerate(images):
            plt.subplot(2, 3, i + 1)
            plt.imshow(img.squeeze(), cmap='gray')
            if i in [0, 1, 2]:
                plt.clim(-1, 1)
            if i < len(titles):
                plt.title(titles[i])
            plt.axis('off')
        plt.show(block=self.block)

    def plot_real_vs_fake_batch(self):
        """Plot a batch of real inputs, target outputs and generated outputs.
        Before running this function, at least one train or test pass must have
        been made.
        """
        # Plot the target outputs (i.e. clean sinograms)
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 3, 1)
        plt.axis("off")
        plt.title("Target Outputs")
        plt.imshow(np.transpose(
            tv_utils.make_grid(self.model.realB.detach(), padding=5,
                               normalize=True, nrow=4).cpu(), (1, 2, 0)),
                   cmap='gray')
        # Plot the real inputs (i.e. centre sinograms)
        plt.subplot(1, 3, 2)
        plt.axis("off")
        plt.title("Real Inputs")
        plt.imshow(np.transpose(
            tv_utils.make_grid(self.model.realA.detach(), padding=5,
                               normalize=True, nrow=4).cpu(), (1, 2, 0)),
                   cmap='gray')
        # Plot the fake outputs (i.e. generated sinograms)
        plt.subplot(1, 3, 3)
        plt.axis("off")
        plt.title("Generated Outputs")
        plt.imshow(np.transpose(
            tv_utils.make_grid(self.model.fakeB.detach(), padding=5,
                               normalize=True, nrow=4).cpu(), (1, 2, 0)),
                   cmap='gray')
        plt.show(block=self.block)

    def plot_real_vs_fake_recon(self):
        """Plot a batch of *reconstructed* real inputs, target outputs and
        generated outputs.
        Before running this function, at least one train or test pass must have
        been made.
        """
        # Reconstruct all
        if self.recon_size is None:
            self.recon_size = round(self.model.realA.shape[-1] / np.sqrt(2))
        input_recon = batch_reconstruct(self.model.realA.detach().cpu())
        target_recon = batch_reconstruct(self.model.realB.detach().cpu())
        fake_recon = batch_reconstruct(self.model.fakeB.detach().cpu())
        # Plot clean vs centre vs generated reconstructions
        plt.figure(figsize=(8, 8))
        plt.subplot(131)
        plt.axis("off")
        plt.title("Targets")
        plt.imshow(np.transpose(
            tv_utils.make_grid(target_recon, normalize=True, nrow=4,
                               scale_each=True).cpu(), (1, 2, 0)),
                   cmap='gray')
        plt.subplot(132)
        plt.axis("off")
        plt.title("Inputs")
        plt.imshow(np.transpose(
            tv_utils.make_grid(input_recon, normalize=True, nrow=4,
                               scale_each=True).cpu(), (1, 2, 0)),
                   cmap='gray')
        plt.subplot(133)
        plt.axis("off")
        plt.title("Generated")
        plt.imshow(np.transpose(
            tv_utils.make_grid(fake_recon, normalize=True, nrow=4,
                               scale_each=True).cpu(), (1, 2, 0)),
                   cmap='gray')
        plt.show(block=self.block)

    def plot_disc_predictions(self):
        """Plot a series of real and fake images, with the label predicted by
        the discriminator.
        """
        real_outputs = torch.cat((self.model.realA[:5], self.model.realB[:5]),
                                 dim=1)
        fake_outputs = torch.cat((self.model.realA[:5], self.model.fakeB[:5]),
                                 dim=1)
        disc_inputs = torch.cat((real_outputs, fake_outputs), dim=0)
        disc_outputs = self.model.disc(disc_inputs).detach().cpu()
        disc_inputs = disc_inputs.detach().cpu()
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(disc_inputs[i][1], cmap='gray')
            plt.axis('off')
            pred = torch.sigmoid(disc_outputs[i])
            plt.title(f"Actual: {'Real' if i < 5 else 'Fake'}\n"
                      f"Pred: {'Real' if pred >= 0.5 else 'Fake'}")
        plt.show(block=self.block)


class MaskedVisualizer(BaseGANVisualizer):
    """Visualizer class to plot images during training/testing.
    Specifically for class MaskedGAN.
    """
    def plot_one(self):
        """Plot an image showing the following sinograms and their
        reconstructions:
            Target (i.e. clean sinogram with no stripes)
            With Artifacts (i.e. sinogram with stripes)
            Mask (i.e. binary mask showing where stripes are in sinogram)
            Input (i.e. sinogram with stripes masked, input to Generator)
            Output (i.e. output from Generator, hopefully with no stripes)
        """
        item = np.random.randint(0, self.model.realA.shape[0])
        clean = self.model.realB.detach().cpu()[item]
        stripe = self.model.realA.detach().cpu()[item]
        mask = self.model.mask.detach().cpu()[item]
        gen_in = stripe.clone()
        gen_in[mask] = 0
        gen_out = self.model.fakeB.detach().cpu()[item]
        images = [clean, stripe, mask, gen_in, gen_out]
        images += batch_reconstruct(torch.stack(images, dim=0))

        titles = ['Target', 'With Artifacts', 'Mask', 'Input', 'Output']
        for i, img in enumerate(images):
            plt.subplot(2, 5, i + 1)
            plt.imshow(img.squeeze(), cmap='gray')
            if i in [0, 1, 3, 4]:
                plt.clim(-1, 1)
            if i < len(titles):
                plt.title(titles[i])
            plt.axis('off')
        plt.show(block=self.block)
