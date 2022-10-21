import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils

from tomobar.methodsDIR import RecToolsDIR

from models import BaseGAN, WindowGAN, init_weights
from models.discriminators import SinoDiscriminator, PairedWindowDiscriminator
from models.generators import SinoUNet, PairedWindowUNet
from datasets import PairedWindowDataset, BaseDataset


def batch_reconstruct(batch, size, device='cpu'):
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
    assert (batch.shape[1] == 1)
    new_batch = torch.zeros((batch.shape[0], 1, size, size))
    for i in range(batch.shape[0]):  # is looping best idea? possible to vectorise?
        sinogram = batch[i].squeeze().numpy()
        new_batch[i] = torch.from_numpy(rectools.FBP(sinogram)).unsqueeze(0)
    return new_batch


def trainBase(epochs, dataloader, model):
    lossG_values = []
    lossD_values = []
    for epoch in range(epochs):
        total_step = len(dataloader)
        for i, (clean, *shifts) in enumerate(dataloader):
            centre = shifts[num_shifts // 2]
            # Pre-process data
            model.preprocess(centre, clean)
            # Run forward and backward passes
            model.run_passes()

            # Print out some useful info
            if i % 1 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{total_step}], Loss_D: {model.lossD.item()}, "
                      f"Loss_G: {model.lossG.item()}")
                lossG_values.append(model.lossG.item())
                lossD_values.append(model.lossD.item())

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(lossG_values, label="G")
    plt.plot(lossD_values, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    return lossG_values, lossD_values


def testBase(dataloader, model):
    # Test
    model.setMode('test')
    # Get one batch from dataloader
    iterable_dataloader = iter(dataloader)
    next(iterable_dataloader)
    clean, *shifts = next(iterable_dataloader)
    centre = shifts[num_shifts // 2]

    # Generate fakes - only need to run forward pass
    with torch.no_grad():
        model.preprocess(centre, clean)
        model.forward()
        fakes = model.fakeB

    # Plot clean vs centre vs generated sinograms
    plt.figure(figsize=(8, 8))
    plt.subplot(131)
    plt.axis("off")
    plt.title("Clean")
    plt.imshow(np.transpose(utils.make_grid(clean, normalize=True, nrow=4).cpu(), (1, 2, 0)), cmap='gray')
    plt.subplot(132)
    plt.axis("off")
    plt.title("Centre")
    plt.imshow(np.transpose(utils.make_grid(centre, normalize=True, nrow=4).cpu(), (1, 2, 0)), cmap='gray')
    plt.subplot(133)
    plt.axis("off")
    plt.title("Generated")
    plt.imshow(np.transpose(utils.make_grid(fakes, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
               cmap='gray')
    plt.show()

    # Reconstruct all
    clean_recon = batch_reconstruct(clean, size)
    centre_recon = batch_reconstruct(centre, size)
    fake_recon = batch_reconstruct(fakes, size)
    # Plot clean vs centre vs generated reconstructions
    plt.figure(figsize=(8, 8))
    plt.subplot(131)
    plt.axis("off")
    plt.title("Clean")
    plt.imshow(np.transpose(utils.make_grid(clean_recon, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
               cmap='gray')
    plt.subplot(132)
    plt.axis("off")
    plt.title("Centre")
    plt.imshow(np.transpose(utils.make_grid(centre_recon, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
               cmap='gray')
    plt.subplot(133)
    plt.axis("off")
    plt.title("Generated")
    plt.imshow(np.transpose(utils.make_grid(fake_recon, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
               cmap='gray')
    plt.show()


def trainPairedWindows(epochs, dataloader, model):
    for epoch in range(epochs):
        total_step = len(dataloader) * windowWidth
        for i, (clean, centre, _) in enumerate(dataloader):
            # Pre-process data
            model.preprocess(centre, clean)
            # Run forward and backward passes
            model.run_passes()

            # Print out some useful info
            if i % 1 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{(i + 1) * windowWidth}/{total_step}], "
                      f"Loss_D: {model.lossD_values[-1]}, Loss_G: {model.lossG_values[-1]}")

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(model.lossG_values, label="G")
    plt.plot(model.lossD_values, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    return model.lossG_values, model.lossD_values


def testPairedWindows(dataloader, model):
    # Test
    model.setMode('test')
    # Get one batch from dataloader
    iterable_dataloader = iter(dataloader)
    next(iterable_dataloader)
    clean, centre, _ = next(iterable_dataloader)

    # Generate fakes - only need to run forward pass
    with torch.no_grad():
        model.preprocess(centre, clean)
        model.forward()
        fakes = model.fakeB

    # Combine windows into sinograms
    clean = dataloader.dataset.combineWindows(clean)
    centre = dataloader.dataset.combineWindows(centre)
    fakes = dataloader.dataset.combineWindows(fakes)

    # Plot clean vs centre vs generated sinograms
    plt.figure(figsize=(8, 8))
    plt.subplot(131)
    plt.axis("off")
    plt.title("Clean")
    plt.imshow(np.transpose(utils.make_grid(clean, normalize=True, nrow=4).cpu(), (1, 2, 0)), cmap='gray')
    plt.subplot(132)
    plt.axis("off")
    plt.title("Centre")
    plt.imshow(np.transpose(utils.make_grid(centre, normalize=True, nrow=4).cpu(), (1, 2, 0)), cmap='gray')
    plt.subplot(133)
    plt.axis("off")
    plt.title("Generated")
    plt.imshow(np.transpose(utils.make_grid(fakes, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
               cmap='gray')
    plt.show()

    # Reconstruct all
    clean_recon = batch_reconstruct(clean, size)
    centre_recon = batch_reconstruct(centre, size)
    fake_recon = batch_reconstruct(fakes, size)
    # Plot clean vs centre vs generated reconstructions
    plt.figure(figsize=(8, 8))
    plt.subplot(131)
    plt.axis("off")
    plt.title("Clean")
    plt.imshow(np.transpose(utils.make_grid(clean_recon, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
               cmap='gray')
    plt.subplot(132)
    plt.axis("off")
    plt.title("Centre")
    plt.imshow(np.transpose(utils.make_grid(centre_recon, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
               cmap='gray')
    plt.subplot(133)
    plt.axis("off")
    plt.title("Generated")
    plt.imshow(np.transpose(utils.make_grid(fake_recon, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
               cmap='gray')
    plt.show()


if __name__ == '__main__':
    # Hyperparameters
    dataroot = os.path.join(os.pardir, 'data')
    size = 256
    num_shifts = 5
    windowWidth = 25

    epochs = 1
    learning_rate = 0.0002
    betas = (0.5, 0.999)
    batch_size = 32
    subplot_size = (2, 3)

    # Create dataset and dataloader
    dataset = PairedWindowDataset(root=dataroot,
                                  mode='test', tvt=(1, 0, 0), size=size, shifts=num_shifts, windowWidth=windowWidth,
                                  transform=transforms.Compose([
                                      transforms.ToTensor()
                                  ]))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Create models
    disc = PairedWindowDiscriminator()
    disc.apply(init_weights)
    gen = PairedWindowUNet()
    gen.apply(init_weights)
    model = WindowGAN(windowWidth, gen, disc, train=True, learning_rate=learning_rate, betas=betas)

    # Train
    lossesG, lossesD = trainPairedWindows(epochs, dataloader, model)

    # Test
    testPairedWindows(dataloader, model)
