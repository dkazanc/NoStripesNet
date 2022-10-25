import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils

from models import BaseGAN, WindowGAN, init_weights
from models.discriminators import SinoDiscriminator, PairedWindowDiscriminator
from models.generators import SinoUNet, PairedWindowUNet
from datasets import PairedWindowDataset, BaseDataset
from visualizers import BaseGANVisualizer, PairedWindowGANVisualizer


def trainBase(epochs, dataloader, model):
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


def trainPairedWindows(epochs, dataloader, model):
    total_step = len(dataloader) * num_windows
    print(f"Training has begun. Epochs: {epochs}, Batches: {len(dataloader)}, Steps/epoch: {total_step}\n"
          f"Data processed per batch: {num_windows * dataloader.batch_size}")
    for epoch in range(epochs):
        for i, (clean, centre, _) in enumerate(dataloader):
            # Pre-process data
            model.preprocess(centre, clean)
            # Run forward and backward passes
            model.run_passes()

            # Print out some useful info
            if i % 1 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{(i + 1) * num_windows}/{total_step}], "
                      f"Loss_D: {model.lossD_values[-1]}, Loss_G: {model.lossG_values[-1]}")


if __name__ == '__main__':
    # Hyperparameters
    dataroot = os.path.join(os.pardir, 'data')
    size = 256
    num_shifts = 5
    windowWidth = 25
    num_windows = (int(np.sqrt(2) * size) // windowWidth + 1)

    epochs = 1
    learning_rate = 0.0002
    betas = (0.5, 0.999)
    batch_size = 32
    subplot_size = (2, 3)

    # Create dataset and dataloader
    dataset = PairedWindowDataset(root=dataroot,
                                  mode='train', tvt=(1, 0, 0), size=size, shifts=num_shifts, windowWidth=windowWidth,
                                  transform=transforms.Compose([
                                      transforms.ToTensor()
                                  ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create models
    disc = PairedWindowDiscriminator()
    disc.apply(init_weights)
    gen = PairedWindowUNet()
    gen.apply(init_weights)
    model = WindowGAN(windowWidth, gen, disc, mode='train', learning_rate=learning_rate, betas=betas)
    vis = PairedWindowGANVisualizer(model, dataloader, size)

    # Train
    trainPairedWindows(epochs, dataloader, model)
    vis.plot_losses()

    # Visualize some data
    model.setMode('test')  # might not be necessary
    vis.plot_real_vs_fake_batch()
    vis.plot_real_vs_fake_recon()
