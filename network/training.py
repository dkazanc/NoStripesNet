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
from metrics import gradient_sum_tv
from visualizers import BaseGANVisualizer, PairedWindowGANVisualizer


def trainBase(epochs, dataloader, model):
    vis = BaseGANVisualizer(model, dataloader, dataloader.dataset.size)
    num_batches = len(dataloader)
    print(f"Training has begun. Epochs: {epochs}, Batches: {num_batches}, Steps/batch: {dataloader.batch_size}")
    for epoch in range(epochs):
        dataloader.dataset.setMode('train')
        model.setMode('train')
        num_batches = len(dataloader)
        for i, (clean, *shifts) in enumerate(dataloader):
            centre = shifts[num_shifts // 2]
            # Pre-process data
            model.preprocess(centre, clean)
            # Run forward and backward passes
            model.run_passes()

            # Print out some useful info
            if i % 1 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{num_batches}], Loss_D: {model.lossD.item()}, "
                      f"Loss_G: {model.lossG.item()}")

        # At the end of every epoch, run through validate dataset
        print("Epoch finished. Validating model...")
        dataloader.dataset.setMode('validate')
        model.setMode('validate')
        num_batches = len(dataloader)
        validation_lossesG = torch.Tensor(num_batches)
        validation_lossesD = torch.Tensor(num_batches)
        for i, (clean, *shifts) in enumerate(dataloader):
            centre = shifts[num_shifts // 2]
            # Pre-process data
            model.preprocess(centre, clean)
            # Run forward and backward passes
            model.run_passes()
            # Print out some useful info
            if i % 1 == 0:
                print(f"\tEpoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{num_batches}], Loss_D: {model.lossD.item()}, "
                      f"Loss_G: {model.lossG.item()}")
            # Collate validation losses
            validation_lossesG[i] = model.lossG.item()
            validation_lossesD[i] = model.lossD.item()
        # Step scheduler with max of all validation losses (best practice?)
        model.schedulerG.step(validation_lossesG.max())
        model.schedulerD.step(validation_lossesD.max())
        print("Validation finished.")

        # At the end of every epoch, display some data and save model state
        vis.plot_losses()
        vis.plot_real_vs_fake_batch()
        vis.plot_real_vs_fake_recon()
        # Save models (could be made better using custom state dict, e.g. one that contains epoch number)
        saveModels = input("Save Models? (y/[n]): ")
        if saveModels == 'y' and model_save_dir and model_name:
            torch.save(gen.state_dict(), os.path.join(model_save_dir, f"gen_{model_name}_epoch{epoch}_sd.pt"))
            torch.save(disc.state_dict(), os.path.join(model_save_dir, f"disc_{model_name}_epoch{epoch}_sd.pt"))
            print(f"Models <{model_name}> saved to {model_save_dir}")
        else:
            print("Models not saved.")


def trainPairedWindows(epochs, dataloader, model):
    vis = PairedWindowGANVisualizer(model, dataloader, dataloader.dataset.size)
    num_batches = len(dataloader)
    total_step = num_batches * dataloader.batch_size * num_windows
    print(f"Training has begun. Epochs: {epochs}, Batches: {num_batches}, "
          f"Steps/batch: {num_windows * dataloader.batch_size}, Steps/epoch: {total_step}")
    for epoch in range(epochs):
        for i, (clean, centre, _) in enumerate(dataloader):
            # Pre-process data
            model.preprocess(centre, clean)
            # Run forward and backward passes
            model.run_passes()

            # Print out some useful info
            if i % 1 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i+1}/{num_batches}], Loss_D: {model.lossD_values[-1]}, "
                      f"Loss_G: {model.lossG_values[-1]}")

        # At the end of every epoch, run through validate dataset
        print("Epoch finished. Validating model...")
        dataloader.dataset.setMode('validate')
        model.setMode('validate')
        num_batches = len(dataloader)
        validation_lossesG = torch.Tensor(num_batches)
        validation_lossesD = torch.Tensor(num_batches)
        for i, (clean, centre, _) in enumerate(dataloader):
            # Pre-process data
            model.preprocess(centre, clean)
            # Run forward and backward passes
            model.run_passes()
            # Print out some useful info
            if i % 1 == 0:
                print(f"\tEpoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{num_batches}], Loss_D: {model.lossD.item()}, "
                      f"Loss_G: {model.lossG.item()}")
            # Collate validation losses
            validation_lossesG[i] = model.lossG.item()
            validation_lossesD[i] = model.lossD.item()
        # Step scheduler with max of all validation losses
        # best practice? - actually not sure if matters, it's all relative anyway
        model.schedulerG.step(validation_lossesG.max())
        model.schedulerD.step(validation_lossesD.max())
        print("Validation finished.")

        # At the end of every epoch, display some data and save model state
        vis.plot_losses()
        vis.plot_real_vs_fake_batch()
        vis.plot_real_vs_fake_recon()
        # Save models (could be made better using custom state dict, e.g. one that contains epoch number)
        saveModels = input("Save Models? (y/[n]): ")
        if saveModels == 'y' and model_save_dir and model_name:
            torch.save(gen.state_dict(), os.path.join(model_save_dir, f"gen_{model_name}_epoch{epoch}_sd.pt"))
            torch.save(disc.state_dict(), os.path.join(model_save_dir, f"disc_{model_name}_epoch{epoch}_sd.pt"))
            print(f"Models <{model_name}> saved to {model_save_dir}")
        else:
            print("Models not saved.")


if __name__ == '__main__':
    # Hyperparameters
    dataroot = os.path.join(os.pardir, 'data')
    model_save_dir = os.path.join(os.curdir, 'models')
    model_name = 'windowed-gan'
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
    dataset = PairedWindowDataset(root=dataroot, mode='train', tvt=(3, 1, 1), size=size, shifts=num_shifts,
                                  windowWidth=windowWidth, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create models
    disc = PairedWindowDiscriminator()
    disc.apply(init_weights)
    gen = PairedWindowUNet()
    gen.apply(init_weights)
    model = WindowGAN(windowWidth, gen, disc, mode='train', learning_rate=learning_rate, betas=betas)

    # Train
    trainPairedWindows(epochs, dataloader, model)
