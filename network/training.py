import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils

from models import BaseGAN, WindowGAN, init_weights
from models.discriminators import SinoDiscriminator, PairedWindowDiscriminator
from models.generators import SinoUNet, PairedWindowUNet
from visualizers import BaseGANVisualizer, PairedWindowGANVisualizer
from datasets import PairedWindowDataset, BaseDataset


def trainBase(epochs, dataloader, model, save_every_epoch=False, save_dir=None, save_name=None, verbose=True):
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
            if verbose:
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
            if verbose:
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
        # Save models
        if save_every_epoch and save_dir is not None and save_name is not None:
            torch.save({'epoch': epoch,
                        'model_state_dict': gen.state_dict(),
                        'optimizer_state_dict': model.optimizerG.state_dict(),
                        'loss': model.lossG.item()},
                       os.path.join(save_dir, f"gen_{save_name}_{epoch}_sd.pt"))
            torch.save({'epoch': epoch,
                        'model_state_dict': disc.state_dict(),
                        'optimizer_state_dict': model.optimizerD.state_dict(),
                        'loss': model.lossD.item()},
                       os.path.join(save_dir, f"disc_{save_name}_{epoch}_sd.pt"))
            print(f"Models <{save_name}> saved to {save_dir}")
        else:
            if verbose:
                print("Models not saved.")


def trainPairedWindows(epochs, dataloader, model, save_every_epoch=False, save_dir=None, save_name=None, verbose=True):
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
            if verbose:
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
            if verbose:
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
        # Save models
        if save_every_epoch and save_dir is not None and save_name is not None:
            torch.save({'epoch': epoch,
                        'model_state_dict': gen.state_dict(),
                        'optimizer_state_dict': model.optimizerG.state_dict(),
                        'loss': model.lossG.item()},
                       os.path.join(save_dir, f"gen_{save_name}_{epoch}_sd.pt"))
            torch.save({'epoch': epoch,
                        'model_state_dict': disc.state_dict(),
                        'optimizer_state_dict': model.optimizerD.state_dict(),
                        'loss': model.lossD.item()},
                       os.path.join(save_dir, f"disc_{save_name}_{epoch}_sd.pt"))
            print(f"Models <{save_name}> saved to {save_dir}")
        else:
            if verbose:
                print("Models not saved.")


def get_args():
    parser = argparse.ArgumentParser(description="Train neural network.")
    parser.add_argument('-r', "--root", type=str, default='../data',
                        help="Path to input data used in network")
    parser.add_argument('-m', "--model", type=str, default='window',
                        help="Type of model to train. Must be one of 'window' or 'base'.")
    parser.add_argument('-N', "--size", type=int, default=256,
                        help="Size of image generated (cubic). Also height of sinogram")
    parser.add_argument('-s', "--shifts", type=int, default=5,
                        help="Number of vertical shifts applied to each sample in data generation")
    parser.add_argument('-w', "--window-width", type=int, default=25,
                        help="Width of windows that sinograms are split into")
    parser.add_argument('-e', "--epochs", type=int, default=1,
                        help="Number of epochs (i.e. total passes through the dataset)")
    parser.add_argument('-l', "--learning-rate", type=float, default=0.0002,
                        help="Learning rate of the network")
    parser.add_argument('-b', "--betas", type=float, default=[0.5, 0.999], nargs=2,
                        help="Values of the beta parameters used in the Adam optimizer")
    parser.add_argument('-B', "--batch-size", type=int, default=32,
                        help="Batch size used for loading data and for minibatches for Adam optimizer")
    parser.add_argument('-d', "--save-dir", type=str, default=None,
                        help="Directory to save models to once training has finished.")
    parser.add_argument("--save-every-epoch", action="store_true", help="Save model every epoch")
    parser.add_argument('-v', "--verbose", action="store_true", help="Print some extra information when running")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataroot = args.root
    model_save_dir = args.save_dir
    size = args.size
    windowWidth = args.window_width
    num_windows = (int(np.sqrt(2) * size) // windowWidth + 1)

    epochs = args.epochs
    learning_rate = args.learning_rate
    betas = args.betas
    num_shifts = args.shifts
    batch_size = args.batch_size

    save_every_epoch = args.save_every_epoch
    verbose = args.verbose

    if args.model == 'window':
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
        trainPairedWindows(epochs, dataloader, model, save_every_epoch=save_every_epoch, save_dir=model_save_dir,
                           save_name=args.model, verbose=verbose)
    elif args.model == 'base':
        # Create dataset and dataloader
        dataset = BaseDataset(root=dataroot, mode='train', tvt=(3, 1, 1), size=size, shifts=num_shifts,
                              transform=transforms.ToTensor())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Create models
        disc = SinoDiscriminator()
        disc.apply(init_weights)
        gen = SinoUNet()
        gen.apply(init_weights)
        model = BaseGAN(gen, disc, mode='train', learning_rate=learning_rate, betas=betas)
        # Train
        trainBase(epochs, dataloader, model, save_every_epoch=save_every_epoch, save_dir=model_save_dir,
                  save_name=args.model, verbose=verbose)
    else:
        raise ValueError(f"Argument '--model' should be one of 'window', 'base'. Instead got {args.model}'")
