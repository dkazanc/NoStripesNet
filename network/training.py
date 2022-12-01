import os
import random
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.utils as utils

from models import BaseGAN, WindowGAN, MaskedGAN, init_weights
from models.discriminators import *
from models.generators import *
from visualizers import BaseGANVisualizer, PairedWindowGANVisualizer, MaskedVisualizer
from datasets import PairedWindowDataset, BaseDataset, PairedFullDataset, MaskedDataset, RandomSubset


# In the future this should be in NoStripesNet/simulator/data_io.py
# But that would cause a lot of relative import errors so for now it's staying here
def saveModel(model, epoch, save_dir, save_name):
    if save_dir is None or save_name is None:
        raise ValueError("If saving a model, both save directory and save name should be passed as arguments.")
    torch.save({'epoch': epoch,
                'gen_state_dict': model.gen.state_dict(),
                'gen_optimizer_state_dict': model.optimizerG.state_dict(),
                'gen_loss': model.lossG,
                'disc_state_dict': model.disc.state_dict(),
                'disc_optimizer_state_dict': model.optimizerD.state_dict(),
                'disc_loss': model.lossD},
               os.path.join(save_dir, f"{save_name}_{epoch}.tar"))


def createModelParams(model, path):
    if path is None:
        print(f"Training new model from scratch.")
        model.gen.apply(init_weights)
        model.disc.apply(init_weights)
        return 0
    else:
        print(f"Loading model from '{path}'")
        checkpoint = torch.load(path)
        model.gen.load_state_dict(checkpoint['gen_state_dict'])
        model.optimizerG.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        model.lossG = checkpoint['gen_loss']
        model.disc.load_state_dict(checkpoint['disc_state_dict'])
        model.optimizerD.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        model.lossD = checkpoint['disc_loss']
        return checkpoint['epoch']


def getTrainingData(dataset, data):
    if type(dataset) == BaseDataset:
        clean, *shifts = data
        centre = shifts[len(shifts) // 2]
        return centre, clean
    elif type(dataset) == PairedWindowDataset:
        clean, stripe, plain = data
        return stripe, clean
    elif type(dataset) == PairedFullDataset:
        clean, stripe, plain = data
        return stripe, clean
    elif type(dataset) == MaskedDataset:
        clean, stripe, mask = data
        inpt = torch.cat((stripe, mask), dim=-3)
        return inpt, clean
    else:
        raise ValueError(f"Dataset '{dataset}' not recognised.")


def getVisualizer(model, dataset, size):
    if type(dataset) == BaseDataset:
        return BaseGANVisualizer(model, dataset, size)
    elif type(dataset) == PairedWindowDataset:
        return PairedWindowGANVisualizer(model, dataset, size)
    elif type(dataset) == PairedFullDataset:
        return BaseGANVisualizer(model, dataset, size)
    elif type(dataset) == MaskedDataset:
        return MaskedVisualizer(model, dataset, size)
    else:
        raise ValueError(f"Dataset '{dataset}' not recognised.")


def train(model, dataloader, epochs, save_every_epoch=False, save_name=None, save_dir=None, start_epoch=0, verbose=True):
    if isinstance(dataloader.dataset, Subset):
        dataset = dataloader.dataset.dataset
    else:
        dataset = dataloader.dataset
    epochs += start_epoch
    num_batches = len(dataloader)
    vis = getVisualizer(model, dataset, dataset.size)
    print(f"Training has begun. Epochs: {epochs}, Batches: {num_batches}, Steps/batch: {dataloader.batch_size}")
    for epoch in range(start_epoch, epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]: Training model...")
        dataloader.dataset.setMode('train')
        model.setMode('train')
        num_batches = len(dataloader)
        for i, data in enumerate(dataloader):
            inpt, target = getTrainingData(dataset, data)
            # Pre-process data
            model.preprocess(inpt, target)
            # Run forward and backward passes
            model.run_passes()
            # Print out some useful info
            if verbose:
                print(f"\tEpoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{num_batches}], Loss_D: {model.lossD.item():2.5f}, "
                      f"Loss_G: {model.lossG.item():2.5f}, D(x): {model.D_x:.5f}, D(G(x)): {model.D_G_x1:.5f} / {model.D_G_x2:.5f}")

        # At the end of every epoch, run through validate dataset
        print(f"Epoch [{epoch + 1}/{epochs}]: Training finished. Validating model...")
        dataloader.dataset.setMode('validate')
        model.setMode('validate')
        num_batches = len(dataloader)
        validation_lossesG = torch.Tensor(num_batches)
        validation_lossesD = torch.Tensor(num_batches)
        for i, data in enumerate(dataloader):
            inpt, target = getTrainingData(dataset, data)
            # Pre-process data
            model.preprocess(inpt, target)
            # Run forward and backward passes
            model.run_passes()
            # Print out some useful info
            if verbose:
                print(f"\tEpoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{num_batches}], Loss_D: {model.lossD.item():2.5f}, "
                      f"Loss_G: {model.lossG.item():2.5f}, D(x): {model.D_x:.5f}, D(G(x)): {model.D_G_x1:.5f} / {model.D_G_x2:.5f}")
            # Collate validation losses
            validation_lossesG[i] = model.lossG.item()
            validation_lossesD[i] = model.lossD.item()
        # Step scheduler with median of all validation losses (avoids outliers at start of validation)
        model.schedulerG.step(np.median(validation_lossesG))
        model.schedulerD.step(np.median(validation_lossesD))
        print(f"Epoch [{epoch + 1}/{epochs}]: Validation finished.")

        # At the end of every epoch, save model state
        if save_every_epoch and save_dir is not None and save_name is not None:
            saveModel(model, epoch, save_dir, save_name)
            print(f"Epoch [{epoch+1}/{epochs}]: Model '{save_name}_{epoch}' saved to '{save_dir}'")
        else:
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}]: Model not saved.")
    # Once training has finished, plot some data and save model state
    vis.plot_losses()
    vis.plot_real_vs_fake_batch()
    vis.plot_real_vs_fake_recon()
    vis.plot_disc_predictions()
    # Save models if user desires and save_every_epoch is False
    if not save_every_epoch and input("Save model? (y/[n]): ") == 'y':
        saveModel(model, epochs, save_dir, save_name)
        print(f"Training finished: Model '{save_name}_{epochs}' saved to '{save_dir}'")
    else:
        print("Training finished: Model not saved.")


def get_args():
    parser = argparse.ArgumentParser(description="Train neural network.")
    parser.add_argument('-r', "--root", type=str, default='../data',
                        help="Path to input data used in network")
    parser.add_argument('-m', "--model", type=str, default='window',
                        help="Type of model to train. Must be one of ['window', 'base', 'full', 'mask', 'simple].")
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
    parser.add_argument('--lambda', type=int, default=100, dest='lambdal1',
                        help="Parameter by which L1 loss in the generator is multiplied")
    parser.add_argument('-d', "--save-dir", type=str, default=None,
                        help="Directory to save models to once training has finished.")
    parser.add_argument('-f', "--model-file", type=str, default=None,
                        help="Location of model on disk. If specified, this will override other hyperparameters and "
                             "load a pre-trained model from disk.")
    parser.add_argument("--tvt", type=int, default=[3, 1, 1], nargs=3,
                        help="Train/Validate/Test split, entered as a ratio")
    parser.add_argument("--subset", type=int, default=None,
                        help="Option to use a subset of the full dataset")
    parser.add_argument("--save-every-epoch", action="store_true", help="Save model every epoch")
    parser.add_argument('-v', "--verbose", action="store_true", help="Print some extra information when running")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataroot = args.root
    model_save_dir = args.save_dir
    model_file = args.model_file
    size = args.size
    windowWidth = args.window_width
    num_windows = (int(np.sqrt(2) * size) // windowWidth + 1)

    epochs = args.epochs
    learning_rate = args.learning_rate
    betas = args.betas
    lambdal1 = args.lambdal1
    num_shifts = args.shifts
    batch_size = args.batch_size
    tvt = args.tvt
    sbst_size = args.subset

    save_every_epoch = args.save_every_epoch
    verbose = args.verbose

    # mean: 0.1780845671892166, std: 0.02912825345993042
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    disc = BaseDiscriminator()
    gen = BaseUNet()
    if args.model == 'base':
        # Create dataset
        dataset = BaseDataset(root=dataroot, mode='train', tvt=tvt, size=size, shifts=num_shifts,
                              transform=transform)
        model = BaseGAN(gen, disc, mode='train', learning_rate=learning_rate, betas=betas, lambdaL1=lambdal1)
    elif args.model == 'window':
        # Create dataset
        dataset = PairedWindowDataset(root=dataroot, mode='train', tvt=tvt, size=size, shifts=num_shifts,
                                      windowWidth=windowWidth, transform=transform)
        # Create models
        disc = WindowDiscriminator()
        gen = WindowUNet()
        model = WindowGAN(windowWidth, gen, disc, mode='train', learning_rate=learning_rate, betas=betas, lambdaL1=lambdal1)
    elif args.model == 'full':
        # Create dataset
        dataset = PairedFullDataset(root=dataroot, mode='train', tvt=tvt, size=size, shifts=num_shifts,
                                    windowWidth=windowWidth, transform=transform)
        model = BaseGAN(gen, disc, mode='train', learning_rate=learning_rate, betas=betas, lambdaL1=lambdal1)
    elif args.model == 'mask' or args.model == 'simple':
        # Create dataset
        dataset = MaskedDataset(root=dataroot, mode='train', tvt=tvt, size=size, shifts=num_shifts, transform=transform,
                                simple=args.model=='simple')
        model = MaskedGAN(gen, disc, mode='train', learning_rate=learning_rate, betas=betas, lambdaL1=lambdal1)
    else:
        raise ValueError(f"Argument '--model' should be one of ['window', 'base', 'full', 'mask', 'simple]. "
                         f"Instead got '{args.model}'")

    # Train
    start_epoch = createModelParams(model, model_file)
    if save_every_epoch and model_save_dir is None:
        warnings.warn("Argument --save-every-epoch is True, but a save directory has not been specified. "
                      "Models will not be saved at all!", RuntimeWarning)
    if sbst_size is not None:
        dataset = RandomSubset(dataset, sbst_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train(model, dataloader, epochs, save_every_epoch=save_every_epoch, save_dir=model_save_dir, save_name=args.model,
          start_epoch=start_epoch, verbose=verbose)