import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.utils as utils

from training import getVisualizer, getTrainingData
from models import BaseGAN, WindowGAN, MaskedGAN, init_weights
from models.generators import *
from models.discriminators import *
from datasets import PairedWindowDataset, BaseDataset, PairedFullDataset, MaskedDataset
from visualizers import BaseGANVisualizer, PairedWindowGANVisualizer, MaskedVisualizer
from metrics import *


def createModelParams(model, path, device):
    if path is None:
        print("No model directory specified, so new model will be created. This model will not have been trained.")
        cont = input("Are you sure you want to continue? ([y]/n): ")
        if cont == 'n':
            print("Quitting testing.")
            exit(0)
        model.gen.apply(init_weights)
        model.disc.apply(init_weights)
    else:
        checkpoint = torch.load(path, map_location=device)
        model.gen.load_state_dict(checkpoint['gen_state_dict'])
        model.disc.load_state_dict(checkpoint['disc_state_dict'])


def batch_metric(metric, data1_batch, data2_batch):
    return np.mean([metric(data1.squeeze().detach().cpu().numpy(), data2.squeeze().detach().cpu().numpy())
                    for data1, data2 in zip(data1_batch, data2_batch)])


def accuracy(predictions, labels):
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    return predictions[predictions == labels].nelement() / labels.nelement()


def test(model, dataloader, metrics, display_each_batch=False, verbose=True, visual_only=False):
    if isinstance(dataloader.dataset, Subset):
        dataset = dataloader.dataset.dataset
    else:
        dataset = dataloader.dataset
    vis = getVisualizer(model, dataset, dataset.size)
    overall_mean_scores = {metric.__name__: [] for metric in metrics}
    overall_accuracies = {'total': 0, 'fake': 0, 'real': 0}
    start_time = datetime.now()
    print(f"Testing has begun. Batches: {len(dataloader)}, Steps/batch: {dataloader.batch_size}")
    for i, data in enumerate(dataloader):
        if verbose and not visual_only:
            print(f"\tBatch [{i + 1}/{len(dataloader)}]")
        inpt, target = getTrainingData(dataset, data)
        # Pre-process data
        model.preprocess(inpt, target)
        # Run forward and backward passes
        model.run_passes()

        if isinstance(model, WindowGAN):
            model.realA = dataset.combineWindows(model.realAs)
            model.realB = dataset.combineWindows(model.realBs)
            model.fakeB = dataset.combineWindows(model.fakeBs)

        if visual_only:
            break

        # Calculate model evaluation metrics
        metric_scores = apply_metrics(model.realB, model.fakeB, metrics)
        [overall_mean_scores[key].append(metric_scores[key]) for key in metric_scores]
        # Calculate discriminator accuracies
        # fake
        disc_in = torch.cat((model.realA, model.fakeB), dim=1)
        disc_out = model.disc(disc_in)
        fake_accuracy = accuracy(torch.sigmoid(disc_out), torch.zeros_like(disc_out))
        # real
        disc_in = torch.cat((model.realA, model.realB), dim=1)
        disc_out = model.disc(disc_in)
        real_accuracy = accuracy(torch.sigmoid(disc_out), torch.ones_like(disc_out))
        total_accuracy = (fake_accuracy + real_accuracy) * 0.5
        overall_accuracies['total'] += total_accuracy
        overall_accuracies['fake'] += fake_accuracy
        overall_accuracies['real'] += real_accuracy

        if display_each_batch:
            # Print test statistics for each batch
            [print(f"\t\t{key: <23}: {np.mean(overall_mean_scores[key])}") for key in overall_mean_scores]
            # Print discriminator accuracy for each batch
            print(f"\t\tDiscriminator Accuracy - Total: {total_accuracy}, Fake: {fake_accuracy}, Real: {real_accuracy}")
            # Plot images each batch
            if verbose:
                print(f"\t\tPlotting batch [{i + 1}/{len(dataloader)}]...")
            vis.plot_real_vs_fake_batch()
    print("Testing completed.")
    if not visual_only:
        print("Total mean scores for all batches:")
        [print(f"\t{key: <23}: {np.mean(overall_mean_scores[key])}") for key in overall_mean_scores]
        print("Overall Accuracy for discriminator:")
        print(f"\tTotal : {overall_accuracies['total'] / len(dataloader)}"
              f"\n\tFake  : {overall_accuracies['fake'] / len(dataloader)}" 
              f"\n\tReal  : {overall_accuracies['real'] / len(dataloader)}")
    finish_time = datetime.now()
    print(f"Total test time: {finish_time - start_time}")
    vis.plot_one()
    if verbose:
        print("Plotting last batch...")
    vis.plot_real_vs_fake_batch()
    if verbose:
        print("Reconstructing last batch...")
    vis.plot_real_vs_fake_recon()


def get_args():
    parser = argparse.ArgumentParser(description="Test neural network.")
    parser.add_argument('-r', "--root", type=str, default='../data',
                        help="Path to input data used in network")
    parser.add_argument('-m', "--model", type=str, default='base',
                        help="Type of model to test. Must be one of ['window', 'base', 'full', 'mask', 'simple].")
    parser.add_argument('-f', "--model-file", type=str, default=None,
                        help="Path from which to load models for testing")
    parser.add_argument('-N', "--size", type=int, default=256,
                        help="Size of image generated (cubic). Also height of sinogram")
    parser.add_argument('-s', "--shifts", type=int, default=5,
                        help="Number of vertical shifts applied to each sample in data generation")
    parser.add_argument('-w', "--window-width", type=int, default=25,
                        help="Width of windows that sinograms are split into")
    parser.add_argument('-B', "--batch-size", type=int, default=16,
                        help="Batch size used for loading data  ")
    parser.add_argument('-M', "--metrics", type=str, default='all', nargs='*',
                        help="Metrics used to evaluate model.")
    parser.add_argument("--display-each-batch", action="store_true",
                        help="Plot each batch of generated images during testing")
    parser.add_argument("--tvt", type=int, default=[3, 1, 1], nargs=3,
                        help="Train/Validate/Test split, entered as a ratio")
    parser.add_argument("--visual-only", action="store_true",
                        help="Don't calculate metric scores; only display batches of images")
    parser.add_argument('-v', "--verbose", action="store_true", help="Print some extra information when running")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataroot = args.root
    model_name = args.model
    model_file = args.model_file
    size = args.size
    windowWidth = args.window_width
    num_windows = (int(np.sqrt(2) * size) // windowWidth + 1)

    num_shifts = args.shifts
    batch_size = args.batch_size
    tvt = args.tvt

    if args.metrics == 'all':
        ms = test_metrics
    else:
        try:
            ms = [test_metrics[int(m)] for m in args.metrics]
        except (IndexError, ValueError):
            raise ValueError(f"Argument --metrics should be either string 'all' or integers in range [0, {len(test_metrics)}). "
                             f"Instead got {args.metrics}.")

    display_each_batch = args.display_each_batch
    verbose = args.verbose
    visual = args.visual_only

    transform = transforms.Compose([
        transforms.ToTensor(),
        Rescale(a=-1, b=1, imin=0, imax=1)
    ])

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    disc = BaseDiscriminator()
    gen = BaseUNet()
    if model_name == 'base':
        # Create dataset and dataloader
        dataset = BaseDataset(root=dataroot, mode='test', tvt=tvt, size=size, shifts=num_shifts,
                              transform=transform)
        model = BaseGAN(gen, disc, mode='test', device=device)
    elif model_name == 'window':
        # Create dataset and dataloader
        dataset = PairedWindowDataset(root=dataroot, mode='test', tvt=tvt, size=size, shifts=num_shifts,
                                      windowWidth=windowWidth, transform=transform)
        # Create models
        gen = WindowUNet()
        disc = WindowDiscriminator()
        model = WindowGAN(windowWidth, gen, disc, mode='test', device=device)
    elif model_name == 'full':
        dataset = PairedFullDataset(root=dataroot, mode='test', tvt=tvt, size=size, shifts=num_shifts,
                                    windowWidth=windowWidth, transform=transform)
        model = BaseGAN(gen, disc, mode='test', device=device)
    elif model_name == 'mask' or model_name == 'simple':
        dataset = MaskedDataset(root=dataroot, mode='test', tvt=tvt, size=size, shifts=num_shifts,
                                transform=transform, simple=model_name=='simple')
        model = MaskedGAN(gen, disc, mode='test', device=device)
    else:
        raise ValueError(f"Argument '--model' should be one of ['window', 'base', 'full', 'mask', 'simple]. "
                         f"Instead got '{model_name}'")

    # Test
    createModelParams(model, model_file, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test(model, dataloader, ms, display_each_batch=display_each_batch, verbose=verbose, visual_only=visual)
