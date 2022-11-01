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
from datasets import PairedWindowDataset, BaseDataset
from visualizers import BaseGANVisualizer, PairedWindowGANVisualizer
from metrics import *


def createGenParams(gen, path):
    if path is None:
        print("No model directory specified, so new model will be created. This model will not have been trained.")
        cont = input("Are you sure you want to continue? ([y]/n): ")
        if cont == 'n':
            print("Quitting testing.")
            exit(0)
        gen.apply(init_weights)
    else:
        checkpoint = torch.load(path)
        gen.load_state_dict(checkpoint['gen_state_dict'])


def batch_metric(metric, data1_batch, data2_batch):
    # scores = []
    # for data1, data2 in zip(data1_batch, data2_batch):
    #     m = metric(data1.squeeze().numpy(), data2.squeeze().numpy())
    #     scores.append(m)
    # r = np.mean(scores)
    # return r
    return np.mean([metric(data1.squeeze().numpy(), data2.squeeze().numpy()) for data1, data2 in zip(data1_batch, data2_batch)])


def testBase(model, dataloader, metrics, display_each_batch=False):
    vis = BaseGANVisualizer(model, dataloader, dataloader.dataset.size)
    overall_mean_scores = {metric.__name__: [] for metric in metrics}
    for i, (clean, *shifts) in enumerate(dataloader):
        print(f"Batch [{i+1}/{len(dataloader)}]")
        centre = shifts[num_shifts // 2]
        # Pre-process data
        model.preprocess(centre, clean)
        # Run forward and backward passes
        model.run_passes()

        metric_scores = {metric.__name__: batch_metric(metric, model.realB, model.fakeB) for metric in metrics}
        [overall_mean_scores[key].append(metric_scores[key]) for key in metric_scores]

        if display_each_batch:
            # Print test statistics for each batch
            [print(f"{key}: {metric_scores[key]}", end='\t') for key in metric_scores]
            # Plot images each batch
            print("\nPlotting batch...")
            vis.plot_real_vs_fake_batch()

    print("Total mean scores for all batches:")
    [print(f"{key}: {np.mean(overall_mean_scores[key])}", end='\t') for key in overall_mean_scores]
    print("\nPlotting last batch...")
    vis.plot_real_vs_fake_batch()
    vis.plot_real_vs_fake_recon()


def testPairedWindows(model, dataloader, metrics, display_each_batch=False):
    vis = PairedWindowGANVisualizer(model, dataloader, dataloader.dataset.size)
    overall_mean_scores = {metric.__name__: [] for metric in metrics}
    for i, (clean, centre, _) in enumerate(dataloader):
        print(f"Batch [{i+1}/{len(dataloader)}]")
        # Pre-process data
        model.preprocess(centre, clean)
        # Run forward and backward passes
        model.run_passes()

        model.realB = dataloader.dataset.combineWindows(model.realBs)
        model.fakeB = dataloader.dataset.combineWindows(model.fakeBs)

        metric_scores = {metric.__name__: batch_metric(metric, model.realB, model.fakeB) for metric in metrics}
        [overall_mean_scores[key].append(metric_scores[key]) for key in metric_scores]

        if display_each_batch:
            # Print test statistics for each batch
            [print(f"{key}: {metric_scores[key]}", end='\t') for key in metric_scores]
            print()
            # Plot images each batch
            print("\nPlotting batch...")
            vis.plot_real_vs_fake_recon()

    print("Total mean scores for all batches:")
    [print(f"{key}: {np.mean(overall_mean_scores[key])}", end='\t') for key in overall_mean_scores]
    print("\nPlotting last batch...")
    vis.plot_real_vs_fake_batch()
    vis.plot_real_vs_fake_recon()


def get_args():
    parser = argparse.ArgumentParser(description="Test neural network.")
    parser.add_argument('-r', "--root", type=str, default='../data',
                        help="Path to input data used in network")
    parser.add_argument('-m', "--model", type=str, default='window',
                        help="Type of model to test. Must be one of 'window' or 'base'")
    parser.add_argument('-f', "--model-file", type=str, default=None,
                        help="Path from which to load models for testing")
    parser.add_argument('-N', "--size", type=int, default=256,
                        help="Size of image generated (cubic). Also height of sinogram")
    parser.add_argument('-s', "--shifts", type=int, default=5,
                        help="Number of vertical shifts applied to each sample in data generation")
    parser.add_argument('-w', "--window-width", type=int, default=25,
                        help="Width of windows that sinograms are split into")
    parser.add_argument('-B', "--batch-size", type=int, default=32,
                        help="Batch size used for loading data  ")
    parser.add_argument('-M', "--metrics", type=str, default='all', nargs='*',
                        help="Metrics used to evaluate model.")
    parser.add_argument("--display-each-batch", action="store_true",
                        help="Plot each batch of generated images during testing")
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

    if model_name == 'base':
        # Create dataset and dataloader
        dataset = BaseDataset(root=dataroot, mode='test', tvt=(3, 1, 1), size=size, shifts=num_shifts,
                              transform=transforms.ToTensor())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Create models
        gen = SinoUNet()
        createGenParams(gen, model_file)
        model = BaseGAN(gen, mode='test')
        # Test
        testBase(model, dataloader, ms, display_each_batch=display_each_batch)
    elif model_name == 'window':
        # Create dataset and dataloader
        dataset = PairedWindowDataset(root=dataroot, mode='test', tvt=(3, 1, 1), size=size, shifts=num_shifts,
                                      windowWidth=windowWidth, transform=transforms.ToTensor())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Create models
        gen = PairedWindowUNet()
        createGenParams(gen, model_file)
        model = WindowGAN(windowWidth, gen, mode='test')
        # Test
        testPairedWindows(model, dataloader, ms, display_each_batch=display_each_batch)
    else:
        raise ValueError(f"Argument '--model' should be one of 'window', 'base'. Instead got '{model_name}'")
