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
from metrics import *


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
        print(f"Batch [{i}/{len(dataloader)}]")
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
            vis.plot_real_vs_fake_recon()

    print("Total mean scores for all batches:")
    [print(f"{key}: {np.mean(overall_mean_scores[key])}", end='\t') for key in overall_mean_scores]
    print("\nPlotting last batch...")
    vis.plot_real_vs_fake_recon()


def testPairedWindows(model, dataloader, metrics, display_each_batch=False):
    vis = PairedWindowGANVisualizer(model, dataloader, dataloader.dataset.size)
    overall_mean_scores = {metric.__name__: [] for metric in metrics}
    for i, (clean, centre, _) in enumerate(dataloader):
        print(f"Batch [{i}/{len(dataloader)}]")
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
    vis.plot_real_vs_fake_recon()


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
    dataset = PairedWindowDataset(root=dataroot, mode='test', tvt=(3, 1, 1), size=size, shifts=num_shifts,
                                  windowWidth=windowWidth, stripeMetric=gradient_sum_tv, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create models
    disc = PairedWindowDiscriminator()
    disc.apply(init_weights)
    gen = PairedWindowUNet()
    gen.apply(init_weights)
    model = WindowGAN(windowWidth, gen, disc, mode='test', learning_rate=learning_rate, betas=betas)

    # Train
    testPairedWindows(model, dataloader, test_metrics[:-2], display_each_batch=False)
