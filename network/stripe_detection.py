import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tomobar.methodsDIR import RecToolsDIR

from datasets import BaseDataset, WindowDataset, PairedWindowDataset
from simulator.data_io import loadTiff, rescale
from visualizers import MetricVisualizer
import metrics


def compareDifferentWidths(metric, num_widths, lo, hi, size=256, num_shifts=5):
    dataroot = os.path.join(os.pardir, 'data')
    subplot_size = (5, num_widths + 2)
    vis = MetricVisualizer(size, subplot_size)

    for x in range(5):
        dataset = BaseDataset(root=dataroot, mode='train', tvt=(1, 0, 0), size=size, shifts=num_shifts)
        item = random.randint(0, len(dataset))
        print(len(dataset))
        clean_full, *shifts_full = dataset[item]
        centre_full = shifts_full[num_shifts // 2]
        vis.plot_recon(clean_full, f"Clean {item}")
        vis.plot_recon(centre_full, f"Centre {item}")

        for widthRatio in np.linspace(lo, hi, num_widths):
            width = int(widthRatio * np.sqrt(2) * size)
            # Define dataset
            dataset = PairedWindowDataset(root=dataroot, mode='train', tvt=(1, 0, 0), windowWidth=width,
                                          stripeMetric=metric, size=size, shifts=num_shifts)

            # 1 - GET SINOGRAM OF BOB
            _, _, bob_full = dataset.getFullSinograms(item)
            bob_full = bob_full.squeeze()

            # 3 - RECONSTRUCT & PLOT
            vis.plot_recon(bob_full, f"B-o-B {width}")
    plt.show()


def compareDifferentMetrics(metric_list, width, size=256, num_shifts=5):
    dataroot = os.path.join(os.pardir, 'data')
    length = len(metric_list) + 2
    vis = MetricVisualizer(size, (5, len(metric_list) + 2))

    for x in range(5):
        dataset = BaseDataset(root=dataroot, mode='train', tvt=(1, 0, 0), size=size, shifts=num_shifts)
        item = random.randint(0, len(dataset))
        clean_full, *shifts_full = dataset[item]
        centre_full = shifts_full[num_shifts // 2]

        vis.plot_recon(clean_full, f"Clean {item}")
        vis.plot_recon(centre_full, f"Centre {item}")

        for metric in metric_list:
            dataset = PairedWindowDataset(root=dataroot, mode='train', tvt=(1, 0, 0), windowWidth=width,
                                          stripeMetric=metric, size=size, shifts=num_shifts)
            # GET SINOGRAM OF BOB
            _, _, bob_full = dataset.getFullSinograms(item)
            bob_full = bob_full.squeeze()

            # RECONSTRUCT & PLOT
            vis.plot_recon(bob_full, title=metric.__name__)
    plt.show()


def test_metric(stripe_metrics, test_scores, width, size=256, num_shifts=5, visual=True):
    dataroot = os.path.join(os.pardir, 'data')
    if type(stripe_metrics) != list:
        stripe_metrics = [stripe_metrics]
    if type(test_scores) != list:
        test_scores = [test_scores]
    vis = MetricVisualizer(size, (1, 1))
    rows = []
    for test_score in test_scores:
        total_scores = []
        for stripe_metric in stripe_metrics:
            # create dataset
            dataset = PairedWindowDataset(root=dataroot, mode='train', tvt=(1, 0, 0), windowWidth=width,
                                          stripeMetric=stripe_metric, size=size, shifts=num_shifts)
            # loop through each item in dataset, calculate dice coefficient between clean & bob
            scores = []
            for i in range(len(dataset) // 10):
                clean, _, bob = dataset.getFullSinograms(i)
                score = test_score(clean, bob)
                scores.append(score)
            # calculate mean of all dice coefficients
            mean_score = np.mean(scores)
            print(f"Stripe Metric: {stripe_metric.__name__}, Score Metric: {test_score.__name__}, Mean: {mean_score}")
            total_scores.append(scores)
        if visual:
            vis.plot_box(total_scores, titles=[stripe_metric.__name__ for stripe_metric in stripe_metrics])
            plt.show()

        rows.append([np.mean(s) for s in total_scores])
    saveCSV = input("Save data? (y/[n]): ")
    if saveCSV == 'y':
        with open('scores.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print("Data saved to 'scores.csv'")
    else:
        print("Data not saved.")


if __name__ == '__main__':
    # compareDifferentWidths(metrics.gradient_sum_tv, 10, 0.01, 0.2)
    # compareDifferentMetrics(metrics.all_metrics, width=18)
    test_metric(metrics.gradient_sum_tv, metrics.test_metrics, width=18)
