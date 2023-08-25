import argparse
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from .training import getTrainingData
from .models import BaseGAN, MaskedGAN, init_weights
from .models.generators import BaseUNet, PatchUNet
from .models.discriminators import BaseDiscriminator, PatchDiscriminator
from .datasets import BaseDataset, MaskedDataset
from utils.metrics import apply_metrics, test_metrics
from utils.misc import Rescale
from .visualizers import BaseGANVisualizer, MaskedVisualizer
from .patch_visualizer import PatchVisualizer
import wandb


def createModelParams(model, path, device):
    """Initialise parameters for a model.
    Supports either testing a brand new model from scratch,
    or loading a model from disk and using the parameters from that.
    Parameters:
            model : BaseGAN
                Model in which to initialise parameters
            path : str
                Path to a pre-trained model on disk. If None, parameters will
                be initialised according to the `init_weights` function from
                ./models/gans.py
            device : torch.device
                Device to load parameters on to.
    """
    if path is None:
        print("No model directory specified, so new model will be created. "
              "This model will not have been trained.")
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


def accuracy(predictions, labels):
    """Calculate accuracy given predictions and labels.
    Parameters:
        predictions : torch.Tensor
            Predictions from a Discriminator. Assumed to be in range [0, 1].
        labels : torch.Tensor
            Labels for `predictions`.
    """
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    return predictions[predictions == labels].nelement() / labels.nelement()


def test(model, dataloader, metrics, vis, display_each_batch=False,
         verbose=True, visual_only=False, run=None):
    """Train a model.
    Parameters:
        model : torch.nn.Module
            Model to test
        dataloader : torch.utils.data.DataLoader
            DataLoader instance from which to load data
        metrics : List[function]
            List of test metrics to apply to data
        vis : object
            Visualizer to plot results of training.
        display_each_batch : bool
            Whether each batch should be plotted. Default is False.
        verbose : bool
            Print out some extra information.
        visual_only : bool
            If True, test metrics will not be calculated;
            Only one batch will be ran, and then immediately plotted.
            Useful if you just want to do a visual analysis of performance,
            and not wait for the entire testing process to complete.
            Default is False.
        run : wandb.Run
            Weights & Biases Run object to log metrics to.
    """
    if isinstance(dataloader.dataset, Subset):
        dataset = dataloader.dataset.dataset
    else:
        dataset = dataloader.dataset
    overall_mean_scores = {metric.__name__: [] for metric in metrics}
    overall_accuracies = {'total': 0, 'fake': 0, 'real': 0}
    start_time = datetime.now()
    print(f"Testing has begun. "
          f"Batches: {len(dataloader)}, "
          f"Steps/batch: {dataloader.batch_size}")
    for i, data in enumerate(dataloader):
        if verbose and not visual_only:
            print(f"\tBatch [{i + 1}/{len(dataloader)}]")
        inpt, target = getTrainingData(dataset, data)
        # Pre-process data
        model.preprocess(inpt, target)
        # Run forward and backward passes
        model.run_passes()

        if visual_only:
            break

        # Calculate model evaluation metrics
        metric_scores = apply_metrics(model.realB, model.fakeB, metrics)
        for key in metric_scores:
            overall_mean_scores[key].append(metric_scores[key])
        # Calculate discriminator accuracies
        # fake
        disc_in = torch.cat((model.realA, model.fakeB), dim=1)
        disc_out = model.disc(disc_in)
        fake_accuracy = accuracy(torch.sigmoid(disc_out),
                                 torch.zeros_like(disc_out))
        # real
        disc_in = torch.cat((model.realA, model.realB), dim=1)
        disc_out = model.disc(disc_in)
        real_accuracy = accuracy(torch.sigmoid(disc_out),
                                 torch.ones_like(disc_out))
        total_accuracy = (fake_accuracy + real_accuracy) * 0.5
        overall_accuracies['total'] += total_accuracy
        overall_accuracies['fake'] += fake_accuracy
        overall_accuracies['real'] += real_accuracy

        if display_each_batch:
            # Print test statistics for each batch
            for key in overall_mean_scores:
                print(f"\t\t{key: <23}: {np.mean(overall_mean_scores[key])}")
            # Print discriminator accuracy for each batch
            print(f"\t\tDiscriminator Accuracy - Total: {total_accuracy}, "
                  f"Fake: {fake_accuracy}, Real: {real_accuracy}")
            # Plot images each batch
            if verbose:
                print(f"\t\tPlotting batch [{i + 1}/{len(dataloader)}]...")
            vis.plot_real_vs_fake_batch()
    print("Testing completed.")
    if not visual_only:
        print("Total mean scores for all batches:")
        for key in overall_mean_scores:
            print(f"\t\t{key: <23}: {np.mean(overall_mean_scores[key])}")
            if run:
                run.summary[key] = np.mean(overall_mean_scores[key])
        print("Overall Accuracy for discriminator:")
        print(f"\tTotal : {overall_accuracies['total'] / len(dataloader)}"
              f"\n\tFake  : {overall_accuracies['fake'] / len(dataloader)}" 
              f"\n\tReal  : {overall_accuracies['real'] / len(dataloader)}")
        if run:
            run.summary[f"Total"] = overall_accuracies['total'] / len(dataloader)
            run.summary[f"Fake"] = overall_accuracies['fake'] / len(dataloader)
            run.summary[f"Real"] = overall_accuracies['real'] / len(dataloader)
            run.summary.update()
            print("Mertics logged")
    finish_time = datetime.now()
    print(f"Total test time: {finish_time - start_time}")
    fig_syn = vis.plot_one()
    fig_syn.savefig('synth_stripes.png', dpi=200)

    if verbose:
        print("Plotting last batch...")
    fig_rf = vis.plot_real_vs_fake_batch()
    fig_rf.savefig('last_batch.png', dpi=200)

    if verbose:
        print("Reconstructing last batch...")
    fig_rf_recon = vis.plot_real_vs_fake_recon()
    fig_rf_recon.savefig('last_recon.png', dpi=200)


def get_args():
    parser = argparse.ArgumentParser(description="Test neural network.")
    parser.add_argument('-r', "--root", type=str, default='./data',
                        help="Path to input data used in network.")
    parser.add_argument('-m', "--model", type=str, default='base',
                        help="Type of model to train. Must be one of ['base', "
                             "'mask', 'simple', 'patch'].")
    parser.add_argument("--tvt", type=int, default=[3, 1, 1], nargs=3,
                        help="Train/Validate/Test split, entered as a ratio.")
    parser.add_argument('-B', "--batch-size", type=int, default=16,
                        help="Batch size used for loading data.")
    parser.add_argument('-f', "--model-file", type=str, default=None,
                        help="Path to load model to test from.")
    parser.add_argument('-M', "--metrics", type=str, default='all', nargs='*',
                        help="Metrics used to evaluate model.")
    parser.add_argument("--display-each-batch", action="store_true",
                        help="Plot each batch of generated images during "
                             "testing.")
    parser.add_argument("--visual-only", action="store_true",
                        help="Don't calculate metric scores; only display a "
                             "batch of images.")
    parser.add_argument('-v', "--verbose", action="store_true",
                        help="Print some extra information when running.")
    parser.add_argument('-n', '--name', type=str, default=None,
                        help="W&b training log run name")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataroot = args.root
    model_name = args.model
    model_file = args.model_file

    batch_size = args.batch_size
    tvt = args.tvt

    api = wandb.Api()
    runs = api.runs("nostripegan/NoStripesGAN")

    run = None
    for rns in runs:
        if rns.name == args.name:
            run = api.run(f"/nostripegan/NoStripesGAN/{rns.id}")

    if args.metrics == 'all':
        ms = test_metrics
    else:
        try:
            ms = [test_metrics[int(m)] for m in args.metrics]
        except (IndexError, ValueError):
            raise ValueError(
                f"Argument --metrics should be either "
                f"string 'all' or integers in range [0, {len(test_metrics)}). "
                f"Instead got {args.metrics}."
            )

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
        dataset = BaseDataset(root=dataroot, mode='test', tvt=tvt,
                              transform=transform)
        model = BaseGAN(gen, disc, mode='test', device=device)
        vis = BaseGANVisualizer(model, dataset, True)
    elif model_name == 'mask' or model_name == 'simple':
        dataset = MaskedDataset(root=dataroot, mode='test', tvt=tvt,
                                transform=transform,
                                simple=model_name=='simple')
        model = MaskedGAN(gen, disc, mode='test', device=device)
        vis = MaskedVisualizer(model, dataset, True)
    elif args.model == 'patch':
        dataset = MaskedDataset(root=dataroot, mode='test', tvt=tvt,
                                transform=transform, simple=True)
        disc = PatchDiscriminator()
        gen = PatchUNet()
        model = MaskedGAN(gen, disc, mode='test', device=device)
        vis = PatchVisualizer(dataroot, model, block=True)
    else:
        raise ValueError(f"Argument '--model' should be one of ['base', 'mask'"
                         f", 'simple', 'patch']. "
                         f"Instead got '{args.model}'")

    # Test
    createModelParams(model, model_file, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=20)
    test(model, dataloader, ms, vis, display_each_batch=display_each_batch,
         verbose=verbose, visual_only=visual, run=run)
