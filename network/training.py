import os
import argparse
import warnings
import numpy as np
from datetime import datetime

from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from .models import BaseGAN, WindowGAN, MaskedGAN, init_weights
from .models.discriminators import *
from .models.generators import *
from .visualizers import BaseGANVisualizer, PairedWindowGANVisualizer, \
    MaskedVisualizer
from .patch_visualizer import PatchVisualizer
from .datasets import PairedWindowDataset, BaseDataset, PairedFullDataset, \
    MaskedDataset, RandomSubset
from utils.misc import Rescale
import wandb



def saveModel(model, epoch, save_dir, save_name):
    """Save a model to disk as a .tar file.
    Saves the following parameters: epoch, model state, optimizer state & loss.
    Parameters:
        model : BaseGAN
            Model to save. Must be a GAN containing both generator and
            discriminator.
        epoch : int
            Last training epoch the model reached.
        save_dir : str
            Directory where model will be saved.
        save_name : str
            Name model will be saved with. Will be appended by `epoch`.
            i.e. <save_name>_<epoch>.tar
    """
    if save_dir is None or save_name is None:
        raise ValueError("If saving a model, both save directory and save "
                         "name should be passed as arguments.")
    torch.save({'epoch': epoch,
                'gen_state_dict': model.gen.state_dict(),
                'gen_optimizer_state_dict': model.optimizerG.state_dict(),
                'gen_loss': model.lossG,
                'disc_state_dict': model.disc.state_dict(),
                'disc_optimizer_state_dict': model.optimizerD.state_dict(),
                'disc_loss': model.lossD},
               os.path.join(save_dir, f"{save_name}_{epoch}.tar"))


def createModelParams(model, path, device):
    """Initialise parameters for a model.
    Supports either training a brand new model from scratch,
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
        print(f"Training new model from scratch.")
        model.gen.apply(init_weights)
        model.disc.apply(init_weights)
        return 0
    else:
        print(f"Loading model from '{path}'")
        checkpoint = torch.load(path, map_location=device)
        model.gen.load_state_dict(checkpoint['gen_state_dict'])
        model.optimizerG.load_state_dict(
            checkpoint['gen_optimizer_state_dict']
        )
        model.lossG = checkpoint['gen_loss']
        model.disc.load_state_dict(checkpoint['disc_state_dict'])
        model.optimizerD.load_state_dict(
            checkpoint['disc_optimizer_state_dict']
        )
        model.lossD = checkpoint['disc_loss']
        return checkpoint['epoch']


def getTrainingData(dataset, data):
    """Deconstruct output from a Dataset class into a pair of input & target.
    Parameters:
            dataset : torch.utils.data.Dataset
                Dataset class from which the data has come.
            data : Tuple[torch.Tensor]
                Tuple containing raw output from the dataset.
    """
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


def train(model, dataloader, epochs, vis, save_every_epoch=False,
          save_name=None, save_dir=None, start_epoch=0, verbose=True,
          force=False):
    """Train a model.
    Parameters:
        model : torch.nn.Module
            Model to train
        dataloader : torch.utils.data.DataLoader
            DataLoader instance from which to load data
        epochs : int
            Number of epochs for which to train the model
        vis : object
            Visualizer to plot results of training.
        save_every_epoch : bool
            Whether the model should be saved to disk after the completion of
            each epoch. Default is False.
            If True, `save_name` and `save_dir` must also be specified.
        save_name : str
            Name with which to save model. Default is None.
        save_dir : str
            Directory to save model to. Default is None.
        start_epoch : int
            If a model has been pre-trained, and you want to continue training
            where it left off, this can be set to the last training epoch.
            Default is 0.
        verbose : bool
            Print out some extra information.
        force : bool
            Whether execution should continue without waiting for plots to be
            closed. Default is False.
    """
    if isinstance(dataloader.dataset, Subset):
        dataset = dataloader.dataset.dataset
    else:
        dataset = dataloader.dataset
    epochs += start_epoch
    num_batches = len(dataloader)
    start_time = datetime.now()
    print(f"Training has begun. "
          f"Epochs: {epochs}, "
          f"Batches: {num_batches}, "
          f"Steps/batch: {dataloader.batch_size}")
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
                print(f"\tEpoch [{epoch + 1}/{epochs}], "
                      f"Batch [{i + 1}/{num_batches}], "
                      f"Loss_D: {model.lossD.item():2.5f}, "
                      f"Loss_G: {model.lossG.item():2.5f}, "
                      f"D(x): {model.D_x:.5f}, "
                      f"D(G(x)): {model.D_G_x1:.5f} / {model.D_G_x2:.5f}")
            
            # Log metrics 
            wandb.log({
                'Loss_D': model.lossD.item(),
                'Loss_G': model.lossG.item(), 
                'D(x)'  : model.D_x,
                'D_G_X1': model.D_G_x1,
                'D_G_X2': model.D_G_x2
            })


        # At the end of every epoch, run through validate dataset
        print(f"Epoch [{epoch + 1}/{epochs}]: "
              f"Training finished. Validating model...")
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
                print(f"\tEpoch [{epoch + 1}/{epochs}], "
                      f"Batch [{i + 1}/{num_batches}], "
                      f"Loss_D: {model.lossD.item():2.5f}, "
                      f"Loss_G: {model.lossG.item():2.5f}, "
                      f"D(x): {model.D_x:.5f}, "
                      f"D(G(x)): {model.D_G_x1:.5f} / {model.D_G_x2:.5f}")
            # Collate validation losses
            validation_lossesG[i] = model.lossG.item()
            validation_lossesD[i] = model.lossD.item()
        # Step scheduler with median of all validation losses
        # (avoids outliers at start of validation)
        model.schedulerG.step(np.median(validation_lossesG))
        model.schedulerD.step(np.median(validation_lossesD))
        print(f"Epoch [{epoch + 1}/{epochs}]: Validation finished.")

        # At the end of every epoch, save model state
        if save_every_epoch and save_dir is not None and save_name is not None:
            saveModel(model, epoch, save_dir, save_name)
            print(f"Epoch [{epoch+1}/{epochs}]: "
                  f"Model '{save_name}_{epoch}' saved to '{save_dir}'")
        else:
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}]: Model not saved.")
    # Once training has finished, plot some data and save model state
    finish_time = datetime.now()
    print(f"Total Training time: {finish_time - start_time}")
    # try:
    #
    #     fig_syn = vis.plot_one()
    #     wandb.log({"Synthetic Stripes": fig_syn})
    #     if verbose:
    #         print('logged Synthetic stripes')
    #
    #     fig_rf = vis.plot_real_vs_fake_batch()
    #     wandb.log({"Last batch plot": fig_rf})
    #     if verbose:
    #         print('logged last batch')
    #
    #     fig_rf_recon = vis.plot_real_vs_fake_recon()
    #     wandb.log({"Last Batch Recon": fig_rf_recon})
    #     if verbose:
    #         print('logged Last batch recon')
    #
    # except OSError as e:
    #     # if plotting causes OoM, don't crash so model can still be saved
    #     print(e)
    # Save models if user desires and save_every_epoch is False
    if not save_every_epoch and (force
                                 or input("Save model? (y/[n]): ") == 'y'):
        saveModel(model, epochs, save_dir, save_name)
        print(f"Training finished: "
              f"Model '{save_name}_{epochs}' saved to '{save_dir}'")
    else:
        print("Training finished: Model not saved.")


def get_args():
    parser = argparse.ArgumentParser(description="Train neural network.")
    parser.add_argument('-r', "--root", type=str, default='./data',
                        help="Path to input data used in network.")
    parser.add_argument('-m', "--model", type=str, default='base',
                        help="Type of model to train. Must be one of ['base', "
                             "'mask', 'simple', 'patch', 'window', 'full'].")
    parser.add_argument('-N', "--size", type=int, default=256,
                        help="Number of sinograms per sample.")
    parser.add_argument('-s', "--shifts", type=int, default=1,
                        help="Number of shifts per sample.")
    parser.add_argument("--tvt", type=int, default=[3, 1, 1], nargs=3,
                        help="Train/Validate/Test split, entered as a ratio.")
    parser.add_argument('-B', "--batch-size", type=int, default=16,
                        help="Batch size used for loading data.")
    parser.add_argument('-e', "--epochs", type=int, default=1,
                        help="Number of epochs "
                             "(i.e. total passes through the dataset).")
    parser.add_argument('-l', "--learning-rate", type=float, default=0.0002,
                        help="Learning rate of the network.")
    parser.add_argument('-b', "--betas", type=float, default=[0.5, 0.999],
                        nargs=2,
                        help="Values of the beta parameters used in the Adam "
                             "optimizer.")
    parser.add_argument('--lambda', type=float, default=100, dest='lambdal1',
                        help="Weight by which L1 loss in the generator is "
                             "multiplied.")
    parser.add_argument("--lsgan", action="store_true",
                        help="Train an LSGAN, rather than a normal GAN.")
    parser.add_argument('-d', "--save-dir", type=str, default=None,
                        help="Directory to save models to once training has "
                             "finished.")
    parser.add_argument('-f', "--model-file", type=str, default=None,
                        help="Path to a pre-trained model.")
    parser.add_argument("--subset", type=int, default=None,
                        help="Train using a subset of the full dataset.")
    parser.add_argument("--force", action="store_true",
                        help="Force the script to keep running, regardless of "
                             "any waits/pauses.")
    parser.add_argument("--save-every-epoch", action="store_true",
                        help="Save model at the end of every epoch.")
    parser.add_argument('-v', "--verbose", action="store_true",
                        help="Print some extra information when running.")
    parser.add_argument('-w', "--window-width", type=int, default=25,
                        help="Width of windows that sinograms are split into.")
    parser.add_argument('-n', '--name', type=str, default=datetime.now().strftime("%d/%m/%Y %H:%M"),
                        help="Log run name")
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

    lsgan = args.lsgan
    save_every_epoch = args.save_every_epoch
    force = args.force
    verbose = args.verbose

    # mean: 0.1780845671892166, std: 0.02912825345993042
    transform = transforms.Compose([
        transforms.ToTensor(),
        Rescale(a=-1, b=1, imin=0, imax=1)
    ])

    wandb.init(project='NoStripesNet',
        entity='nostripesnet',
        name=args.name
    )

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    disc = BaseDiscriminator()
    gen = BaseUNet()
    if args.model == 'base':
        # Create dataset
        dataset = BaseDataset(root=dataroot, mode='train', tvt=tvt, size=size,
                              shifts=num_shifts, transform=transform)
        model = BaseGAN(gen, disc, mode='train', learning_rate=learning_rate,
                        betas=betas, lambdaL1=lambdal1, lsgan=lsgan,
                        device=device)
        vis = BaseGANVisualizer(model, dataset, size, not force)
    elif args.model == 'window':
        # Create dataset
        dataset = PairedWindowDataset(root=dataroot, mode='train', tvt=tvt,
                                      size=size, shifts=num_shifts,
                                      windowWidth=windowWidth,
                                      transform=transform)
        # Create models
        disc = WindowDiscriminator()
        gen = WindowUNet()
        model = WindowGAN(windowWidth, gen, disc, mode='train',
                          learning_rate=learning_rate, betas=betas,
                          lambdaL1=lambdal1, lsgan=lsgan, device=device)
        vis = PairedWindowGANVisualizer(model, dataset, size, not force)
    elif args.model == 'full':
        # Create dataset
        dataset = PairedFullDataset(root=dataroot, mode='train', tvt=tvt,
                                    size=size, shifts=num_shifts,
                                    windowWidth=windowWidth,
                                    transform=transform)
        model = BaseGAN(gen, disc, mode='train', learning_rate=learning_rate,
                        betas=betas, lambdaL1=lambdal1, lsgan=lsgan,
                        device=device)
        vis = BaseGANVisualizer(model, dataset, size, not force)
    elif args.model == 'mask' or args.model == 'simple':
        # Create dataset
        dataset = MaskedDataset(root=dataroot, mode='train', tvt=tvt,
                                size=size, shifts=num_shifts,
                                transform=transform,
                                simple=args.model=='simple')
        model = MaskedGAN(gen, disc, mode='train', learning_rate=learning_rate,
                          betas=betas, lambdaL1=lambdal1, lsgan=lsgan,
                          device=device)
        vis = MaskedVisualizer(model, dataset, size, not force)
    elif args.model == 'patch':
        dataset = MaskedDataset(root=dataroot, mode='train', tvt=tvt,
                                size=size, shifts=num_shifts,
                                transform=transform, simple=True)
        disc = PatchDiscriminator()
        gen = PatchUNet()
        model = MaskedGAN(gen, disc, mode='train', learning_rate=learning_rate,
                          betas=betas, lambdaL1=lambdal1, lsgan=lsgan,
                          device=device)
        vis = PatchVisualizer(dataroot, model, block=not force)
    else:
        raise ValueError(f"Argument '--model' should be one of ['base', 'mask'"
                         f", 'simple', 'patch', 'window', 'full']. "
                         f"Instead got '{args.model}'")

    # Train
    start_epoch = createModelParams(model, model_file, device)
    if save_every_epoch and model_save_dir is None:
        warnings.warn("Argument --save-every-epoch is True, "
                      "but a save directory has not been specified. "
                      "Models will not be saved at all!", RuntimeWarning)
    if sbst_size is not None:
        dataset = RandomSubset(dataset, sbst_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    train(model, dataloader, epochs, vis, save_every_epoch=save_every_epoch,
          save_dir=model_save_dir, save_name=args.model,
          start_epoch=start_epoch, verbose=verbose, force=force)
