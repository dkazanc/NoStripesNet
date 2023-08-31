import os
import argparse
from pathlib import Path
from datetime import datetime
from socket import gethostname
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
import torch.distributed as dist

from network.models.generators import PatchUNet
from network.models.discriminators import PatchDiscriminator
from network.models.gans import MaskedGAN
from simulator.generate_mask import append_npz
from utils.tomography import TomoH5
from utils.misc import toNumpy, Rescale


class H5Dataset(Dataset):
    def __init__(self, h5_file, mask_file, flat_file=None, transform=None):
        self.h5 = TomoH5(h5_file)
        if flat_file is not None:
            flat_h5 = TomoH5(flat_file)
            self.flats = flat_h5.get_flats()
            self.darks = flat_h5.get_darks()
        else:
            self.flats = self.h5.get_flats()
            self.darks = self.h5.get_darks()
        self.tomo = self.h5.get_normalized(np.s_[:], self.flats, self.darks, ncore=16)
        self.mask = np.load(mask_file)[f'm{h5_file.stem}']
        self.patch_size = (1801, 256)
        self.patch_h = self.patch_size[0]
        self.patch_w = self.patch_size[1]
        self.transform = transform

    def item_to_idx(self, item):
        sino_idx = item // 10
        patch_idx = item % 10
        return np.s_[:, sino_idx,
                     patch_idx * self.patch_w:(patch_idx + 1) * self.patch_w]

    def __len__(self):
        return self.h5.shape[1] * 10

    def __getitem__(self, item):
        idx = self.item_to_idx(item)
        stripe = self.tomo[idx]
        mask = self.mask[idx]
        if self.transform is not None:
            stripe = self.transform(stripe)
            mask = torch.Tensor(mask).unsqueeze(0)
        # assert stripe.shape == mask.shape == (1, 1801, 256)
        inpt = torch.cat((stripe, mask), dim=-3)
        # assert inpt.shape == (2, 1801, 256)
        target = torch.empty_like(stripe)  # target is not needed
        return inpt, target, item


def load_model(path, device=None, ddp=False):
    # Initialize Empty Model
    model = MaskedGAN(PatchUNet(), PatchDiscriminator(),
                      mode='test', device=device, ddp=ddp)
    # Load model state dict from disk
    checkpoint = torch.load(path, map_location=device)
    # Initialize Generator and Discriminator
    gsd = {}
    for k, v in checkpoint['gen_state_dict'].items():
        if ddp:
            gsd[f'module.{k}'] = v
        else:
            gsd[k] = v
    dsd = {}
    for k, v in checkpoint['disc_state_dict'].items():
        if ddp:
            dsd[f'module.{k}'] = v
        else:
            dsd[k] = v
    model.gen.load_state_dict(gsd)
    model.disc.load_state_dict(dsd)
    return model


def save_output(data, save_dir, file_name=None):
    ext = Path(save_dir).suffix
    if file_name is None:
        file_name = f'test_{datetime.now().strftime("%d-%m-%Y_%H:%M")}'
    if ext in ['.h5', '.hdf5', '.hdf', '.nxs']:
        with h5py.File(save_dir, 'a') as f:
            f.create_dataset(file_name, data=data, chunks=(1801, 1, 2560))
    elif ext == '.npz':
        append_npz(save_dir, file_name, data)
    else:
        raise ValueError(f"Unsupported file extension: '{ext}'")


def _run_model(model, dataloader, save_dir, rank=0):
    # Apply model
    if rank == 0:
        print(f"Applying Model ({len(dataloader)} Batches)...", flush=True)
        start = datetime.now()
    model_out = torch.zeros(1801, 2160, 2560, dtype=torch.float32)
    for i, (inpt, target, item) in enumerate(dataloader):
        # Run model on batch
        model.preprocess(inpt, target)
        model.run_passes()

        # if ddp, sync output batches across processes
        if dist.is_initialized():
            all_outputs = torch.empty(
                model.fakeB.shape[0] * dist.get_world_size(),
                *model.fakeB.shape[1:], dtype=torch.float32).cuda()
            dist.all_gather_into_tensor(all_outputs, model.fakeB)
            all_outputs.cpu()

            all_items = torch.empty(
                item.shape[0] * dist.get_world_size(),
                *item.shape[1:], dtype=item.dtype).cuda()
            dist.all_gather_into_tensor(all_items, item.cuda())
            all_items.cpu()
        else:
            all_outputs = model.fakeB
            all_items = item

        # Append batch to final array
        for p in range(len(all_items)):
            idx = dataloader.dataset.item_to_idx(all_items[p])
            model_out[idx] = all_outputs[p].detach()

        if i % 10 == 0 and rank == 0:
            print(f"\tBatch [{i + 1}/{len(dataloader)}]", flush=True)
    if rank == 0:
        print(f"Finished in {datetime.now() - start}\n", flush=True)

    # Save Model Output
    if rank == 0:
        print("Saving Output...", flush=True)
        start = datetime.now()
        save_output(toNumpy(model_out), save_dir)
        print(f"Done in {datetime.now() - start}\n", flush=True)


def init_ddp():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Rank {rank} of {world_size} on {gethostname()} "
          f"{gpus_per_node} allocated GPUs per node.", flush=True)
    print('Group initialization')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0:
        print(f"Group initialized: {dist.is_initialized()}", flush=True)
    else:
        print(f'Group initilized from rank {rank}')

    local_rank = torch.distributed.get_rank()
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")
    return world_size, rank


def load_dataset(data_dir, mask_file=None, ddp=False, world_size=0, rank=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        Rescale(a=-1, b=1, imin=0, imax=1)
    ])
    dataset = H5Dataset(data_dir, mask_file, transform=transform)
    if ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=10, sampler=sampler,
                                num_workers=int(
                                    os.environ["SLURM_CPUS_PER_TASK"]))
    else:
        dataloader = DataLoader(dataset, batch_size=10, shuffle=False,
                                num_workers=10)
    return dataloader


def apply_model(model_file, data_dir, save_dir, mask_file=None, device=None,
                ddp=False):
    # Initialise multi-node, multi-GPU training
    if ddp:
        world_size, rank = init_ddp()
    else:
        world_size = rank = 0

    # Load Dataset
    print("Loading Dataset...", flush=True)
    start = datetime.now()
    dataloader = load_dataset(data_dir, mask_file, ddp, world_size, rank)
    print(f"Done in {datetime.now() - start}\n", flush=True)

    # Load GAN
    print("\nLoading GAN...", flush=True)
    start = datetime.now()
    gan = load_model(model_file, device, ddp)
    print(f"Done in {datetime.now() - start}\n", flush=True)

    # Run model & save output
    _run_model(gan, dataloader, save_dir, rank)

    if ddp:
        dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser(description="Generate a stripe mask.")
    parser.add_argument("model", type=str,
                        help="Checkpoint file of model to use.")
    parser.add_argument("h5", type=str,
                        help="HDF5 or Nexus file containing raw tomographic "
                             "data to apply model to.")
    parser.add_argument("--flats", type=str, default=None,
                        help="HDF5 or Nexus file containing flats and darks.")
    parser.add_argument('-m', "--mask", type=str, default=None,
                        help="Path to archive file containg mask.")
    parser.add_argument('-o', "--out", type=str, default=None,
                        help="File to save output to. Can be HDF5 or NPZ.")
    parser.add_argument("--ddp", action="store_true",
                        help="Use multiple nodes and multiple GPUs")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model_path = Path(args.model)
    data_path = Path(args.h5)
    mask_path = Path(args.mask)
    save_path = Path(args.out)
    multi_node = args.ddp

    # Use GPU if available
    if torch.cuda.is_available():
        d = torch.device('cuda')
    else:
        d = torch.device('cpu')

    apply_model(model_path, data_path, save_path, mask_file=mask_path,
                device=d, ddp=multi_node)
