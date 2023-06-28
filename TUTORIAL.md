# NoStripesNet Tutorial
## Introduction
This document details the entire process of creating an ML model using NoStripesNet, from generating a dataset to visualizing the results of a model.

### Prerequisites
Before starting this tutorial, you should have cloned the repository and created a conda environment for the project.


## Step 1 - Generating a Dataset
A Dataset should contain pairs of "clean" images (*without* artifacts) and "stripe" images (*with* artifacts). One dataset can be used for both training and testing (and a train/test split can be specified).<br>
There are a few different types of dataset that can be generated, which are referred to as **modes**. These are properly detailed in [simulator/README.md](simulator/README.md), but for this tutorial we will just focus on *patch* mode.<br>

Patch mode requires real-life tomographic data, which should be stored in an HDF file. Each sinogram is split into patches, which are saved to the dataset and used to train a model.<br>
Naturally, real-life tomographic data will contain real-life stripe artifacts. However, we cannot use these artifacts to train a model, as we have no clean images to compare them to.<br>
This limits the dataset to only include patches that *don't* have any artifacts in them. We save these patches as "clean", and then simulate artifacts on each patch to get a corresponding "stripe" image. These clean/stripe pairs are stored under a subdirectory called `fake_artifacts`.<br>
Rather than completely discarding patches containing artifacts, we store them in a separate subdirectory `real_artifacts`, so at the very least a visual analysis can still be performed.<br>

Therefore, the structure of a patch dataset looks like this:
```
root
└── 0000
    ├── fake_artifacts
    │   ├── clean
    │   │   ├── 0000_shift00_0000_w00.tif
    │   │   ├── 0000_shift00_0000_w01.tif
    │   │   ├── 0000_shift00_0000_w02.tif
    │   │   │   ...
    │   │   ├── 0000_shift00_0001_w00.tif
    │   │   │   ...
    │   │   ├── 0000_shift00_2159_w09.tif
    │   │   │   ...
    │   │   ...
    │   └── stripe
    └── real_artifacts
        ├── clean
        └── stripe
```
`root` is the top-level directory of the dataset.
`0000` is the sample number, identifying different samples (e.g. different HDF files or scans).
<br>Each patch is stored as a `.tif` image, with the following filename:
<br>`<sample_no>_<shift_no>_<sinogram_idx>_<patch_idx>.tif`<br>
where:
 - `<sample_no>` is the sample number
 - `<shift_no>` is the shift number (left over from an older version of the project)
 - `<sinogram_idx>` is the index of the sinogram within the full 3D tomographic data
 - `<patch_idx>` is the index of the patch within the sinogram

<br>It is worth noting that `real_artifacts/clean` is empty.

### Step 1a - Generating a Mask
In order to know which patches contain artifacts, we must detect real-life stripes in the tomographic data. The stripe detection algorithm used is from Larix, a combination of `larix.methods.misc.STRIPES_DETECT` and `larix.methods.misc.STRIPES_MERGE`.<br>
The mask generated is stored in a compressed NumPy zip archive `.npz`. There are multiple mask files stored in one archive. For example, `/dls/i12/data/2022/nt33730-1/processing/NoStripesNet/stripe_masks.npz` contains the stripe masks for lots of different samples.<br>
In order for dataset creation to run correctly, the `.npz` archive must be in the correct format. Each file in the archive must be named `m<hdf_name>`, where `<hdf_name>` is the name of the hdf file containing the tomographic data used to generate the mask.<br>
For example, if you were generating a a mask from `119617.nxs`, the mask file (within the archive) should be named `m119617`.<br>

The command below can be used to generate a mask:
```shell script
python -m simulator.generate_mask
```
It takes the following arguments:
 - `--h5`
   - The hdf file used to detect stripes and generate a mask.
 - `--flats`
   - An hdf file that contains flats and darks to normalise `--h5` with.
   - If `--h5` already contains flats and darks, this argument can be left out.
 - `-a`, `--archive`
   - The path to the `.npz` archive.
   - If it does not exist, it will be created.
   - If it does exist, a new file will be created within the archive.
 - `-f`, `--file-name`
   - The name of the file within the archive.
   - As stated above, this should be `m<hdf_name>`.
   
Alternatively, a [bash script](run_scripts/generate_mask.sh) exists to generate masks. This is SLURM-compatible, so can be run on an HPC cluster node. Arguments must be specified inside the script.

### Step 1b - Creating a dataset
Now that a mask has been generated, you can create a dataset from an HDF file.
This can be done using the following command:
```shell script
python -m simulator.data_generator
```
It takes a range of arguments, including the dataset root, the HDF file, and the mask file.<br>
    
An example command to generate patch data might look like this:
```shell script
python -m simulator.data_generator --mode patch --root ./data --hdf-root 119617.nxs --chunk-size 2160 --mask stripe_masks.npz --verbose
```

There is also a [bash script](run_scripts/data_generator.sh) to generate a dataset. This is SLURM-compatible, so can be run on an HPC cluster node. Arguments must be specified inside the script.

#### N.B.
There are many pre-existing datasets in `/dls/i12/data/2022/nt33730-1/processing/NoStripesNet/data`.<br>
From now on, we will be using `/dls/i12/data/2022/nt33730-1/processing/NoStripesNet/data/wider_stripes`.


## Step 2 - Training a Model
Like with Datasets, there are many different modes of training a model. The mode we will focus on is *patch*. A more in-depth (and slightly outdated) description of training and its modes can be found [here](network/README.md#types-of-models).<br>
Models can be trained from scratch, or from a pre-trained model.

The following command can be used to train a model:
```shell script
python -m network.training
``` 
You can specify a dataset directory, a path to save the model, a train/test split, as well as any hyperparameters.<br>
An example command used to train a patch model might look like this:
```shell script
python -m network.training --mode patch --root ./data --save-dir ./pretrained_models --name patch_example --epochs 10 --verbose
```
A full description of arguments can be found [here](network/README.md).<br>

There is also a [bash script](submit.sh) to train a model. This is SLURM-compatible, so can be run on an HPC cluster node. Arguments must be specified inside the script.

#### N.B.
There are many pre-trained models in `/dls/i12/data/2022/nt33730-1/processing/NoStripesNet/pretrained_models`.<br>
Some good examples are `.../pretrained_models/more_data/masked_chkpts/md_masked_100.tar`, or `.../pretrained_models/wider_stripes/wider_stripes_100.tar`.


## Step 3 - Testing a Model
Testing a model is very similar to training. A detailed description can be found [here](network/README.md#testing).<br>

The following command can be used to test a model:
```shell script
python -m network.testing
``` 
You can specify a dataset directory, a path to the model, a train/test split, as well as a number of test metrics to calculate.<br>
An example command used to test a patch model might look like this:
```shell script
python -m network.training --mode patch --root ./data --model-file ./pretrained_models/patch_example_1.tar --name patch_example --verbose
```
A full description of arguments can be found [here](network/README.md)

There is a [bash script](run_scripts/train_test.sh) which combines training and testing into one script. However, this is not SLURM-compatible, so might be slow. Arguments must be specified within the script.

## Step 4 - Visualizing the Results
Once a model has been trained and tested, the final step is the visualize its results. This can be done using the [PatchVisualizer](network/patch_visualizer.py) class. There is a Jupyter Notebook which will set up the model and plot various images.<br>