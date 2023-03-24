# Data Generation
This document aims to explain the different methods of generating data, 
the parameters for data generation,
and how each parameter affects each method & the data it generates.<br>
There are two main ways of generating data for a network: 
 - **Synthetic** (*simulated using TomoPhantom*)
 - **Real-life** (*loaded from an HDF5 file*)
 
There are several sub-methods within these two:
 - **Synthetic**
   - Simple (*for a simplistic simulation of artifacts*)
   - Complex (*for a more realistic simulation of artifacts*)
 - **Real-life**
   - Raw (*saves the data as-is, with no post-processing/pair-finding*)
   - Paired (*creates input/target pairs based on artifacts in sinograms*)
   - Dynamic (*for dynamic tomography scans*)
   - Patch (*splits sinograms into smaller patches*)

The following parameters affect *all* methods of generating data:
 - `--mode`, `-m`
   - The method to use when generating data. Default is `complex`.
   - Must be one of `simple`, `complex`, `raw`, `paired`, `dynamic`, or `patch`.
   - If it is not recognised, an error will be raised.
 - `--root`, `-r`
   - The main directory under which to store generated data. Default is `./data`.
   - If the path given doesn't exist, it will be created.
   - Each method creates sub-directories in a specific way, so see the corresponding section for information about
   the method you want.
 - `--samples`, `-s`
   - The number of samples to generate (for synthetic) or load (for real-life). Default is `1`.
   - For synthetic, this will determine how many different 3D samples are generated.
   - For real-life, this will only determine the prefix to each image saved. Functionality for multiple hdf5 files of 
   different samples is not yet implemented.
 - `--start`
   - Sample number to begin counting at. Default is `0`.
   - This is useful if some data has already been generated, and you do not wish to overwrite it.
   - For example, if `samples=10` and `start=0`, sinograms will be saved under directory `0000`.
   Likewise, if `samples=10` and `start=5`, sinograms will be saved under directory `0005`.
 - `--verbose`, `-v`
   - Print out some extra information when running.

More detail is given below about each method, and how the parameters in the data generation 
script affect the data generated.


## Synthetic
All synthetic data is simulated using TomoPhantom.<br>
The general process of creating a synthetic dataset goes like this:<br>
- For each sample:
    1. Simulate a 3D sample. Samples consist of a number of spheres of various sizes and intensities.
    2. Forward project the sample to get a tomogram.<br>
    3. Add artifacts to the tomogram. The exact way in which this is done depends on the mode selected.<br>
    4. Save the tomogram with no stripes (clean) and with stripes (stripe) to disk as a series of pairs of 2D tiff images.
 
The following parameters affect the simulation of samples in both sub-methods of Synthetic data:
 - `--size`, `-N`: 
   - Controls the size in each dimension of the simulated sample. Default is `256`.
   - Size is equal in each dimension, i.e. the sample is cubic.
   - This will be different to the size of sinograms generated. This is because in forward projection, the size of
   dimensions changes.
   - For example, if `size=256`, the samples will be `(256, 256, 256)` but tomograms will be `(256, 402, 362)`.
 - `--objects`, `-o`:
   - Controls the number of spheres created for each sample. Default is `300`.
   - Other parameters for spheres, such as radius, position, and intensity, are hard-coded.

### Simple
In Simple mode, artifacts are simulated using TomoPhantom's `_Artifacts_` method.<br>
*Only* stripes are simulated; no flat fields, no noise, no other type of artifact.<br>
Data is stored in the following directory structure:<br>
```
root
├── <sample number>
│   ├── clean
│   │   ├── 0000_clean_0000.tif
│   │   │   ...
│   │   └── 0000_clean_0255.tif
│   └── stripe
│       ├── 0000_stripe_0000.tif
│       │   ...
│       └── 0000_stripe_0255.tif
...
```

There are no extra parameters that affect Simple mode.<br>

### Complex
In Complex mode, artifacts are created by simulating flat fields. Specifically, this adds more realistic noise and 
stripes to sinograms.<br>
Additionally, flat fields are vertically shifted (in detector Y dimension) to simulate the shifting of the sample in 
real-life data. This means we end up with not just one clean/stripe pair, but as many pairs as there are shifts.<br>
The directory structure looks similar to that of Simple mode, but contains multiple sub-directories for each shift:<br>
```
├── 0000
│   ├── clean
│   │   ├── 0000_clean_0000.tif
│   │   │   ...
│   │   └── 0000_clean_0255.tif
│   ├── shift00
│   │   ├── 0000_shift00_0000.tif
│   │   │   ...
│   │   └── 0000_shift00_0255.tif
│   ├── shift01
│   │   ├── 0000_shift01_0000.tif
│   │   │   ...
│   │   └── 0000_shift01_0255.tif
│   ├── shift02
│   │   ├── 0000_shift02_0000.tif
│   │   │   ...
│   │   └── 0000_shift02_0255.tif
│   ├── shift03
│   │   ├── 0000_shift03_0000.tif
│   │   │   ...
│   │   └── 0000_shift03_0255.tif
│   └── shift04
│       ├── 0000_shift04_0000.tif
│       │   ...
│       └── 0000_shift04_0255.tif
...
``` 
In Complex mode, even the `clean` sinograms still contain noise. This is so that the *only* difference between inputs 
and targets is the *stripes*; in previous experiments where `clean` sinograms contained no noise, the network did not 
perform as well, most likely because addition of *both* noise *and* stripes in inputs was confusing it.<br>

The following parameters affect Complex mode:
  - `--shifts`, `-s`
    - The number of shifts to simulate. Default is `5`.
    - Each shift is stored under its own sub-directory.
  - `--shiftstep`, `-p`
    - The step in pixels to shift each flat field by. Default is `2`.
  - `--flatsnum`, `-f`
    - The number of flat fields to generate. Default is `20`.
    - A larger number of flat fields will have more varied effects on noise and stripes.
  - `--I0`, `-I`
    - Full beam photon flux intensity. Default is `40000`.
    - Affects the amount of noise added to data, as well as the offset for certain detector pixels.


## Real-life    
It is assumed that real-life data exists in a .nxs file.<br>
Files are loaded using HTTomo, and the [HTTomo pipeline yaml file](../tomo_pipeline.yml) can be used to specify things 
like the path to the data, the path to the image key, the padding, etc.<br>
A lot of tomographic data will be too large to fit in memory all at once, and so the data generation scripts all load
data in *chunks*, by loading only a subset of sinograms at a time.<br>
If you are running the script on a workstation that has enough memory to load the entire tomogram at once (such as an 
HPC cluster node), you can set chunk size to the total number of sinograms.<br>

Most of the real-life data generating scripts follow these steps:
- For each shift:
  1. For each chunk:
      1. Load the data from the .nxs file
      2. Downsample sinograms to size `(402, 362)`
      3. Save this chunk to disk
  2. Once the entire shift has been saved, re-load each chunk from disk and normalise w.r.t. the entire 3D sample.
  
The following parameters affect all sub-methods of Real-life data:
  - `--pipeline`
    - The filename of the HTTomo pipeline yaml file. Default is `tomo_pipeline.yml`.
  - `--hdf-root`
    - The .nxs file to load data from. Default is `None`.
    - This should be shift 0, e.g. `1000.nxs` in the below examples.
    - If not specified, and mode is one of `raw`, `paired`, or `dynamic`, an error will be raised.
  - `-C`, `--chunk-size`
    - Size of chunks to load data in. Default is `243`.
    - If you have a computer with lots of memory, you can set chunk size to the full size of the dataset to load the 
    whole dataset at once.
    - This means the data does not need to be re-loaded (i.e. step 2 above does not occur).
  - `--flats`
    - Path to HDF/Nexus file containing flat & dark field data to normalize with. Default is `None`.
    - If not specified, flats & darks will be loaded from `hdf-root`. If `hdf-root` contains no flats or darks, an error will be raised.
    

### Raw
This method applies no post-processing or pair-finding, and just saves sinograms as they are in the hdf5 file.<br>
As a result, no input/target or stripe/clean pairs are saved; sinograms are just saved under their shift directory.<br>
This method stores data in the following directory structure:<br>
```
root
└── 0000
    ├── shift00
    │   ├── 0000_shift00_0000.tif
    │   │   ...
    │   └── 0000_shift00_2159.tif
    ├── shift01
    │   ├── 0000_shift01_0000.tif
    │   │   ...
    │   └── 0000_shift01_2159.tif
    ├── shift02
    │   ├── 0000_shift02_0000.tif
    │   │   ...
    │   └── 0000_shift02_2159.tif
    ...
```  

The following parameters affect Raw mode:
  - `--shifts`, `-s`
    - The number of vertical shifts when sample was scanned. Default is `5`.
    - Each shift is stored under its own sub-directory.
    - Assumes shifts are each stored under a different .nxs file, with the name incrementing by one for each shift.
    - For example, if shift 0 was in `1000.nxs`, shift 1 will be in `1001.nxs`, shift 2 in `1002.nxs`, etc.
  - `--shiftstep`, `-p`
    - The step in pixels between each shift. Default is `2`.

### Paired
This method creates input/target (or stripe/clean) pairs based on whether sinograms from the data contain stripes.<br>
It calculates a binary mask indicating the locations of stripes, then based on this mask determines whether a sinogram
has a stripe in or not.<br>
If it doesn't have a stripe, stripes are added synthetically using TomoPhantoms `stripe` method.<br>
If it does have a stripe, a 'clean' image is created by combining the parts of different shifts that don't contain 
stripes.<br>
> N.B. combining different shifts to get a clean image is not implemented in the current version of this method.
> The current method assumes the 'clean' images have already been generated using the old function [getCleanStripe](../simulator/realdata_loader.py).
> This is not useful and is therefore likely to be changed in a future version.<br>

The sinograms with and without stripes are saved to disk as a stripe/clean pair.<br>
However, depending on whether the raw sinogram contained stripes, the sub-directory under which each pair is stored is different.<br>
- If the raw sinogram *didn't* contain a stripe, the pair is stored under `root/fake_artifacts`.<br>
- If the raw sinogram *did* contain a stripe, the pair is stored under `root/real_artifacts`.<br>

And so, this method stores data in the following directory structure:<br>
```
root
├── fake_artifacts
│   ├── clean
│   │   ├── 0000_shift00_0000.tif
│   │   │   ...
│   │   ├── 0000_shift00_2159.tif
│   │   ├── 0000_shift01_0000.tif
│   │   │   ...
│   │   ├── 0000_shift01_2159.tif
│   │   ...
│   └── stripe
│       ├── 0000_shift00_0000.tif
│       │   ...
│       ├── 0000_shift00_2159.tif
│       ├── 0000_shift01_0000.tif
│       │   ...
│       ├── 0000_shift01_2159.tif
│       ...
└── real_artifacts
    ├── clean
    │   ├── 0000_shift00_0255.tif
    │   │   ...
    │   ├── 0000_shift00_1815.tif
    │   ├── 0000_shift01_0255.tif
    │   │   ...
    │   ├── 0000_shift01_1815.tif
    │   ...
    └── stripe
        ├── 0000_shift00_0255.tif
        │   ...
        ├── 0000_shift00_1815.tif
        ├── 0000_shift01_0255.tif
        │   ...
        ├── 0000_shift01_1815.tif
        ...
```
Due to the nature of the pair-creation algorithm, any sinogram in `root/fake_artifacts` cannot also exist in `root/real_artifacts`.<br>
Additionally, there are no shift sub-directories; all shifts are stored under one directory.<br>

The following parameters affect Paired mode:
  - `--shifts`, `-s`
    - The number of vertical shifts when sample was scanned. Default is `5`.
    - All shifts are stored under the same directory.
    - Assumes shifts are each stored under a different .nxs file, with the name incrementing by one for each shift.
    - For example, if shift 0 was in `1000.nxs`, shift 1 will be in `1001.nxs`, shift 2 in `1002.nxs`, etc.
  - `--shiftstep`, `-p`
    - The step in pixels between each shift. Default is `2`.
  - `--mask`
    - The path to the mask containing locations of stripes. Default is `None`.
    - This should be a numpy file `.npy`, generated using the stripe detection method in Larix.
    - If not given, a mask will be generated at runtime. However, this can take multiple hours. 

### Dynamic
This method should only be used for dynamic tomographic experiments.<br>
It assumes that all "frames" of the dynamic experiment are stored in one hdf5 file in the angles dimension.<br>
Therefore, shifts are not used in this method.<br>
"Frames" are created by only retrieving a subset of angles, assumed to create a sinogram for an individual time period.<br>

Data is stored in the following structure:<br>
```
root
├── 0000_frame00_0000.tif
│   ...
├── 0000_frame00_2159.tif
├── 0000_frame01_0000.tif
│   ...
├── 0000_frame01_2159.tif
├── 0000_frame02_0000.tif
│   ...
├── 0000_frame02_2159.tif
...
```
All samples and frames are stored under root.<br>

The following parameters affect Dynamic mode:<br>
  - `--frame-angles`
    - The number of angles that make up a sinogram, i.e. one 'frame' of a dynamic scan.
    - Default is `900`.

### Patch
This method is the same as Paired, but split sinograms into patches of a given size.<br>
Therefore, no downsampling is applied to sinograms.<br>

This method stores data in the following directory structure:<br>
```
root
├── fake_artifacts
│   ├── clean
│   │   ├── 0000_shift00_0000_w00.tif
│   │   ├── 0000_shift00_0000_w01.tif
│   │   ├── 0000_shift00_0000_w02.tif
│   │   │   ...
│   │   ├── 0000_shift00_0001_w00.tif
│   │   │   ...
│   │   ├── 0000_shift01_0000_w00.tif
│   │   │   ...
│   │   ...
│   └── stripe
└── real_artifacts
    ├── clean
    └── stripe
```
It is the same as Paired mode, but for each sinogram there is a series of patches `w00`, `w01`, `w02`, ...<br>

The following parameters affect Patch mode:<br>
  - `--shifts`, `-s`
    - The number of vertical shifts when sample was scanned. Default is `5`.
    - All shifts are stored under the same directory.
    - Assumes shifts are each stored under a different .nxs file, with the name incrementing by one for each shift.
    - For example, if shift 0 was in `1000.nxs`, shift 1 will be in `1001.nxs`, shift 2 in `1002.nxs`, etc.
  - `--mask`
    - The path to the mask containing locations of stripes. Default is `None`.
    - This should be a numpy file `.npy`, generated using the stripe detection method in Larix.
    - If not given, a mask will be generated at runtime. However, this can take multiple hours.
  - `--patch-size`
    - The size of patches to split sinograms into. Must be a tuple. Default is `(1801, 256)`.
    - If the sinogram does not evenly go into patches of size `patch-size`, the sinogram will be cropped.
