# Training and Testing
This document aims to explain how to train & test the models in this project, including a description of the possible
parameters you can specify and how they affect a model's performance.<br>

It is recommended that you create a separate directory to store models in, for example `NoStripesNet/pretrained_models/`.<br>
This is because often you will have quite a few models stored at once, and storing them directly in `NoStripesNet/` 
or one of the pre-existing subdirectories can quickly clutter up the project.<br>

When a model is saved, the name is determined by the type of model and the epoch.<br>
For example, if training a masked model for 10 epochs, the name of the model will be `mask_10.tar`.<br>
It is recommended that immediately after the model is saved, you rename it to something more descriptive. This is 
because you will likely be training a number of similar models with slightly different hyperparameters, and it is easy 
to get confused between which model is which. In addition, if you are not careful, you may end up overwriting a model.<br>

There are several types of models that can be trained:
  - base (*full sinograms for input & output*)
  - mask (*masked sinograms for input & output*)
  - simple (*same as mask, but masks are calculated in a simpler way*)
  - patch (*same as simple, but works on patches of sinograms*)
  - window & full (*uses windowed sinograms, left over from an old method & not used anymore*)

More details about each mode can be found below.<br>

The following parameters are shared between both training and testing:
  - `--root`, `-r`
    - The path to the directory in which input data is stored. Default is `./data`.
  - `--model`, `-m`
    - The type of model to train. Default is `base`.
    - For more details about each type of model, see below.
  - `--size`, `-N`
    - The number of sinograms per sample. Default is `256`.
    - This will be the same as `size` provided in data generation.
  - `--shifts`, `-s`,
    - The number of shifts per sample. Default is `5`.
    - This will be the same as `shifts` provided in data generation.
  - `--tvt`
    - The train:validate:test ratio for the data. Default is `3 1 1`.
    - Should be entered as three numbers with a space between each. For example, the ratio `7:2:1` would be entered as `7 2 1`.
  - `--batch-size`, `-B`
    - The size of batches to load data in. Default is `16`.
  - `--verbose`, `-v`
    - Print out some extra information when running.
  - `--window-width`, `-w`
    - Width of windows to split sinograms into. Only used if `mode = window`.
    - This is left over from an old version of the project, and is not really used anymore.


### Training
To train a model, run the following command:
```
python -m network.training <options here>
```
The training script supports both training from scratch and pretrained models.<br>
The training script will run for the specified number of epochs, then once completed will plot a graph of the loss, and
display images from the training process (i.e. input & output sinograms and their reconstructions).
Finally, you will be given the option to save the model at the end of training. You must specify a save directory for this.<br>

The following options are specific to training:
  - `--epochs`, `-e`
    - The number of epochs to train the model for. Default is `1`. 
  - `--learning-rate`, `-l`
    - The learning rate to use when training the model. Default is `0.0002`.
    - A learning rate scheduler is used that reduces the LR on plateau.
  - `--betas`, `-b`
    - The two beta parameters used in the Adam optimizer. Default is `0.5 0.999`.
    - Should be entered as two floats with a space between. The first is beta 1, the second is beta 2.
  - `--lambda`
    - The weight for the L1 loss in the generator. Default is `100`.
    - This balances the two loss functions for the generator: adversarial (BCE with D) and absolute (L1 with targets).
    - The L1 loss is multiplied by this parameter. Set to `0` if you wish to train a purely adversarial GAN.
  - `--lsgan`
    - Flag to enable training of a Least Squares GAN.
    - This means no Sigmoid is applied to the final layer of the Discriminator, and MSE is used as the adversarial loss.
  - `--save-dir`, `-d`
    - Directory to save models to. If no directory is specified, models cannot be saved.
  - `--model-file`, `-f`
    - Path to a file containing a model, if you wish to train a pretrained model rather than from scratch.
  - `--subset`
    - Train using a subset of the full dataset.
    - If given as an option, it should contain the size of the subset as a parameter
    - For example, if you wanted to train with a subset of size 1000: `--subset 1000`
  - `--force`
    - Force the process to keep running regardless of any stops/user inputs
    - This will make all plots non-blocking, and will pass 'y' to all user inputs. It will also save the graph of the loss to `../images`.
    - This option is used in `../run_scripts/train_test.sh` so that testing runs straight after training.
  - `--save-every-epoch`
    - Save the model at the end of every epoch.
    - If given, option `--save-dir` must be given as well.

To see a re-cap of the options and what they do, include `-h`: `python -m network.training -h`<br>


### Testing
To test a model, run the following command:
```
python -m network.testing <options here>
```
The testing script loads models from a given path and evaluates their performance.<br>
If you really want to, the testing script also supports testing a brand new, untrained model.<br>
The testing script will run for a number of batches, specified by the `--tvt` option.<br>
Once finished, the testing script will show inputs & outputs of the generator, the accuracy of the discriminator, as 
well as the scores for each testing metric.<br>

There are a number of possible testing metrics, each identified using a number:
0. L1 (or Mean Absolute Error)
1. L2
2. Mean Squared Error
3. Sum of the absolute difference between gradients of inputs and targets
4. Dice Coefficient (or F1 score)
5. Intersection over Union
6. Histogram Intersection
7. Structural Similarity
8. Peak Signal to Noise Ratio

The following options are specific for testing:<br>
  - `--model-file`, `-f`
    - The filepath of the model to test.
  - `--metrics`, `-M`
    - The test metrics to calculate for the model. Must either be string `all` or a list of integers in range `[0, 9)`.
    - For example, if you wanted to calculate metrics 1, 4, 5, and 7: `--metrics 1 4 5 7`.
    - Default is `all`
  - `--display-each-batch`
    - Whether to plot images & display metric scores each batch.
  - `--visual-only`
    - If given, metric scores will not be calculated and only one batch will be run. The results of this batch will be
    immediately displayed.

To see a re-cap of the options and what they do, include `-h`: `python -m network.testing -h`<br>


## Types of Models
For information about the architecture of each model, see the classes inside [models](./models).

### Base
Base models take full sinograms `(402, 362)` as input, and outputs full sinograms.<br>
The corresponding Generator class is `BaseUNet`, and the Discriminator class is `BaseDiscriminator`.<br>
The GAN class which contains the Generator & Discriminator is `BaseGAN`.<br>
As this model outputs full sinograms, the results can often be blurry/low resolution.<br>
Therefore, this model is mainly used as a parent for other models, rather than to actually train a model.<br>

### Mask
Masked models take masked sinograms `(402, 362)` as input, and output masked sinograms.<br>
A "masked" sinogram is one that has had its stripes set to zero. This is done by the use of a binary mask indicating the 
locations of stripes in the sinogram. The code segment looks like this:<br>
```python
sinogram[mask] = 0
```
This masked sinogram is input to the Generator, and the output of the Generator is then only applied to the locations of
stripes, like so:
```python
gen_out = generator(sinogram)
sinogram[mask] = gen_out[mask]
```
This means the generator only inpaints in the parts of the sinogram affected by stripes, increasing the resolution of the output.<br>

The masks can be calculated in a number of ways. The current implementation uses a combination of a few different stripe
detection methods; for more information see [utils/stripe_detection.py](../utils/stripe_detection.py)

The Generator class for Mask models is `BaseUNet`, and the Discriminator class is `BaseDiscriminator`.<br>
The GAN class is `MaskedGAN`.<br>

### Simple
Simple models are identical to Mask models, other than how masks are calculated.<br>
While Mask models use a number of different stripe detection algorithms, Simple models only compute the absolute 
difference between a "clean" sinogram with no stripes, and the corresponding sinogram with stripes.
```python
mask = np.abs(clean - stripe)
mask[mask > 0] = 1
mask = mask.astype(np.bool_)
```
The Generator class for Simple models is `BaseUNet`, and the Discriminator class is `BaseDiscriminator`.<br>
The GAN class is `MaskedGAN`.<br>

### Patch
Patch models are identical to Simple models, but rather than working on full sinograms, the model is applied to patches.<br>
Patches must be of size `(1801, 256)` and generated according to [Patch](../simulator/README.md#Patch) mode in data generation.<br>
The Generator class for Path models is `PatchUNet`, and the Discriminator class is `PatchDiscriminator`<br>
The GAN class is `MaskedGAN`.<br>

### Window and Full
These two types of model are left over from an old version of the project, and are no longer used.<br>
They use windowed sinograms.<br>
The Generator class for Window models is `WindowUNet`, and the Discriminator class is `WindowDiscriminator`.<br>
The GAN class is `WindowGAN`.<br>
The Generator class for Full models is `BaseUNet`, and the Discriminator class is `BaseDiscriminator`.<br>
The GAN class is `BaseGAN`.<br>
