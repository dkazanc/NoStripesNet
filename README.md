# No Stripes Net

A neural network to remove stripe artifacts in sinograms.<br>
The network takes a sinogram with stripes as input, and learns to remove those stripes from it.<br>
The network can train on both synthetic and real-life data.<br>

### Requirements
 - A Linux machine with a GPU and CUDA
 - Conda
 - Python 3.9
 - PyTorch
 - TomoPy
 - HTTomo

For a full list of requirements, see [environment.yml](environment.yml)

### Installation

First, set up the project environment:
 - Clone the repository: `git clone https://github.com/dkazanc/NoStripesNet.git`
 - Create the conda environment: `conda env create -f environment.yml`
 - Activate the conda environment: `conda activate nostripesnet`
 
#### Generate Data
Next, generate some data to train/test on.<br>
If you have access to an HDF5 or Nexus file, you can generate a dataset from real-life data.<br>
Otherwise, you are limited to just synthetic data.<br>
 - Open the data generation script in a text editor: [run_scripts/data_generator.sh](run_scripts/data_generator.sh)
 - Change the parameters to suit your use case
 - Run the script: `./run_scripts/data_generator.sh`

For more information about the data created & the parameters, see [simulator/data_generator.py](simulator/data_generator.py)

#### Train a Model
Finally, train a model on the generated data.<br>
Run the following to see information about the options you can specify:<br>
`python -m network.training -h`<br>
Then choose the values of the options you want to specify, and run the same command again (without `-h`).<br>
For example, if training a masked model, you might run something like this:<br>
`python -m network.training --model mask --epochs 10 --save-dir ./data --verbose`<br>

Follow the same procedure for testing:<br>
`python -m network.testing -h`<br>
Choose your parameters, then run again with those options.<br>

If you want to both train and test a model all at once, follow these steps:<br>
 - Open the [train/test script](run_scripts/train_test.sh) in a text editor
 - Specify the values of the parameters you want to train & test with
 - Run the script: `./run_scripts/train_test.sh`

A graph of the training losses will be saved to `NoStripesNet/images`.<br>