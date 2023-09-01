# No Stripes Net

A neural network to remove stripe artifacts in sinograms.<br>
The network takes a sinogram with stripes as input, and learns to remove those stripes from it.<br>
The network can train on both synthetic and real-life data.<br>

### Requirements
 - A Linux machine with a GPU and CUDA
 - Conda
 - Python 3.9+
 - PyTorch
 - TomoPy

For a full list of requirements, see [environment.yml](environment.yml)

### Installation

First, set up the project environment:
 - Clone the repository: `git clone https://github.com/dkazanc/NoStripesNet.git`
 - Create the conda environment: `conda env create -f environment.yml`
 - Activate the conda environment: `conda activate nostripesnet`
 

### The Repository
- `network/` contains Python code to train and test a model, as well as the dataset and visualiser classes.
- `run_scripts/` contains bash scripts to generate masks & datasets and train/test models.
- `simulator` contains Python code to generate masks & datasets.
- `utils/` - contains utility functions used throughout the codebase.
- `TUTORIAL.md` is a walkthrough of how to generate a dataset, and train & apply a model.
- `apply_model.py` is a program that applies a model to a given tomographic scan.
- `graphs.ipynb` is a Jupyter Notebook used to create the graphs in the paper.
- `residuals.ipynb` is a Jupyter Notebook used to create the residual images in the paper.
- `rmse.ipynb` is a Jupyter Notebook used to calculate the RMSEs in the paper.
- `submit.sh` is a bash script to train a model on multiple nodes, using multiple GPUs on each.
- `visualize_results.ipynb` is a Jupyter Notebook used to visualize the results of a model.  
 

### Running the Code
A full walkthrough of how to generate a dataset and train a model can be found [here](./TUTORIAL.md).<br>
To apply a trained model to a tomographic scan, see [run_scripts/apply_model.sh](./run_scripts/apply_model.sh).
