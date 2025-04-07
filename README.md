### Physics-Based Machine Learning for Mantle Convection <br />

This repository contains code for a physics-based surrogate model that approximates Stokes flow in mantle convection simulations using deep learning.

#### Step 1: Download Data and Pretrained Model Weights 
Download the sample datasets and pretrained model weights from the Zenodo archive: 
📦 https://doi.org/10.5281/zenodo.15088589

#### Step 2: Load a Trained Model
After installing PyTorch, run `load_fluidnet.ipynb` to load a pretrained neural network and perform velocity predictions.

#### Step 3: Train a New Model
To train the surrogate model from scratch or explore training configurations, look at `network_lists.ipynb`.

#### Step 4: Integrate with GAIA (Optional)
If you have access to the GAIA mantle convection code, you can use the surrogate model’s velocities in advection-diffusion simulations. See `advection_runs.ipynb` for details.
