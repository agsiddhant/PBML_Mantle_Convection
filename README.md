Code for a physics-based surrogate stokes model for mantle convection simulations.

Step 1: Download some sample data and model weights from:
https://doi.org/10.5281/zenodo.15088589

Step 2: After installing PyTorch, run load_fluidnet.ipynb to see how to load a trained network and predict.

Step 3: See network_lists.ipynb to see how to perform a training run.

Step 4: If you have access to mantle convection numerical code GAIA, see advection_runs.ipynb for running advection-diffusion with GAIA based on velocities from the PyTorch model.