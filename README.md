### Physics-Based Machine Learning for Mantle Convection <br />

This repository contains code for a physics-based surrogate model that approximates Stokes flow in mantle convection simulations using deep learning.

## Quick Start - Real-Time Visualization 🌋

**NEW!** Interactive real-time visualization of mantle convection and thermal plumes:

```bash
# Quick demo (no data download required)
python simple_plume_visualizer.py

# Or see a static snapshot first
python demo_visualization.py

# Full neural network simulator (requires pretrained weights)
python realtime_mantle_simulation.py
```

See **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)** for complete documentation.

## Getting Started

#### Step 1: Download Data and Pretrained Model Weights
Download the sample datasets and pretrained model weights from the Zenodo archive: https://doi.org/10.5281/zenodo.15088589

#### Step 2: Load a Trained Model
After installing PyTorch, run `load_fluidnet.ipynb` to load a pretrained neural network and perform velocity predictions.

#### Step 3: Train a New Model
To train the surrogate model from scratch or explore training configurations, look at `network_lists.ipynb`.

#### Step 4: Integrate with GAIA (Optional)
If you have access to the GAIA mantle convection code, you can use the surrogate model's velocities in advection-diffusion simulations. See `advection_runs.ipynb` for details.

#### Step 5: Real-Time Visualization (NEW!)
Explore interactive simulations of mantle convection with thermal plumes:
- **`simple_plume_visualizer.py`** - Interactive visualizer with adjustable parameters (no data required)
- **`realtime_mantle_simulation.py`** - Full neural network simulator with pretrained weights
- **`demo_visualization.py`** - Quick static visualization demo

Features:
- Real-time animation of thermal plumes rising from the mantle
- Multiple views: temperature, velocity, streamlines, viscosity
- Interactive controls to adjust convection vigor
- Works with or without downloaded data
