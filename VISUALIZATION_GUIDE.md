# Real-Time Mantle Convection Visualization Guide

This guide explains how to use the real-time visualization tools for simulating and visualizing mantle convection with thermal plumes using the pretrained neural network models.

## Overview

Two visualization scripts are provided:

1. **`realtime_mantle_simulation.py`** - Full-featured simulator using pretrained neural networks
2. **`simple_plume_visualizer.py`** - Simplified interactive visualizer (works without pretrained weights)

## Quick Start

### Option 1: Simple Visualizer (No Data Required)

The simplest way to see mantle convection in action:

```bash
python simple_plume_visualizer.py
```

This runs a physics-based simulation with:
- Interactive controls to adjust convection vigor (Rayleigh number)
- Real-time visualization of thermal plumes rising from the bottom
- Multiple views: temperature, velocity, streamlines, and velocity vectors

**Features:**
- ✓ No data download required
- ✓ Interactive sliders to control parameters
- ✓ Works on CPU or GPU
- ✓ Shows thermal plumes and mantle circulation patterns

### Option 2: Full Neural Network Simulator

For high-accuracy predictions using the pretrained model:

```bash
# Basic usage (synthetic data)
python realtime_mantle_simulation.py

# With GPU
python realtime_mantle_simulation.py --gpu 0

# Custom Rayleigh number
python realtime_mantle_simulation.py --rayleigh 8.5

# Save animation
python realtime_mantle_simulation.py --save mantle_animation.gif
```

## Installation Requirements

### Basic Requirements (for simple visualizer):
```bash
pip install numpy matplotlib torch
```

### Full Requirements (for neural network simulator):
```bash
pip install numpy matplotlib torch
# Plus the repository code (pytorch_networks_convae.py, etc.)
```

## Download Pretrained Weights (Optional)

For the most accurate neural network predictions:

1. Download data and weights from Zenodo:
   - URL: https://doi.org/10.5281/zenodo.15088589

2. Extract and place the weight file (`81_fluidnet_uvp.pt`) in one of these locations:
   - `./nn_weights/81_fluidnet_uvp.pt`
   - `./weights/81_fluidnet_uvp.pt`
   - `./81_fluidnet_uvp.pt`

## What You'll See

Both visualizers show multiple views of mantle convection:

### 1. **Temperature Field**
- Hot plumes (red) rising from the bottom boundary
- Cold downwellings (blue) sinking from the top
- Dynamic thermal boundary layers

### 2. **Velocity Fields**
- **Horizontal velocity (u)**: Left-right flow patterns
- **Vertical velocity (v)**: Upwelling plumes and downwelling currents
- **Velocity magnitude**: Overall flow strength

### 3. **Streamlines**
- Shows circulation patterns in the mantle
- Convection cells and flow paths
- Closed loops indicate stable circulation

### 4. **Viscosity Field** (full simulator only)
- Temperature and depth-dependent viscosity
- Shows stiff vs. flowing regions
- Based on Frank-Kamenetskii viscosity law

### 5. **Velocity Vectors**
- Arrows showing flow direction and magnitude
- Overlaid on temperature field
- Shows complex flow patterns around plumes

## Understanding the Physics

### Key Parameters

**Rayleigh Number (Ra)**
- Controls the vigor of convection
- Low Ra (<1): Conductive, no plumes
- Moderate Ra (1-10): Steady plumes
- High Ra (>10): Turbulent, chaotic flow

**Frank-Kamenetskii Parameters**
- **γ (gamma)**: Temperature-dependent viscosity contrast
- **β (beta)**: Pressure/depth-dependent viscosity contrast
- Higher values = more viscosity variation

### What Are Mantle Plumes?

Mantle plumes are columns of hot rock rising through Earth's mantle:
- Start at the core-mantle boundary (~2900 km depth)
- Rise due to buoyancy (thermal expansion)
- Create hotspot volcanism (e.g., Hawaii, Iceland)
- Surrounded by cold, downwelling material

The visualizations show:
- **Hot upwellings** (plumes): Red regions with positive vertical velocity
- **Cold downwellings**: Blue regions with negative vertical velocity
- **Circulation cells**: Complete convection patterns

## Command-Line Options

### realtime_mantle_simulation.py

```bash
python realtime_mantle_simulation.py [OPTIONS]

Options:
  --gpu GPU_ID          GPU device to use (default: 0)
  --rayleigh FLOAT      Rayleigh number (default: 5.0)
  --save FILENAME       Save animation to file (gif or mp4)
  --use-data            Use real Zenodo data instead of synthetic
```

### simple_plume_visualizer.py

```bash
python simple_plume_visualizer.py

Interactive controls:
  - Rayleigh slider: Adjust convection vigor in real-time
  - Reset button: Restart simulation with new plumes
```

## Examples

### Example 1: Weak Convection
```bash
python realtime_mantle_simulation.py --rayleigh 1.0
```
Shows gentle, steady plume rise.

### Example 2: Vigorous Convection
```bash
python realtime_mantle_simulation.py --rayleigh 9.0
```
Shows turbulent, time-varying plumes with complex interactions.

### Example 3: Save Animation
```bash
python realtime_mantle_simulation.py --rayleigh 5.0 --save plumes.gif
```
Creates an animated GIF of the simulation.

### Example 4: Interactive Exploration
```bash
python simple_plume_visualizer.py
```
Then adjust the Rayleigh slider while the simulation runs to see how convection changes.

## Technical Details

### Neural Network Architecture

The full simulator uses a physics-informed neural network:
- **Architecture**: NewFluidNet (modified U-Net)
- **Input**: Temperature field + physics parameters
- **Output**: Velocity field (u, v) via curl formulation
- **Training**: Supervised on GAIA mantle convection simulations
- **Loss**: Curl-based divergence-free constraint + data fitting

### Computational Performance

**Simple Visualizer:**
- ~20 FPS on modern CPU
- Grid: 200×50 points
- Memory: <500 MB

**Full Simulator (with neural network):**
- ~15-20 FPS on GPU
- ~5-10 FPS on CPU
- Grid: 506×128 points
- Memory: ~2-4 GB (GPU)

### Grid Specifications

**Spatial Domain:**
- Aspect ratio: 4:1 (width:height)
- Represents: 4× wider than tall (like Earth's mantle in 2D)
- Resolution: Adjustable (default 506×128)

**Boundary Conditions:**
- Top: Cold (T=0), free-slip
- Bottom: Hot (T=1), free-slip
- Sides: Periodic or free-slip

## Troubleshooting

### "No pretrained weights found"
- This is normal if you haven't downloaded the Zenodo data
- The simulator will still run with random initialization
- For accurate results, download weights from Zenodo

### "CUDA out of memory"
- Use CPU mode (remove --gpu flag)
- Or reduce grid resolution in the code

### Visualization is slow
- Use the simple visualizer instead
- Reduce frame rate (increase `interval` in code)
- Use smaller grid (modify nx, ny in code)

### Animation won't save
- Install pillow: `pip install pillow`
- For MP4, install ffmpeg: `conda install ffmpeg` or system package

## Understanding the Output

### Interpreting Temperature Patterns

- **Hot plumes (red)**: Rising hot material from bottom
- **Cold sheets (blue)**: Sinking cold material from top
- **Boundary layers**: Thin thermal gradients at top and bottom

### Interpreting Velocity Patterns

- **Positive v (red)**: Upward flow (plumes)
- **Negative v (blue)**: Downward flow (slabs)
- **Horizontal u**: Compensating flow to conserve mass

### Interpreting Streamlines

- **Closed loops**: Convection cells
- **Dense lines**: Fast flow
- **Sparse lines**: Slow flow
- **Number of cells**: Depends on Rayleigh number and aspect ratio

## Scientific Context

This visualization demonstrates:

1. **Thermal Convection**: Heat transfer by fluid motion
2. **Rayleigh-Bénard Convection**: Classic convection driven by bottom heating
3. **Mantle Dynamics**: Simplified model of Earth's mantle circulation
4. **Plume Tectonics**: Mechanism for hotspot volcanism
5. **Plate Driving Forces**: Sinking slabs drive plate motions

### Limitations

This is a simplified 2D model. Real Earth's mantle has:
- 3D structure
- Compositional variations
- Phase transitions
- Complex rheology
- Time-dependent boundaries

## References

If you use this code, please cite:
- The original repository and paper (see main README.md)
- Zenodo data archive: https://doi.org/10.5281/zenodo.15088589

## Tips for Best Results

1. **Start with simple visualizer** to understand the physics
2. **Download pretrained weights** for accurate neural network predictions
3. **Adjust Rayleigh number** to explore different convection regimes
4. **Save interesting animations** for presentations
5. **Compare with load_fluidnet.ipynb** to validate against ground truth

## Further Exploration

After familiarizing yourself with the visualizers:

1. Modify physics parameters in the code (viscosity contrasts)
2. Change boundary conditions
3. Add compositional heterogeneity
4. Implement 3D visualization
5. Train new models with different physics

---

**Enjoy exploring mantle dynamics!** 🌋🔥

For questions or issues, please open a GitHub issue or consult the main README.md.
