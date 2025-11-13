"""
Quick Demo: Mantle Convection Visualization
===========================================
This script creates a static visualization showing what to expect
from the full animation. Run this first to verify your setup.

Usage:
    python demo_visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys

print("="*70)
print("Mantle Convection Visualization Demo")
print("="*70)
print()
print("This demo creates a static snapshot of mantle convection.")
print("For full animation, run:")
print("  python simple_plume_visualizer.py")
print("  or")
print("  python realtime_mantle_simulation.py")
print()
print("="*70)
print()

# Grid setup
nx, ny = 200, 50
aspect_ratio = 4.0
x = np.linspace(0, aspect_ratio, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Create temperature field with plumes
print("Generating temperature field with thermal plumes...")
T = 1.0 - Y  # base profile: hot at bottom, cold at top

# Add hot plumes
plume_positions = [0.8, 2.0, 3.2]
for px in plume_positions:
    r = np.sqrt(((X - px) / 0.3)**2 + ((Y - 0.3) / 0.5)**2)
    T += 0.4 * np.exp(-r**2)

# Add cold downwellings
for px in [0.2, 1.4, 2.6, 3.8]:
    r = np.sqrt(((X - px) / 0.2)**2 + ((Y - 0.7) / 0.4)**2)
    T -= 0.25 * np.exp(-r**2)

T = np.clip(T, 0, 1.3)

# Approximate velocity field (buoyancy-driven)
print("Computing velocity field...")
Ra = 5.0  # Rayleigh number
v = Ra * (T - 0.5) * 0.15  # vertical velocity from buoyancy
v[0, :] = 0   # no-slip bottom
v[-1, :] = 0  # no-slip top

# Horizontal velocity from mass conservation
dv_dy = np.gradient(v, axis=0)
u = -np.cumsum(dv_dy, axis=1) * (aspect_ratio / nx)
u[:, 0] = 0
u[:, -1] = 0

vel_mag = np.sqrt(u**2 + v**2)

# Approximate viscosity (temperature-dependent)
print("Computing viscosity field...")
eta = np.exp(-5 * (T - 0.5))  # lower viscosity for hot material
eta = np.clip(eta, 0.01, 100)

# Create visualization
print("Creating visualization...")
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Mantle Convection Simulation Snapshot (Rayleigh Number = 5.0)',
             fontsize=16, fontweight='bold')

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

# 1. Temperature field - Main view
ax1 = fig.add_subplot(gs[0, :])
im1 = ax1.contourf(X, Y, T, levels=25, cmap='hot')
ax1.set_title('Temperature Field - Hot Plumes Rising from Bottom',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('Horizontal Distance', fontsize=10)
ax1.set_ylabel('Depth', fontsize=10)
ax1.set_aspect('equal')
cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.1)
cbar1.set_label('Temperature', fontsize=10)

# Add annotations for plumes
ax1.annotate('Hot Plume', xy=(0.8, 0.3), xytext=(0.5, 0.6),
            arrowprops=dict(arrowstyle='->', color='yellow', lw=2),
            fontsize=10, color='yellow', fontweight='bold')
ax1.annotate('Cold Downwelling', xy=(1.4, 0.7), xytext=(1.4, 0.9),
            arrowprops=dict(arrowstyle='->', color='cyan', lw=2),
            fontsize=10, color='cyan', fontweight='bold')

# 2. Vertical velocity (upwelling/downwelling)
ax2 = fig.add_subplot(gs[1, 0])
im2 = ax2.contourf(X, Y, v, levels=20, cmap='RdBu_r')
ax2.set_title('Vertical Velocity\n(Red=Up, Blue=Down)', fontsize=11)
ax2.set_xlabel('x', fontsize=9)
ax2.set_ylabel('y', fontsize=9)
plt.colorbar(im2, ax=ax2)

# 3. Horizontal velocity
ax3 = fig.add_subplot(gs[1, 1])
im3 = ax3.contourf(X, Y, u, levels=20, cmap='RdBu_r')
ax3.set_title('Horizontal Velocity', fontsize=11)
ax3.set_xlabel('x', fontsize=9)
ax3.set_ylabel('y', fontsize=9)
plt.colorbar(im3, ax=ax3)

# 4. Velocity magnitude
ax4 = fig.add_subplot(gs[1, 2])
im4 = ax4.contourf(X, Y, vel_mag, levels=20, cmap='inferno')
ax4.set_title('Velocity Magnitude', fontsize=11)
ax4.set_xlabel('x', fontsize=9)
ax4.set_ylabel('y', fontsize=9)
plt.colorbar(im4, ax=ax4)

# 5. Streamlines
ax5 = fig.add_subplot(gs[2, 0])
psi = np.cumsum(v, axis=1) * (aspect_ratio / nx)
ax5.contour(X, Y, psi, levels=15, colors='black', linewidths=1)
ax5.contourf(X, Y, psi, levels=15, cmap='coolwarm', alpha=0.7)
ax5.set_title('Streamlines\n(Flow Circulation)', fontsize=11)
ax5.set_xlabel('x', fontsize=9)
ax5.set_ylabel('y', fontsize=9)

# 6. Viscosity field
ax6 = fig.add_subplot(gs[2, 1])
im6 = ax6.contourf(X, Y, np.log10(eta), levels=20, cmap='plasma')
ax6.set_title('Log10(Viscosity)', fontsize=11)
ax6.set_xlabel('x', fontsize=9)
ax6.set_ylabel('y', fontsize=9)
plt.colorbar(im6, ax=ax6)

# 7. Velocity vectors on temperature
ax7 = fig.add_subplot(gs[2, 2])
ax7.contourf(X, Y, T, levels=20, cmap='hot', alpha=0.4)
skip = 4
ax7.quiver(X[::skip, ::skip], Y[::skip, ::skip],
          u[::skip, ::skip], v[::skip, ::skip],
          vel_mag[::skip, ::skip], cmap='cool', scale=8, width=0.004)
ax7.set_title('Velocity Vectors', fontsize=11)
ax7.set_xlabel('x', fontsize=9)
ax7.set_ylabel('y', fontsize=9)

plt.tight_layout()

# Save figure
output_file = 'mantle_convection_demo.png'
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"✓ Visualization saved to {output_file}")
print()

# Display
print("Displaying visualization...")
print("Close the window to exit.")
print()
plt.show()

print("="*70)
print("Demo complete!")
print()
print("Next steps:")
print("  1. Run 'python simple_plume_visualizer.py' for interactive animation")
print("  2. Or run 'python realtime_mantle_simulation.py' for neural network mode")
print("  3. See VISUALIZATION_GUIDE.md for full documentation")
print("="*70)
