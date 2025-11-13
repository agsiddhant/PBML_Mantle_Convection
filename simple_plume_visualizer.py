"""
Simple Interactive Plume Visualizer
====================================
A simplified, interactive version for quick visualization of mantle plumes.

Usage:
    python simple_plume_visualizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import torch

# Try to import local modules (if weights are available)
try:
    from pytorch_networks_convae import *
    from scaler import scale_var, unscale_var
    HAS_MODEL = True
except:
    HAS_MODEL = False
    print("⚠ Warning: Could not import model. Using simple fluid dynamics approximation.")


class SimplePlumeVisualizer:
    """Simple interactive visualizer for thermal plumes"""

    def __init__(self, nx=200, ny=50):
        self.nx = nx
        self.ny = ny
        self.aspect_ratio = 4.0

        # Create grid
        x = np.linspace(0, self.aspect_ratio, nx)
        y = np.linspace(0, 1, ny)
        self.X, self.Y = np.meshgrid(x, y)

        # Simulation parameters
        self.rayleigh = 5.0
        self.num_plumes = 3
        self.time = 0
        self.dt = 0.05

        # Temperature field
        self.T = self._initialize_temperature()

        # Velocity field (approximated)
        self.u = np.zeros_like(self.T)
        self.v = np.zeros_like(self.T)

        # Setup figure
        self._setup_figure()

    def _initialize_temperature(self):
        """Initialize temperature field with plumes"""
        # Base profile: hot at bottom, cold at top
        T = 1.0 - self.Y

        # Add random plume seeds at the bottom
        for i in range(self.num_plumes):
            x_pos = np.random.uniform(0.5, 3.5)
            r = np.sqrt(((self.X - x_pos) / 0.2)**2 + ((self.Y - 0.1) / 0.1)**2)
            T += 0.3 * np.exp(-r**2)

        return T

    def _compute_velocity_simple(self):
        """Compute velocity using simple buoyancy-driven flow approximation"""
        # Vertical velocity proportional to temperature anomaly
        T_mean = 0.5  # reference temperature
        self.v = self.rayleigh * (self.T - T_mean) * 0.1

        # Apply boundary conditions
        self.v[0, :] = 0  # no flow at bottom
        self.v[-1, :] = 0  # no flow at top

        # Horizontal velocity from mass conservation (simplified)
        # du/dx + dv/dy = 0
        dv_dy = np.gradient(self.v, axis=0)
        self.u = -np.cumsum(dv_dy, axis=1) * (self.aspect_ratio / self.nx)

        # Apply periodic boundary conditions in x
        self.u[:, 0] = 0
        self.u[:, -1] = 0

    def _update_temperature(self):
        """Update temperature field using advection-diffusion"""
        # Compute velocities
        self._compute_velocity_simple()

        # Simple advection (upwind scheme)
        dT_dx = np.gradient(self.T, axis=1)
        dT_dy = np.gradient(self.T, axis=0)

        # Advection
        T_new = self.T - self.dt * (self.u * dT_dx + self.v * dT_dy)

        # Diffusion (Laplacian)
        kappa = 0.01  # thermal diffusivity
        d2T_dx2 = np.gradient(np.gradient(self.T, axis=1), axis=1)
        d2T_dy2 = np.gradient(np.gradient(self.T, axis=0), axis=0)
        T_new += self.dt * kappa * (d2T_dx2 + d2T_dy2)

        # Apply boundary conditions
        T_new[0, :] = 1.0  # hot bottom
        T_new[-1, :] = 0.0  # cold top

        # Periodic boundaries in x
        T_new[:, 0] = T_new[:, 1]
        T_new[:, -1] = T_new[:, -2]

        self.T = np.clip(T_new, 0, 1.2)

    def _setup_figure(self):
        """Setup interactive matplotlib figure"""
        self.fig = plt.figure(figsize=(14, 10))

        # Create grid for subplots
        gs = self.fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3,
                                    left=0.1, right=0.95, top=0.92, bottom=0.15)

        self.ax_temp = self.fig.add_subplot(gs[0, :])
        self.ax_vel_v = self.fig.add_subplot(gs[1, 0])
        self.ax_vel_u = self.fig.add_subplot(gs[1, 1])
        self.ax_stream = self.fig.add_subplot(gs[2, 0])
        self.ax_quiver = self.fig.add_subplot(gs[2, 1])

        # Add sliders for interactive control
        ax_ra = plt.axes([0.2, 0.05, 0.6, 0.02])
        self.slider_ra = Slider(ax_ra, 'Rayleigh', 0.1, 10.0, valinit=self.rayleigh)
        self.slider_ra.on_changed(self.update_rayleigh)

        # Add reset button
        ax_reset = plt.axes([0.85, 0.05, 0.1, 0.03])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset)

    def update_rayleigh(self, val):
        """Update Rayleigh number from slider"""
        self.rayleigh = val

    def reset(self, event):
        """Reset simulation"""
        self.time = 0
        self.T = self._initialize_temperature()
        self.u = np.zeros_like(self.T)
        self.v = np.zeros_like(self.T)

    def animate(self, frame):
        """Animation function"""
        # Update physics
        self._update_temperature()
        self.time += self.dt

        # Compute derived quantities
        vel_mag = np.sqrt(self.u**2 + self.v**2)

        # Clear axes
        for ax in [self.ax_temp, self.ax_vel_v, self.ax_vel_u, self.ax_stream, self.ax_quiver]:
            ax.clear()

        # 1. Temperature with plumes
        im1 = self.ax_temp.contourf(self.X, self.Y, self.T, levels=25, cmap='hot')
        self.ax_temp.set_title(f'Temperature Field - Mantle Plumes (t={self.time:.2f})',
                               fontweight='bold', fontsize=12)
        self.ax_temp.set_xlabel('Horizontal Distance')
        self.ax_temp.set_ylabel('Depth')
        plt.colorbar(im1, ax=self.ax_temp, label='Temperature')

        # 2. Vertical velocity (plume upwelling)
        im2 = self.ax_vel_v.contourf(self.X, self.Y, self.v, levels=20, cmap='RdBu_r')
        self.ax_vel_v.set_title('Vertical Velocity (Plume Rise)')
        self.ax_vel_v.set_xlabel('x')
        self.ax_vel_v.set_ylabel('Depth')
        plt.colorbar(im2, ax=self.ax_vel_v)

        # 3. Horizontal velocity
        im3 = self.ax_vel_u.contourf(self.X, self.Y, self.u, levels=20, cmap='RdBu_r')
        self.ax_vel_u.set_title('Horizontal Velocity')
        self.ax_vel_u.set_xlabel('x')
        self.ax_vel_u.set_ylabel('Depth')
        plt.colorbar(im3, ax=self.ax_vel_u)

        # 4. Streamlines showing circulation
        # Compute streamfunction
        psi = np.cumsum(self.v, axis=1) * (self.aspect_ratio / self.nx)
        self.ax_stream.contour(self.X, self.Y, psi, levels=15, colors='black', linewidths=1)
        self.ax_stream.contourf(self.X, self.Y, psi, levels=15, cmap='coolwarm', alpha=0.7)
        self.ax_stream.set_title('Streamlines (Flow Circulation)')
        self.ax_stream.set_xlabel('x')
        self.ax_stream.set_ylabel('Depth')

        # 5. Velocity vectors overlaid on temperature
        skip = 4
        self.ax_quiver.contourf(self.X, self.Y, self.T, levels=20, cmap='hot', alpha=0.5)
        self.ax_quiver.quiver(
            self.X[::skip, ::skip], self.Y[::skip, ::skip],
            self.u[::skip, ::skip], self.v[::skip, ::skip],
            vel_mag[::skip, ::skip],
            cmap='cool', scale=5, width=0.003
        )
        self.ax_quiver.set_title('Velocity Vectors')
        self.ax_quiver.set_xlabel('x')
        self.ax_quiver.set_ylabel('Depth')

        # Overall title
        self.fig.suptitle(
            f'Interactive Mantle Convection Simulation (Ra={self.rayleigh:.1f})',
            fontsize=14, fontweight='bold'
        )

    def run(self):
        """Run the interactive visualization"""
        print("="*60)
        print("Simple Plume Visualizer")
        print("="*60)
        print("Controls:")
        print("  - Adjust Rayleigh slider to change convection vigor")
        print("  - Click 'Reset' to restart simulation")
        print("  - Close window to exit")
        print("="*60)
        print()

        anim = FuncAnimation(
            self.fig, self.animate,
            frames=None,  # Run indefinitely
            interval=50,   # 50ms = 20 FPS
            blit=False,
            repeat=True
        )

        plt.show()

        return anim


def main():
    """Main entry point"""
    print("Starting Simple Plume Visualizer...")
    print()

    if not HAS_MODEL:
        print("Note: Running in simple physics mode (no neural network).")
        print("For full accuracy, ensure pytorch_networks_convae.py is available.")
        print()

    visualizer = SimplePlumeVisualizer(nx=200, ny=50)
    visualizer.run()


if __name__ == "__main__":
    main()
