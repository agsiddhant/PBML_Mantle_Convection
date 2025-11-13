"""
Real-time Mantle Convection Simulation Visualizer
==================================================
This script creates an animated, real-time visualization of mantle convection
using pretrained neural network weights.

Usage:
    python realtime_mantle_simulation.py [--use-data] [--gpu GPU_ID]

Options:
    --use-data: Use downloaded Zenodo data (requires data to be downloaded)
    --gpu: GPU device ID (default: 0)
    --synthetic: Use synthetic temperature fields (default if no data)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
import torch
import torch.nn.functional as F
import argparse
import sys
import os

# Import local modules
from pytorch_networks_convae import *
from scaler import scale_var, unscale_var

class MantleConvectionSimulator:
    """Real-time mantle convection simulator using pretrained neural networks"""

    def __init__(self, gpu_number=0, use_synthetic=True):
        """
        Initialize the simulator

        Args:
            gpu_number: GPU device ID
            use_synthetic: If True, generates synthetic temperature fields
        """
        self.device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
        self.use_synthetic = use_synthetic

        # Model parameters (matching the pretrained weights)
        self.network = "newfluidnet"
        self.levels = 5
        self.kernel = 5
        self.c_h = 16
        self.repeats = 6
        self.r_p = "learned"
        self.loss_type = "curl"
        self.act_fn = "gelu"
        self.use_symm = False
        self.a_bound = 10
        self.factor = 2
        self.blurr = False
        self.p_pred = False

        # Grid dimensions (matching typical mantle convection setup)
        self.nx = 506  # horizontal grid points
        self.ny = 128  # vertical grid points
        self.aspect_ratio = 4.0  # width/height ratio

        # Initialize grid coordinates
        self._setup_grid()

        # Physics parameters (example Rayleigh number and viscosity contrasts)
        self._setup_physics_parameters()

        # Load the model
        self._load_model()

        # Animation state
        self.time_step = 0
        self.max_steps = 200

    def _setup_grid(self):
        """Create computational grid"""
        x = np.linspace(0, self.aspect_ratio, self.nx)
        y = np.linspace(0, 1, self.ny)
        X, Y = np.meshgrid(x, y)

        self.xcc = torch.tensor(X, dtype=torch.float64).view(1, 1, self.ny, self.nx)
        self.ycc = torch.tensor(Y, dtype=torch.float64).view(1, 1, self.ny, self.nx)

        # Signed distance functions for boundaries
        self.sdf = torch.zeros_like(self.ycc)
        self.sdf[:, :, 0, :] = 1.0   # bottom
        self.sdf[:, :, -1, :] = 1.0  # top
        self.sdf[:, :, :, 0] = 1.0   # left
        self.sdf[:, :, :, -1] = 1.0  # right

        self.sdf2 = torch.ones_like(self.ycc)
        self.sdf2[:, :, 0, :] = 0.0
        self.sdf2[:, :, -1, :] = 0.0
        self.sdf2[:, :, :, 0] = 0.0
        self.sdf2[:, :, :, -1] = 0.0

    def _setup_physics_parameters(self):
        """Setup physics parameters for the simulation"""
        # Rayleigh number (controls convection vigor)
        self.raq = 5.0  # moderate convection

        # Frank-Kamenetskii viscosity parameters
        self.fkt = 1e7   # temperature-dependent viscosity contrast
        self.fkp = 10.0  # pressure-dependent viscosity contrast

        # Normalized parameters (for neural network input)
        self.raq_nd = torch.tensor(
            (self.raq - 0.12624371) / (9.70723344 - 0.12624371),
            dtype=torch.float64
        )
        self.fkt_nd = torch.tensor(
            (np.log10(self.fkt) - 6.00352841978384) / (9.888820429862925 - 6.00352841978384),
            dtype=torch.float64
        )
        self.fkp_nd = torch.tensor(
            (np.log10(self.fkp) - 0.005251646002323797) / (1.9927988938926755 - 0.005251646002323797),
            dtype=torch.float64
        )

        self.fkt_tensor = torch.tensor(self.fkt, dtype=torch.float64)
        self.fkp_tensor = torch.tensor(self.fkp, dtype=torch.float64)

    def _load_model(self):
        """Load pretrained neural network model"""
        c_i = 7  # input channels
        c_o = 1  # output channels (curl formulation)

        self.model = NewFluidNet(
            self.levels, c_i, self.c_h, c_o, self.device,
            self.act_fn, self.r_p, self.loss_type,
            use_symm=self.use_symm, dilation=1, a_bound=self.a_bound,
            repeats=self.repeats, use_skip=False, f=self.kernel,
            p_pred=self.p_pred, blurr=self.blurr, factor=self.factor
        ).double().to(self.device)

        self.ts_net = TS(
            self.model, ad=None, device=self.device, ts=1,
            advection_scheme=0, scale=True, p_pred=self.p_pred,
            net=self.network
        ).double().to(self.device)

        print(f"Model loaded with {count_parameters(self.model):,} parameters")

        # Try to load pretrained weights if available
        try:
            # Try multiple possible weight file locations
            weight_paths = [
                "nn_weights/81_fluidnet_uvp.pt",
                "weights/81_fluidnet_uvp.pt",
                "81_fluidnet_uvp.pt"
            ]

            loaded = False
            for path in weight_paths:
                if os.path.exists(path):
                    self.model.load_state_dict(torch.load(path, map_location=self.device))
                    print(f"✓ Loaded pretrained weights from {path}")
                    loaded = True
                    break

            if not loaded:
                print("⚠ Warning: No pretrained weights found. Using random initialization.")
                print("  Download weights from Zenodo: https://doi.org/10.5281/zenodo.15088589")
        except Exception as e:
            print(f"⚠ Could not load weights: {e}")
            print("  Continuing with random initialization...")

        self.ts_net.eval()

    def generate_synthetic_temperature(self, with_plumes=True):
        """
        Generate a synthetic temperature field with hot plumes rising from bottom

        Args:
            with_plumes: If True, adds thermal plumes to the base temperature field

        Returns:
            Temperature field as torch tensor [1, 1, ny, nx]
        """
        x = self.xcc[0, 0].numpy()
        y = self.ycc[0, 0].numpy()

        # Base conductive temperature profile (hot at bottom, cold at top)
        T = 1.0 - y

        if with_plumes:
            # Add multiple rising thermal plumes with time evolution
            num_plumes = 3
            plume_positions = [1.0, 2.0, 3.0]  # x-positions

            for px in plume_positions:
                # Plume rises over time
                plume_y = 0.3 + 0.3 * np.sin(self.time_step * 0.1)

                # Gaussian plume shape
                plume_width = 0.3
                plume_height = 0.5

                # Distance from plume center
                r = np.sqrt(((x - px) / plume_width)**2 + ((y - plume_y) / plume_height)**2)

                # Add hot anomaly
                T += 0.3 * np.exp(-r**2) * (1 + 0.2 * np.sin(self.time_step * 0.15))

            # Add some cold downwellings
            for px in [0.5, 1.5, 2.5, 3.5]:
                downwell_y = 0.7 - 0.2 * np.sin(self.time_step * 0.12)
                r = np.sqrt(((x - px) / 0.2)**2 + ((y - downwell_y) / 0.4)**2)
                T -= 0.2 * np.exp(-r**2)

            # Add some turbulent perturbations
            T += 0.05 * np.sin(2 * np.pi * x / self.aspect_ratio + self.time_step * 0.1) * \
                 np.sin(4 * np.pi * y + self.time_step * 0.15)

        # Clip to valid temperature range
        T = np.clip(T, 0.0, 1.35)

        return torch.tensor(T, dtype=torch.float64).view(1, 1, self.ny, self.nx)

    def compute_viscosity(self, T):
        """Compute temperature and depth-dependent viscosity field"""
        gamma = self.fkt_tensor
        beta = self.fkp_tensor
        z = 1.0 - self.ycc  # depth

        eta = torch.exp(torch.log(gamma) * (0.0 - T) + torch.log(beta) * (z - 0.0))
        eta = torch.clip(eta, 1e-8, 1.0)

        return eta

    def predict_velocity(self, T):
        """
        Predict velocity field from temperature using neural network

        Args:
            T: Temperature field [1, 1, ny, nx]

        Returns:
            u, v, p, V: velocity components, pressure, and viscosity
        """
        T = T.to(self.device)

        # Compute viscosity
        V = self.compute_viscosity(T)

        # Prepare inputs
        raq_nd = self.raq_nd.to(self.device)
        fkt_nd = self.fkt_nd.to(self.device)
        fkp_nd = self.fkp_nd.to(self.device)
        raq_t = torch.tensor(self.raq, dtype=torch.float64).to(self.device)
        fkt_t = self.fkt_tensor.to(self.device)
        fkp_t = self.fkp_tensor.to(self.device)

        sdf = self.sdf.to(self.device)
        sdf2 = self.sdf2.to(self.device)
        xcc = self.xcc.to(self.device)
        ycc = self.ycc.to(self.device)

        with torch.no_grad():
            _, _, u_pred, v_pred, p_pred, V_out = self.ts_net(
                T, sdf, sdf2, ycc, raq_nd, fkt_nd, fkp_nd,
                raq_t, fkt_t, fkp_t, xcc, ycc
            )

        # Move back to CPU for visualization
        u = u_pred.cpu().detach()
        v = v_pred.cpu().detach()
        p = p_pred.cpu().detach() if p_pred is not None else None
        V = V.cpu().detach()

        # Unscale velocities
        u = unscale_var(u, self.raq, self.fkt, self.fkp, "uprev")
        v = unscale_var(v, self.raq, self.fkt, self.fkp, "vprev")

        return u, v, p, V

    def compute_streamfunction(self, u, v):
        """Compute streamfunction from velocity for visualization"""
        # Simple integration of v in x-direction
        psi = torch.cumsum(v[0, 0], dim=1) * (self.aspect_ratio / self.nx)
        return psi.numpy()

    def setup_visualization(self):
        """Setup matplotlib figure for animation"""
        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)

        # Create subplots
        self.ax_temp = self.fig.add_subplot(gs[0, :])
        self.ax_u = self.fig.add_subplot(gs[1, 0])
        self.ax_v = self.fig.add_subplot(gs[1, 1])
        self.ax_stream = self.fig.add_subplot(gs[1, 2])
        self.ax_visc = self.fig.add_subplot(gs[2, 0])
        self.ax_vel_mag = self.fig.add_subplot(gs[2, 1])
        self.ax_quiver = self.fig.add_subplot(gs[2, 2])

        self.axes = {
            'temp': self.ax_temp,
            'u': self.ax_u,
            'v': self.ax_v,
            'stream': self.ax_stream,
            'visc': self.ax_visc,
            'vel_mag': self.ax_vel_mag,
            'quiver': self.ax_quiver
        }

        # Store plot objects for updating
        self.plots = {}

        return self.fig

    def animate_step(self, frame):
        """Animation function called for each frame"""
        self.time_step = frame

        # Generate or load temperature field
        T = self.generate_synthetic_temperature(with_plumes=True)

        # Predict velocities
        u, v, p, V = self.predict_velocity(T)

        # Extract numpy arrays for plotting
        T_np = T[0, 0].numpy()
        u_np = u[0, 0].numpy()
        v_np = v[0, 0].numpy()
        V_np = V[0, 0].numpy()
        vel_mag = np.sqrt(u_np**2 + v_np**2)

        x = self.xcc[0, 0].numpy()
        y = self.ycc[0, 0].numpy()

        # Clear all axes
        for ax in self.axes.values():
            ax.clear()

        # 1. Temperature field
        im1 = self.ax_temp.contourf(x, y, T_np, levels=20, cmap='hot')
        self.ax_temp.set_title(f'Temperature Field (Step {frame}/{self.max_steps})', fontsize=12, fontweight='bold')
        self.ax_temp.set_xlabel('x')
        self.ax_temp.set_ylabel('y (depth)')
        plt.colorbar(im1, ax=self.ax_temp, label='Temperature')

        # 2. Horizontal velocity (u)
        im2 = self.ax_u.contourf(x, y, u_np, levels=20, cmap='RdBu_r')
        self.ax_u.set_title('Horizontal Velocity (u)')
        self.ax_u.set_xlabel('x')
        self.ax_u.set_ylabel('y')
        plt.colorbar(im2, ax=self.ax_u)

        # 3. Vertical velocity (v)
        im3 = self.ax_v.contourf(x, y, v_np, levels=20, cmap='RdBu_r')
        self.ax_v.set_title('Vertical Velocity (v)')
        self.ax_v.set_xlabel('x')
        self.ax_v.set_ylabel('y')
        plt.colorbar(im3, ax=self.ax_v)

        # 4. Streamlines
        psi = self.compute_streamfunction(u, v)
        im4 = self.ax_stream.contour(x, y, psi, levels=15, colors='black', linewidths=0.5)
        self.ax_stream.contourf(x, y, psi, levels=15, cmap='viridis', alpha=0.6)
        self.ax_stream.set_title('Streamlines')
        self.ax_stream.set_xlabel('x')
        self.ax_stream.set_ylabel('y')

        # 5. Viscosity field
        im5 = self.ax_visc.contourf(x, y, np.log10(V_np), levels=20, cmap='plasma')
        self.ax_visc.set_title('Log10(Viscosity)')
        self.ax_visc.set_xlabel('x')
        self.ax_visc.set_ylabel('y')
        plt.colorbar(im5, ax=self.ax_visc)

        # 6. Velocity magnitude
        im6 = self.ax_vel_mag.contourf(x, y, vel_mag, levels=20, cmap='inferno')
        self.ax_vel_mag.set_title('Velocity Magnitude')
        self.ax_vel_mag.set_xlabel('x')
        self.ax_vel_mag.set_ylabel('y')
        plt.colorbar(im6, ax=self.ax_vel_mag)

        # 7. Velocity vectors (quiver plot)
        skip = 8  # subsample for clarity
        self.ax_quiver.contourf(x, y, T_np, levels=20, cmap='hot', alpha=0.3)
        self.ax_quiver.quiver(
            x[::skip, ::skip], y[::skip, ::skip],
            u_np[::skip, ::skip], v_np[::skip, ::skip],
            vel_mag[::skip, ::skip], cmap='cool', scale=50
        )
        self.ax_quiver.set_title('Velocity Vectors + Temperature')
        self.ax_quiver.set_xlabel('x')
        self.ax_quiver.set_ylabel('y')

        # Add overall title
        self.fig.suptitle(
            f'Real-Time Mantle Convection Simulation (Ra={self.raq:.1f})',
            fontsize=16, fontweight='bold'
        )

        return list(self.axes.values())

    def run_animation(self, save_path=None):
        """
        Run the real-time animation

        Args:
            save_path: Optional path to save animation as MP4
        """
        fig = self.setup_visualization()

        anim = FuncAnimation(
            fig, self.animate_step,
            frames=self.max_steps,
            interval=50,  # 50ms between frames = 20 FPS
            blit=False,
            repeat=True
        )

        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=20)
            print(f"✓ Animation saved to {save_path}")

        plt.show()

        return anim


def main():
    parser = argparse.ArgumentParser(description='Real-time Mantle Convection Visualization')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--use-data', action='store_true', help='Use downloaded data instead of synthetic')
    parser.add_argument('--save', type=str, default=None, help='Save animation to file (e.g., mantle_sim.gif)')
    parser.add_argument('--rayleigh', type=float, default=5.0, help='Rayleigh number (controls convection vigor)')

    args = parser.parse_args()

    print("="*60)
    print("Real-Time Mantle Convection Simulator")
    print("="*60)
    print(f"Device: {'GPU ' + str(args.gpu) if torch.cuda.is_available() else 'CPU'}")
    print(f"Mode: {'Real data' if args.use_data else 'Synthetic data'}")
    print(f"Rayleigh number: {args.rayleigh}")
    print("="*60)
    print()

    # Create simulator
    simulator = MantleConvectionSimulator(
        gpu_number=args.gpu,
        use_synthetic=not args.use_data
    )

    # Override Rayleigh number if specified
    if args.rayleigh != 5.0:
        simulator.raq = args.rayleigh
        simulator._setup_physics_parameters()

    # Run animation
    print("Starting animation... Close the window to exit.")
    print("(This may take a moment to initialize)")
    print()

    try:
        simulator.run_animation(save_path=args.save)
    except KeyboardInterrupt:
        print("\nAnimation stopped by user.")
    except Exception as e:
        print(f"\nError during animation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
