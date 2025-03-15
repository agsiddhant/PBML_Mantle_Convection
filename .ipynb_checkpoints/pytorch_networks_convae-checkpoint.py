import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import time
from symmetric_layers_torch import SymmetricConv2d
from torchvision.transforms import v2


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def get_mass(u, v, bc=False):
    """
    Computes the mass of a fluid system based on the velocity components in the x and y directions.

    Args:
        u (Tensor): The velocity component in the x-direction.
        v (Tensor): The velocity component in the y-direction.
        bc (bool, optional): If True, adjusts the boundary conditions by scaling the first and last points of the gradients.
                              Defaults to False.

    Returns:
        Tensor: The sum of the gradients of the velocity components in both x and y directions.
    """
    u = u.view(-1, 1, 128, 506)
    v = v.view(-1, 1, 128, 506)
    du_dx = F.conv2d(u, dx_center_kernel)[..., 1:-1, :]
    dv_dy = F.conv2d(v, dy_center_kernel)[..., :, 1:-1]

    if bc:
        du_dx[:, :, :, 0] *= 2.0 / 1.5
        du_dx[:, :, :, -1] *= 2.0 / 1.5

        dv_dy[:, :, 0, :] *= 2.0 / 1.5
        dv_dy[:, :, -1, :] *= 2.0 / 1.5

    return du_dx + dv_dy


def pad_grad(x, p=(1, 1, 1, 1)):
    """
    Pads the gradients of a tensor in all four spatial directions (left, right, top, bottom) using reflective padding.

    Args:
        x (Tensor): The tensor representing the gradient to be padded.
        p (tuple of int, optional): A tuple indicating how much padding should be applied in each direction (left, right, top, bottom).
                                      Defaults to (1, 1, 1, 1).

    Returns:
        Tensor: The padded gradient tensor.
    """
    for _ in range(p[0]):
        x_b = 2 * x[:, :, :, 0:1] - x[:, :, :, 1:2]
        x = torch.cat((x_b, x), axis=-1)

    for _ in range(p[1]):
        x_b = 2 * x[:, :, :, -1:] - x[:, :, :, -2:-1]
        x = torch.cat((x, x_b), axis=-1)

    for _ in range(p[2]):
        x_b = 2 * x[:, :, -1:, :] - x[:, :, -2:-1, :]
        x = torch.cat((x, x_b), axis=-2)

    for _ in range(p[3]):
        x_b = 2 * x[:, :, 0:1, :] - x[:, :, 1:2, :]
        x = torch.cat((x_b, x), axis=-2)

    return x


def eta_torch(gamma, beta, z, T, Tref=0, zref=0):
    """
    Calculates a temperature- and pressure-dependent viscosity field based on input parameters using Frank-Kamenetskii.

    Args:
        gamma (float): Viscosity contrast due to temperature.
        beta (float): Viscosity contrast due to pressure.
        z (float): Vertical distance parameter.
        T (float): Temperature value at a given point.
        Tref (float, optional): Reference temperature. Defaults to 0.
        zref (float, optional): Reference vertical distance. Defaults to 0.

    Returns:
        float: Viscosity field.
    """
    eta = torch.exp(torch.log(gamma) * (Tref - T) + torch.log(beta) * (z - zref))
    return eta


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model whose parameters are to be counted.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def exists(val):
    """
    Checks if a given value is not None.

    Args:
        val (any): The value to check.

    Returns:
        bool: True if the value is not None, False otherwise.
    """
    return val is not None


def get_lr(optimizer):
    """
    Retrieves the learning rate from a given optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer from which to extract the learning rate.

    Returns:
        float: The learning rate of the optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def pad_uvp(u, v, p=None):
    """
    Pads the velocity components u and v, and optionally the pressure field p, using replicate padding at boundaries.

    Args:
        u (Tensor): The velocity component in the x-direction.
        v (Tensor): The velocity component in the y-direction.
        p (Tensor, optional): The pressure field to be padded. Defaults to None.

    Returns:
        Tensor, Tensor, Tensor (optional): The padded velocity components u, v, and the optionally padded pressure field p.
    """
    u = F.pad(u, (0, 0, 1, 1), mode="replicate")
    u = torch.cat((-u[:, :, :, 0:1], u, -u[:, :, :, -1:]), axis=3)
    u[:, :, 0, 0] = 0.0
    u[:, :, 0, -1] = 0.0
    u[:, :, -1, 0] = 0.0
    u[:, :, -1, -1] = 0.0

    v = F.pad(v, (1, 1, 0, 0), mode="replicate")
    v = torch.cat((-v[:, :, 0:1, :], v, -v[:, :, -1:, :]), axis=2)
    v[:, :, 0, 0] = 0.0
    v[:, :, 0, -1] = 0.0
    v[:, :, -1, 0] = 0.0
    v[:, :, -1, -1] = 0.0

    if p is not None:
        p = F.pad(p, (1, 1, 1, 1), mode="replicate")
        p[:, :, 0, 0] = 0.0
        p[:, :, 0, -1] = 0.0
        p[:, :, -1, 0] = 0.0
        p[:, :, -1, -1] = 0.0

    return u, v, p


## ------
## taken from https://github.com/vc-bonn/Unsupervised_Deep_Learning_of_Incompressible_Fluid_Dynamics
dx_right_kernel = (
    torch.Tensor([0, -1, 1]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2)
)


def dx_right(v, device):
    return F.conv2d(v, dx_right_kernel.to(device))  # ,padding=(0,1))


dy_bottom_kernel = (
    torch.Tensor([0, -1, 1]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3)
)


def dy_bot(v, device):
    return F.conv2d(v, dy_bottom_kernel.to(device))  # ,padding=(1,0))


dx_left_kernel = (
    torch.Tensor([-1, 1, 0]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2)
)


def dx_left(v, device):
    return F.conv2d(v, dx_left_kernel.to(device))  # ,padding=(0,1))


dy_top_kernel = torch.Tensor([-1, 1, 0]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3)


def dy_top(v, device):
    return F.conv2d(v, dy_top_kernel.to(device))  # ,padding=(1,0))


dx_center_kernel = (
    torch.Tensor([-0.5, 0, 0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2)
)


def dx_center(v, device):
    return F.conv2d(v, dx_center_kernel.to(device))  # ,padding=(0,1))


dy_center_kernel = (
    torch.Tensor([-0.5, 0, 0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3)
)


def dy_center(v, device):
    return F.conv2d(v, dy_center_kernel.to(device))  # ,padding=(1,0))


du_dy_kernel = (
    torch.Tensor([1, -1, -1, 1]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3)
)


def du_dy(v, device):
    return F.conv2d(v, du_dy_kernel.to(device))


dv_dx_kernel = (
    torch.Tensor([1, -1, -1, 1]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2)
)


def dv_dx(v, device):
    return F.conv2d(v, dv_dx_kernel.to(device))


# laplace_kernel = 0.25*torch.Tensor([[1,2,1],[2,-12,2],[1,2,1]]).double().unsqueeze(0).unsqueeze(1) # isotropic 9 point stencil
laplace_kernel = (
    torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).double().unsqueeze(0).unsqueeze(1)
)  # isotropic 9 point stencil


def laplace(v, device):
    return F.conv2d(v, laplace_kernel.to(device))


## ------


class TS(nn.Module):
    """
    An evaluation wrapper for NewFluidNet and Unet for time-stepping.
    It incorporates a temperature field, velocity, and optionally advection components.

    Args:
        stokes (nn.Module): The Stokes solver model used for velocity and pressure predictions.
        ad (nn.Module): (optional) Advection model used for updating temperature fields.
        device (torch.device): The device (CPU or GPU) on which the computation will run.
        ts (int): Number of time steps to simulate.
        advection_scheme (int): Scheme used for advection. Defaults to 2.
        scale (bool): If True, scales the input variables.
        p_pred (bool): If True, predicts pressure in addition to velocity.
        net (str): Specifies which neural network model to use (e.g., "fluidnet", "newfluidnet").

    Methods:
        __eta_torch(gamma, beta, z, T, Tref=0, zref=0):
            Computes a temperature-dependent scaling factor based on input parameters.

        __unscale_var(x, raq, fkt, fkp, var):
            Unscales the velocity variables based on predefined scaling factors for uprev or vprev.

        forward(T_prev, sdf, sdf2, ycc, raq_nd, fkt_nd, fkp_nd, raq, fkt, fkp, xc, yc, u_prev=None, v_prev=None, dt=None):
            Performs a forward pass for the time-stepping procedure, iterating through time steps and updating
            the temperature and velocity fields, optionally using the Stokes solver and advection model.

            Args:
                T_prev (Tensor): Previous temperature field.
                sdf (Tensor): Signed distance field (not used in this code, might be for boundary conditions).
                sdf2 (Tensor): Another signed distance field (not used in this code).
                ycc (Tensor): Grid coordinates or other model-specific parameters.
                raq_nd, fkt_nd, fkp_nd (Tensor): Normalized simulation parameters.
                raq, fkt, fkp (Tensor): Unnormalized simulation parameters.
                xc, yc (Tensor): Grid coordinates.
                u_prev, v_prev (Tensor): Previous velocity components (used only if self.net == "unet").
                dt (Tensor): Time step size (used for advection scheme).

            Returns:
                dict: A dictionary of the temperature fields at each time step (x) and other computed values
                      like velocity (u, v), pressure (p), and scaling factor V.
                dict: A dictionary of time steps for each iteration (dts).
                Tensor: Velocity field u.
                Tensor: Velocity field v.
                Tensor: Pressure field p (if p_pred is True).
                Tensor: Scaling factor V for temperature.
    """

    def __init__(
        self,
        stokes,
        ad,
        device,
        ts=8,
        advection_scheme=2,
        scale=True,
        p_pred=True,
        net="fluidnet",
    ):
        super().__init__()

        self.stokes = stokes
        self.ad = ad
        self.ts = ts
        self.device = device
        self.advection_scheme = advection_scheme
        self.scale = scale
        self.p_pred = p_pred
        self.net = net

    @staticmethod
    def __eta_torch(gamma, beta, z, T, Tref=0, zref=0):
        eta = torch.exp(torch.log(gamma) * (Tref - T) + torch.log(beta) * (z - zref))
        return eta

    @staticmethod
    def __unscale_var(x, raq, fkt, fkp, var):
        if var == "uprev" or var == "vprev":
            scaler = (
                torch.exp(
                    (raq / 10) * 1.80167667
                    + torch.log(fkt) * 0.4330392
                    + torch.log(fkp) * -0.46052953
                )
                * 5
            )
            x *= scaler
        return x

    def forward(
        self,
        T_prev,
        sdf,
        sdf2,
        ycc,
        raq_nd,
        fkt_nd,
        fkp_nd,
        raq,
        fkt,
        fkp,
        xc,
        yc,
        u_prev=None,
        v_prev=None,
        dt=None,
    ):

        x = {}
        dts = {}
        x[0] = T_prev.to(self.device)

        for i in range(1, self.ts + 1):

            V = TS.__eta_torch(
                fkt.to(self.device),
                fkp.to(self.device),
                1.0 - ycc.to(self.device),
                x[i - 1],
                0,
                0,
            )

            if self.net == "newfluidnet" or self.net == "fluidnet":
                V = torch.clip(V, 1e-08, 1)
                inp = torch.cat(
                    (
                        xc.to(self.device) / 4.0,
                        yc.to(self.device) / 4.0,
                        torch.log10(V) / 8,
                        raq_nd.expand(1, 1, T_prev.shape[-2], T_prev.shape[-1]).to(
                            self.device
                        ),
                        fkt_nd.expand(1, 1, T_prev.shape[-2], T_prev.shape[-1]).to(
                            self.device
                        ),
                        fkp_nd.expand(1, 1, T_prev.shape[-2], T_prev.shape[-1]).to(
                            self.device
                        ),
                        x[i - 1],
                    ),
                    axis=1,
                )

                u, v, p = self.stokes(inp)

                u = TS.__unscale_var(u, raq, fkt, fkp, "uprev")
                v = TS.__unscale_var(v, raq, fkt, fkp, "vprev")

                u = u.view(-1, 1, 128, 506)
                v = v.view(-1, 1, 128, 506)
                if self.p_pred:
                    p = p.view(-1, 1, 128, 506)

            elif self.net == "unet":
                V = torch.log10(torch.clip(V, 1e-8, 1.0)) / 8.0
                inp = torch.cat(
                    (
                        xc.to(self.device) / 4.0,
                        yc.to(self.device) / 4.0,
                        dt.to(self.device),
                        raq_nd.expand(1, 1, u_prev.shape[-2], u_prev.shape[-1]).to(
                            self.device
                        ),
                        fkt_nd.expand(1, 1, u_prev.shape[-2], u_prev.shape[-1]).to(
                            self.device
                        ),
                        fkp_nd.expand(1, 1, u_prev.shape[-2], u_prev.shape[-1]).to(
                            self.device
                        ),
                        V.to(self.device),
                        x[i - 1].to(self.device),
                        u_prev.to(self.device),
                        v_prev.to(self.device),
                    ),
                    axis=1,
                )

                u, v, _, T = self.stokes(inp)
                u = u.view(1, 1, 128, 506)
                v = v.view(1, 1, 128, 506)
                x[i] = T.view(1, 1, 128, 506)
                x[i][:, :, 0, :] = 1
                x[i][:, :, -1, :] = 0
                x[i][:, :, :, 0:1] = x[i][:, :, :, 1:2]
                x[i][:, :, :, -1:] = x[i][:, :, :, -2:-1]
                p = None

            if self.ad is not None and self.net == "newfluidnet":
                inp = torch.cat(
                    (
                        u.to(self.device),
                        v.to(self.device),
                        x[i - 1].to(self.device),
                        torch.zeros_like(u.to(self.device)) + raq,
                        xc.to(self.device),
                        yc.to(self.device),
                    ),
                    axis=1,
                )

                t0 = time.time()
                x[i], dt = self.ad(inp)
                x[i][:, :, 0, :] = 1
                x[i][:, :, -1, :] = 0
                x[i][:, :, :, 0:1] = x[i][:, :, :, 1:2]
                x[i][:, :, :, -1:] = x[i][:, :, :, -2:-1]

                dts[i] = dt

        return x, dts, u, v, p, V


class ADNet(nn.Module):
    """
    ADNet: A neural network-based solver for explicitly solving advection-diffusion.

    This class implements a computational model for simulating the evolution of a temperature field
    governed by the advection-diffusion equation. It takes velocity, temperature, and other related
    inputs and computes the updated temperature field using finite-difference methods.

    Attributes:
        device (torch.device): The device on which computations will be performed (CPU or GPU).
        CN_max (float): The Courant–Friedrichs–Lewy (CFL) condition limit for time-stepping stability.

    Methods:
        forward(inputs, dt=None, T_prev=None):
            Computes the next time step of the temperature field based on the input velocity and
            temperature conditions.

            Args:
                inputs (torch.Tensor): A tensor containing input fields:
                    - `inputs[:,0:1,1:-1,1:-1]`: u-velocity component.
                    - `inputs[:,1:2,1:-1,1:-1]`: v-velocity component.
                    - `inputs[:,2:3,...]`: Initial temperature field (if T_prev is None).
                    - `inputs[:,3:4,1:-1,1:-1]`: RaQ_Ra term (Rayleigh number term for buoyancy-driven flow).
                    - `inputs[:,4:5,...]`: X-coordinates of the grid.
                    - `inputs[:,5:6,...]`: Y-coordinates of the grid.

                dt (float, optional): The time step size. If None, it is computed adaptively.
                T_prev (torch.Tensor, optional): The previous temperature field. If None, it is taken from `inputs`.

            Returns:
                torch.Tensor: The updated temperature field after one time step.
                float: The computed or given time step size.

    Notes:
        - The boundary conditions for the temperature field are enforced via padding.
        - The time step is determined based on CFL constraints for stability.
    """

    def __init__(self, device, r_p="zeros", CN_max=0.1):
        super().__init__()

        self.device = device
        self.CN_max = CN_max

    def forward(self, inputs, dt=None, T_prev=None):

        u = inputs[:, 0:1, 1:-1, 1:-1]
        v = inputs[:, 1:2, 1:-1, 1:-1]
        if T_prev is None:
            T_prev = inputs[:, 2:3, ...]
        RaQ_Ra = inputs[:, 3:4, 1:-1, 1:-1]
        xc = inputs[:, 4:5, ...]
        yc = inputs[:, 5:6, ...]

        xc[:, :, :, 0] = 0.0
        xc[:, :, :, -1] = 4.0
        yc[:, :, 0, :] = 0.0
        yc[:, :, -1, :] = 1.0

        dx_l = dx_left(xc, self.device)[..., 1:-1, :]
        dx_r = dx_right(xc, self.device)[..., 1:-1, :]
        dy_t = dy_top(yc, self.device)[..., 1:-1]
        dy_b = dy_bot(yc, self.device)[..., 1:-1]

        dT_l = dx_left(T_prev, self.device)[..., 1:-1, :]
        dT_r = dx_right(T_prev, self.device)[..., 1:-1, :]
        dT_t = dy_top(T_prev, self.device)[..., 1:-1]
        dT_b = dy_bot(T_prev, self.device)[..., 1:-1]

        dT_dx = (dT_l / dx_l) * (u > 0) + (dT_r / dx_r) * (u < 0)
        dT_dy = (dT_t / dy_t) * (v > 0) + (dT_b / dy_b) * (v < 0)

        T_laplace = (dT_r / dx_r - dT_l / dx_l) / (0.5 * dx_r + 0.5 * dx_l) + (
            dT_b / dy_b - dT_t / dy_t
        ) / (0.5 * dy_b + 0.5 * dy_t)

        if dt is None:
            dx_min = torch.amin(dx_l)
            uv_mag = torch.max(torch.amax(torch.abs(u)), torch.amax(torch.abs(v)))
            dt_advect = 0.5 * self.CN_max * dx_min / uv_mag
            dt_diffuse = 0.5 * ((dx_min * dx_min) ** 2) / (dx_min**2 + dx_min**2)
            dt = torch.min(dt_advect, dt_diffuse)

        T_prev = T_prev[..., 1:-1, 1:-1] + dt * (
            -u * dT_dx - v * dT_dy + T_laplace + RaQ_Ra
        )

        T_prev = F.pad(T_prev, (1, 1, 1, 1), mode="replicate")
        T_prev[:, :, 0, :] = 1.0
        T_prev[:, :, -1, :] = 0.0
        return T_prev, dt


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer. It performs a Fast Fourier Transform (FFT), applies a linear transformation in the frequency space,
    and then applies an Inverse Fast Fourier Transform (IFFT) to return the result to the physical space.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes1 (int): Number of Fourier modes in the first dimension.
        modes2 (int): Number of Fourier modes in the second dimension.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            4  # modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = 4  # modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cdouble
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cdouble
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cdouble,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralFluidLayer(nn.Module):
    """
    A fluid layer that incorporates a spectral convolution, followed by normalization
    and activation functions to process the input.

    Args:
        c_i (int): Number of input channels.
        c_o (int): Number of output channels.
        act_fn (str): Activation function to use (e.g., "selu", "relu", "gelu", etc.).
        r_p (str): Padding method (e.g., "zeros" or "learned").
        use_symm (bool): Whether to use symmetry in convolution.
        dilation (int): Dilation factor for the convolution.
        f (int): Kernel size for the convolution.
    """

    def __init__(
        self,
        c_i: int,
        c_o: int,
        act_fn: str = "selu",
        r_p="zeros",
        use_symm=False,
        dilation=1,
        f=3,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        if r_p == "zeros":
            self.r_p = "constant"
        else:
            self.r_p = r_p

        if act_fn == "selu":
            self.act = nn.SELU()
        elif act_fn == "sine":
            self.act = Sine(30.0)
        elif act_fn == "tanh":
            self.act = nn.Tanh()
        elif act_fn == "elu":
            self.act = nn.ELU()
        elif act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "relu":
            self.act = nn.ReLU()
        elif act_fn == "gelu":
            self.act = nn.GELU()

        h_s = int(c_o / 4)  # int(c_o/8)
        v_s = 0  # int(c_o/8)
        hv_s = 0  # int(c_o/4)

        self.layers.append(SpectralConv2d(c_i, c_o, c_o, c_o))
        self.layers.append(torch.nn.GroupNorm(int(c_o / 4), c_o))

    def forward(self, inputs):

        x = self.layers[0](inputs)
        x = self.layers[1](x)
        x = self.act(x)
        return x


class FluidLayer(nn.Module):
    """
    A fluid layer with a standard convolution (optionally learned boundary convolution),
    followed by normalization, activation, and dropout.

    Args:
        c_i (int): Number of input channels.
        c_o (int): Number of output channels.
        act_fn (str): Activation function to use (e.g., "selu", "relu", "gelu", etc.).
        r_p (str): Padding method (e.g., "zeros" or "learned").
        use_symm (bool): Whether to use symmetry in convolution.
        dilation (int): Dilation factor for the convolution.
        f (int): Kernel size for the convolution.
        drop_rate (float): Dropout rate for regularization.
    """

    def __init__(
        self,
        c_i: int,
        c_o: int,
        act_fn: str = "selu",
        r_p="zeros",
        use_symm=False,
        dilation=1,
        f=3,
        drop_rate=0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        if r_p == "zeros":
            self.r_p = "constant"
        else:
            self.r_p = r_p

        if act_fn == "selu":
            self.act = nn.SELU()
        elif act_fn == "sine":
            self.act = Sine(30.0)
        elif act_fn == "tanh":
            self.act = nn.Tanh()
        elif act_fn == "elu":
            self.act = nn.ELU()
        elif act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "relu":
            self.act = nn.ReLU()
        elif act_fn == "gelu":
            self.act = nn.GELU()

        self.dropout = torch.nn.Dropout(drop_rate)

        h_s = int(c_o / 4) if c_o > 4 else int(c_o / 2)  # int(c_o/8)
        v_s = 0  # int(c_o/8)
        hv_s = 0  # int(c_o/4)

        if r_p == "learned":
            self.layers.append(
                BoundaryLearnedConvolution2D(c_i, c_o, k=f, use_symm=use_symm)
            )
        else:
            if use_symm:
                self.layers.append(
                    SymmetricConv2d(
                        c_i,
                        c_o,
                        kernel_size=f,
                        padding="same",
                        dilation=dilation,
                        padding_mode=r_p,
                        symmetry={"h": h_s, "v": v_s, "hv": hv_s},
                    )
                )
            else:
                self.layers.append(
                    nn.Conv2d(
                        c_i,
                        c_o,
                        kernel_size=f,
                        padding="same",
                        dilation=dilation,
                        padding_mode=r_p,
                    )
                )

        self.layers.append(torch.nn.GroupNorm(int(c_o / min(4, c_o)), c_o))

    def forward(self, inputs, bc_x=1, bc_y=1):

        if self.r_p == "learned":
            x = self.layers[0](inputs, bc_x=bc_x, bc_y=bc_y)
        else:
            x = self.layers[0](inputs)
        x = self.layers[1](x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class BoundaryLearnedConvolution2D(nn.Module):
    """
    Taken from here https://geometry.cs.ucl.ac.uk/projects/2019/investigating-edge/
    A convolutional layer designed for handling boundaries by learning separate
    convolutional filters for edges and corners.

    This module applies separate convolution operations to the interior, edges,
    and corners of an input tensor, allowing for better handling of boundary conditions.
    The module can optionally enforce symmetry in the convolutional operations.

    Args:
        c_i (int): Number of input channels.
        c_o (int): Number of output channels.
        k (int): Kernel size for the convolutional layers.
        stride (int, optional): Stride for the convolution operation (default: 1).
        use_symm (bool, optional): Whether to use symmetric convolutions (default: False).

    Attributes:
        conv (nn.Conv2d or SymmetricConv2d): Main convolutional layer for the interior region.
        conv_top_left (nn.Conv2d or SymmetricConv2d): Convolutional layer for the top-left corner.
        conv_top_right (nn.Conv2d or SymmetricConv2d): Convolutional layer for the top-right corner.
        conv_bottom_left (nn.Conv2d or SymmetricConv2d): Convolutional layer for the bottom-left corner.
        conv_bottom_right (nn.Conv2d or SymmetricConv2d): Convolutional layer for the bottom-right corner.
        conv_top (nn.Conv2d or SymmetricConv2d): Convolutional layer for the top edge.
        conv_bottom (nn.Conv2d or SymmetricConv2d): Convolutional layer for the bottom edge.
        conv_left (nn.Conv2d or SymmetricConv2d): Convolutional layer for the left edge.
        conv_right (nn.Conv2d or SymmetricConv2d): Convolutional layer for the right edge.
        learnable_bias (nn.Parameter): Learnable bias added to the final output.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, c_i, height, width).
        bc_x (int, optional): Boundary condition adjustment along the x-axis (default: 1).
        bc_y (int, optional): Boundary condition adjustment along the y-axis (default: 1).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, c_o, height, width).

    Example:
        >>> layer = BoundaryLearnedConvolution2D(3, 16, k=3, use_symm=False)
        >>> x = torch.randn(1, 3, 32, 32)  # Batch of 1, 3 input channels, 32x32 image
        >>> output = layer(x)
        >>> print(output.shape)  # Expected output shape: (1, 16, 32, 32)
    """

    def __init__(self, c_i, c_o, k, stride=1, use_symm=False):
        super().__init__()
        self.c_i = c_i
        self.c_o = c_o
        self.k = k

        h_s = int(c_o / 4) if c_o > 4 else int(c_o / 2)  # int(c_o/8)
        v_s = 0  # int(c_o/8)
        hv_s = 0  # int(c_o/4)

        # Define convolution layers for corners and edges
        if not use_symm:
            self.conv = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv = SymmetricConv2d(
                c_i,
                c_o,
                k,
                bias=False,
                padding="valid",
                symmetry={"h": h_s, "v": v_s, "hv": hv_s},
            )

        if not use_symm:
            self.conv_top_left = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_top_left = SymmetricConv2d(
                c_i,
                c_o,
                k,
                bias=False,
                padding="valid",
                symmetry={"h": h_s, "v": v_s, "hv": hv_s},
            )

        if not use_symm:
            self.conv_top_right = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_top_right = SymmetricConv2d(
                c_i,
                c_o,
                k,
                bias=False,
                padding="valid",
                symmetry={"h": h_s, "v": v_s, "hv": hv_s},
            )

        if not use_symm:
            self.conv_bottom_left = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_bottom_left = SymmetricConv2d(
                c_i,
                c_o,
                k,
                bias=False,
                padding="valid",
                symmetry={"h": h_s, "v": v_s, "hv": hv_s},
            )

        if not use_symm:
            self.conv_bottom_right = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_bottom_right = SymmetricConv2d(
                c_i,
                c_o,
                k,
                bias=False,
                padding="valid",
                symmetry={"h": h_s, "v": v_s, "hv": hv_s},
            )

        if not use_symm:
            self.conv_top = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_top = SymmetricConv2d(
                c_i,
                c_o,
                k,
                bias=False,
                padding="valid",
                symmetry={"h": h_s, "v": v_s, "hv": hv_s},
            )

        if not use_symm:
            self.conv_bottom = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_bottom = SymmetricConv2d(
                c_i,
                c_o,
                k,
                bias=False,
                padding="valid",
                symmetry={"h": h_s, "v": v_s, "hv": hv_s},
            )

        if not use_symm:
            self.conv_left = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_left = SymmetricConv2d(
                c_i,
                c_o,
                k,
                bias=False,
                padding="valid",
                symmetry={"h": h_s, "v": v_s, "hv": hv_s},
            )

        if not use_symm:
            self.conv_right = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_right = SymmetricConv2d(
                c_i,
                c_o,
                k,
                bias=False,
                padding="valid",
                symmetry={"h": h_s, "v": v_s, "hv": hv_s},
            )

        # Define learnable bias (initialized to zeros)
        self.learnable_bias = nn.Parameter(torch.zeros(1, c_o, 1, 1))

    def forward(self, x, bc_x=1, bc_y=1):
        # Corners
        # pad = self.k + 1 if self.k == 5 else self.k
        pad_x = self.k + 1 + (bc_x - 1) if self.k == 5 else self.k + (bc_x - 1)
        pad_y = self.k + 1 + (bc_y - 1) if self.k == 5 else self.k + (bc_y - 1)

        top_left = x[:, :, :pad_y, :pad_x]  # Top-left corner
        top_left = self.conv_top_left(top_left)

        bottom_left = x[:, :, -pad_y:, :pad_x]  # Bottom-left corner
        bottom_left = self.conv_bottom_left(bottom_left)

        top_right = x[:, :, :pad_y, -pad_x:]  # Top-right corner
        top_right = self.conv_top_right(top_right)

        bottom_right = x[:, :, -pad_y:, -pad_x:]  # Bottom-right corner
        bottom_right = self.conv_bottom_right(bottom_right)

        # Edges
        top = x[:, :, :pad_y, :]  # Top edge
        top = self.conv_top(top)

        left = x[:, :, :, :pad_x]  # Left edge
        left = self.conv_left(left)

        bottom = x[:, :, -pad_y:, :]  # Bottom edge
        bottom = self.conv_bottom(bottom)

        right = x[:, :, :, -pad_x:]  # Right edge
        right = self.conv_right(right)

        x = self.conv(x)  # Apply main convolution

        x = torch.cat([left, x, right], dim=3)

        top = torch.cat([top_left, top, top_right], dim=3)
        bottom = torch.cat([bottom_left, bottom, bottom_right], dim=3)

        x = torch.cat([bottom, x, top], dim=2)

        # Add learnable bias
        x = x + self.learnable_bias  # Learnable bias

        return x


class NewFluidNet(nn.Module):
    """
    inspired by https://proceedings.mlr.press/v70/tompson17a.html
    A neural network for fluid simulation using convolutional layers with optional spectral convolutions,
    multi-level feature extraction, and various activation functions.

    Parameters:
    ----------
    levels : int
        Number of hierarchical levels in the network.
    c_i : int
        Number of input channels.
    c_h : int
        Number of hidden channels.
    c_o : int
        Number of output channels.
    device : torch.device
        Device to run the model on (CPU or GPU).
    act_fn : str, optional
        Activation function to use. Options: "selu", "sine", "tanh", "elu", "silu", "relu", "gelu". Default: "selu".
    r_p : str, optional
        Padding mode, can be "zeros" or other padding strategies. Default: "zeros".
    loss_type : str, optional
        Type of loss function, either "mae" (mean absolute error), "mass", or "curl". Default: "mae".
    use_symm : bool, optional
        Whether to use symmetric padding. Default: False.
    dilation : int, optional
        Dilation rate for convolutions. Default: 1.
    a_bound : float, optional
        Scaling factor for the curl-based loss. Default: 4.0.
    use_cosine : bool, optional
        Whether to use cosine-based normalization. Default: False.
    repeats : int, optional
        Number of convolutional repeats per level. Default: 3.
    use_skip : bool, optional
        Whether to use skip connections. Default: False.
    f : int, optional
        Kernel size for convolutions. Default: 3.
    p_pred : bool, optional
        Whether to predict pressure as an output. Default: True.
    spectral_conv : bool, optional
        Whether to use spectral convolution layers. Default: False.
    blurr : bool, optional
        Whether to apply blurring to outputs. Default: False.
    drop_rate : float, optional
        Dropout rate for convolutional layers. Default: 0.0.
    factor : int, optional
        Pooling factor for downsampling. Default: 2.

    Methods:
    -------
    forward(inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        Forward pass of the model. Computes velocity fields (u, v) and optionally pressure (p).
    """

    def __init__(
        self,
        levels: int,
        c_i: int,
        c_h: int,
        c_o: int,
        device,
        act_fn: str = "selu",
        r_p="zeros",
        loss_type="mae",
        use_symm=False,
        dilation=1,
        a_bound=4.0,
        use_cosine=False,
        repeats=3,
        use_skip=False,
        f=3,
        p_pred=True,
        spectral_conv=False,
        blurr=False,
        drop_rate=0.0,
        factor=2,
    ):
        super().__init__()

        self.conv = nn.ModuleList()
        self.gn = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.unpool = nn.ModuleList()
        self.levels = levels
        # self.device = device
        self.loss_type = loss_type
        self.a_bound = a_bound
        self.use_cosine = use_cosine
        self.repeats = repeats
        self.use_skip = use_skip
        self.p_pred = p_pred
        self.c_h = c_h
        self.c_i = c_i
        self.c_o = c_o
        if blurr:
            self.blurrer = (
                torch.Tensor([[1 / 9] * 3] * 3)
                .double()
                .unsqueeze(0)
                .unsqueeze(1)
                .to(device)
            )
        else:
            self.blurrer = None

        if r_p == "zeros":
            self.r_p = "constant"
        else:
            self.r_p = r_p

        if act_fn == "selu":
            self.act = nn.SELU()
        elif act_fn == "sine":
            self.act = Sine(30.0)
        elif act_fn == "tanh":
            self.act = nn.Tanh()
        elif act_fn == "elu":
            self.act = nn.ELU()
        elif act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "relu":
            self.act = nn.ReLU()
        elif act_fn == "gelu":
            self.act = nn.GELU()

        self.dx_center_kernel = (
            torch.Tensor([-0.5, 0, 0.5])
            .double()
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(2)
            .to(device)
        )
        self.dy_center_kernel = (
            torch.Tensor([-0.5, 0, 0.5])
            .double()
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(3)
            .to(device)
        )

        if spectral_conv:
            self.conv.append(
                SpectralFluidLayer(c_i, c_h, act_fn, r_p, use_symm, dilation, f=f)
            )
        else:
            self.conv.append(
                FluidLayer(
                    c_i, c_h, act_fn, r_p, use_symm, dilation, f=f, drop_rate=drop_rate
                )
            )

        xs = [128]
        ys = [506]

        self.pool = nn.AvgPool2d((factor, factor), stride=factor)
        for l in range(1, levels):
            self.unpool.append(
                nn.Upsample(size=(xs[0], ys[0]), mode="bicubic")
            )  # , align_corners=True

        self.convs = nn.ModuleList()

        for l in range(levels):
            self.convs.append(nn.ModuleList())
            for r in range(self.repeats):
                if spectral_conv:
                    self.convs[l].append(
                        SpectralFluidLayer(
                            c_h, c_h, act_fn, r_p, use_symm, dilation, f=f
                        )
                    )
                else:
                    self.convs[l].append(
                        FluidLayer(
                            c_h,
                            c_h,
                            act_fn,
                            r_p,
                            use_symm,
                            dilation,
                            f=f,
                            drop_rate=drop_rate,
                        )
                    )

        if self.loss_type == "curl":
            padding = (1, 1)
        else:
            padding = (1, 1)

        if self.r_p != "learned":
            self.conv.append(
                nn.Conv2d(
                    c_h * levels + c_i,
                    c_h,
                    kernel_size=3,
                    padding=padding,
                    dilation=dilation,
                    padding_mode=r_p,
                    stride=1,
                )
            )
        else:
            self.conv.append(
                BoundaryLearnedConvolution2D(
                    c_h * levels + c_i, c_h, k=f, use_symm=use_symm
                )
            )
        self.gn.append(torch.nn.GroupNorm(int(c_h / 4), c_h))

        if self.r_p != "learned":
            self.conv.append(
                nn.Conv2d(
                    c_h,
                    c_h,
                    kernel_size=3,
                    padding=(1, 1),
                    dilation=1,
                    padding_mode=r_p,
                    stride=1,
                )
            )
        else:
            self.conv.append(
                BoundaryLearnedConvolution2D(c_h, c_h, k=f, use_symm=use_symm)
            )

        if self.r_p != "learned":
            self.conv.append(
                nn.Conv2d(
                    c_h,
                    c_o,
                    kernel_size=3,
                    padding=(1, 1),
                    dilation=1,
                    padding_mode=r_p,
                    stride=1,
                )
            )
        else:
            self.conv.append(
                BoundaryLearnedConvolution2D(c_h, c_o, k=f, use_symm=use_symm)
            )

    def forward(self, inputs):

        x_in = self.conv[0](inputs)

        for l in range(self.levels):
            y1 = x_in
            for _ in range(l):
                y1 = self.pool(y1)
            for r in range(self.repeats):
                y1 = self.convs[l][r](y1)
            if l > 0:
                y1 = self.unpool[l - 1](y1)
                y = torch.cat((y, y1), axis=1)
            else:
                y = y1

        del x_in, y1
        y = torch.cat((y, inputs), axis=1)
        del inputs

        y = self.conv[1](y)
        y = self.gn[0](y)
        y = self.act(y)

        y = self.conv[2](y)
        y = self.act(y)

        y = self.conv[3](y)
        y = y - torch.mean(y, dim=(2, 3), keepdim=True)

        if self.loss_type == "mae" or self.loss_type == "mass":
            u = y[:, 0:1, ...]
            v = y[:, 1:2, ...]
            if self.p_pred:
                p = y[:, 2:3, ...]
            else:
                p = None
            del y

            return u[:, 0, ...], v[:, 0, ...], p

        elif self.loss_type == "curl":
            a = y[:, 0:1, ...] * self.a_bound

            if self.blurrer is not None:
                a = F.pad(a, (1, 1, 1, 1), mode="replicate")
                a = F.conv2d(a, self.blurrer)

            if self.p_pred:
                p = y[:, 1, ...]
            else:
                p = None
            del y

            u = F.conv2d(a, self.dy_center_kernel)[:, :, :, 1:-1]  # 126 x 506
            v = -F.conv2d(a, self.dx_center_kernel)[:, :, 1:-1, :]  # 128 x 504

            u = F.pad(u, (1, 1, 1, 1), mode="replicate")
            u[:, :, :, 0] = -u[:, :, :, 1]
            u[:, :, :, -1] = -u[:, :, :, -2]
            u[:, :, 0, 0] = 0
            u[:, :, 0, -1] = 0
            u[:, :, -1, 0] = 0
            u[:, :, -1, -1] = 0

            v = F.pad(v, (1, 1, 1, 1), mode="replicate")
            v[:, :, 0, :] = -v[:, :, 1, :]
            v[:, :, -1, :] = -v[:, :, -2, :]
            v[:, :, 0, 0] = 0
            v[:, :, 0, -1] = 0
            v[:, :, -1, 0] = 0
            v[:, :, -1, -1] = 0

            return u[:, 0, ...], v[:, 0, ...], p



class FluidNet(nn.Module):
    """
    inspired by https://proceedings.mlr.press/v70/tompson17a.html
    A neural network for fluid simulation using convolutional layers with optional spectral convolutions,
    multi-level feature extraction, and various activation functions.

    Parameters:
    ----------
    levels : int
        Number of hierarchical levels in the network.
    c_i : int
        Number of input channels.
    c_h : int
        Number of hidden channels.
    c_o : int
        Number of output channels.
    device : torch.device
        Device to run the model on (CPU or GPU).
    act_fn : str, optional
        Activation function to use. Options: "selu", "sine", "tanh", "elu", "silu", "relu", "gelu". Default: "selu".
    r_p : str, optional
        Padding mode, can be "zeros" or other padding strategies. Default: "zeros".
    loss_type : str, optional
        Type of loss function, either "mae" (mean absolute error), "mass", or "curl". Default: "mae".
    use_symm : bool, optional
        Whether to use symmetric padding. Default: False.
    dilation : int, optional
        Dilation rate for convolutions. Default: 1.
    a_bound : float, optional
        Scaling factor for the curl-based loss. Default: 4.0.
    use_cosine : bool, optional
        Whether to use cosine-based normalization. Default: False.
    repeats : int, optional
        Number of convolutional repeats per level. Default: 3.
    use_skip : bool, optional
        Whether to use skip connections. Default: False.
    f : int, optional
        Kernel size for convolutions. Default: 3.
    p_pred : bool, optional
        Whether to predict pressure as an output. Default: True.
    spectral_conv : bool, optional
        Whether to use spectral convolution layers. Default: False.
    blurr : bool, optional
        Whether to apply blurring to outputs. Default: False.
    drop_rate : float, optional
        Dropout rate for convolutional layers. Default: 0.0.
    factor : int, optional
        Pooling factor for downsampling. Default: 2.

    Methods:
    -------
    forward(inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        Forward pass of the model. Computes velocity fields (u, v) and optionally pressure (p).
    """

    def __init__(
        self,
        levels: int,
        c_i: int,
        c_h: int,
        c_o: int,
        device,
        act_fn: str = "selu",
        r_p="zeros",
        loss_type="mae",
        use_symm=False,
        dilation=1,
        a_bound=4.0,
        use_cosine=False,
        repeats=3,
        use_skip=False,
        f=3,
        p_pred=True,
        spectral_conv=False,
        blurr=False,
        drop_rate=0.0,
        factor=2,
    ):
        super().__init__()

        self.conv = nn.ModuleList()
        self.gn = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.unpool = nn.ModuleList()
        self.levels = levels
        # self.device = device
        self.loss_type = loss_type
        self.a_bound = a_bound
        self.use_cosine = use_cosine
        self.repeats = repeats
        self.use_skip = use_skip
        self.p_pred = p_pred
        self.c_h = c_h
        self.c_i = c_i
        self.c_o = c_o
        if blurr:
            self.blurrer = (
                torch.Tensor([[1 / 9] * 3] * 3)
                .double()
                .unsqueeze(0)
                .unsqueeze(1)
                .to(device)
            )
        else:
            self.blurrer = None

        if r_p == "zeros":
            self.r_p = "constant"
        else:
            self.r_p = r_p

        if act_fn == "selu":
            self.act = nn.SELU()
        elif act_fn == "sine":
            self.act = Sine(30.0)
        elif act_fn == "tanh":
            self.act = nn.Tanh()
        elif act_fn == "elu":
            self.act = nn.ELU()
        elif act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "relu":
            self.act = nn.ReLU()
        elif act_fn == "gelu":
            self.act = nn.GELU()

        self.dx_center_kernel = (
            torch.Tensor([-0.5, 0, 0.5])
            .double()
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(2)
            .to(device)
        )
        self.dy_center_kernel = (
            torch.Tensor([-0.5, 0, 0.5])
            .double()
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(3)
            .to(device)
        )

        if spectral_conv:
            self.conv.append(
                SpectralFluidLayer(c_i, c_h, act_fn, r_p, use_symm, dilation, f=f)
            )
        else:
            self.conv.append(
                FluidLayer(
                    c_i, c_h, act_fn, r_p, use_symm, dilation, f=f, drop_rate=drop_rate
                )
            )

        xs = [128]
        ys = [506]

        self.pool = nn.AvgPool2d((factor, factor), stride=factor)
        for l in range(1, levels):
            self.unpool.append(
                nn.Upsample(size=(xs[0], ys[0]), mode="bicubic")
            )  # , align_corners=True

        self.convs = nn.ModuleList()

        for l in range(levels):
            self.convs.append(nn.ModuleList())
            for r in range(self.repeats):
                if spectral_conv:
                    self.convs[l].append(
                        SpectralFluidLayer(
                            c_h, c_h, act_fn, r_p, use_symm, dilation, f=f
                        )
                    )
                else:
                    self.convs[l].append(
                        FluidLayer(
                            c_h,
                            c_h,
                            act_fn,
                            r_p,
                            use_symm,
                            dilation,
                            f=f,
                            drop_rate=drop_rate,
                        )
                    )

        if self.loss_type == "curl":
            padding = (2, 2)
        else:
            padding = (1, 1)

        if self.r_p != "learned":
            self.conv.append(
                nn.Conv2d(
                    c_h * levels + c_i,
                    c_h,
                    kernel_size=3,
                    padding=padding,
                    dilation=dilation,
                    padding_mode=r_p,
                    stride=1,
                )
            )
        else:
            self.conv.append(
                BoundaryLearnedConvolution2D(
                    c_h * levels + c_i, c_h, k=f, use_symm=use_symm
                )
            )
        self.gn.append(torch.nn.GroupNorm(int(c_h / 4), c_h))

        if self.r_p != "learned":
            self.conv.append(
                nn.Conv2d(
                    c_h,
                    c_h,
                    kernel_size=3,
                    padding=(1, 1),
                    dilation=1,
                    padding_mode=r_p,
                    stride=1,
                )
            )
        else:
            self.conv.append(
                BoundaryLearnedConvolution2D(c_h, c_h, k=f, use_symm=use_symm)
            )

        if self.r_p != "learned":
            self.conv.append(
                nn.Conv2d(
                    c_h,
                    c_o,
                    kernel_size=3,
                    padding=(1, 1),
                    dilation=1,
                    padding_mode=r_p,
                    stride=1,
                )
            )
        else:
            self.conv.append(
                BoundaryLearnedConvolution2D(c_h, c_o, k=f, use_symm=use_symm)
            )

    def forward(self, inputs):

        x_in = self.conv[0](inputs)

        for l in range(self.levels):
            y1 = x_in
            for _ in range(l):
                y1 = self.pool(y1)
            for r in range(self.repeats):
                y1 = self.convs[l][r](y1)
            if l > 0:
                y1 = self.unpool[l - 1](y1)
                y = torch.cat((y, y1), axis=1)
            else:
                y = y1

        del x_in, y1
        y = torch.cat((y, inputs), axis=1)
        del inputs

        if self.loss_type == "curl":
            y = self.conv[1](y, bc_x=2,bc_y=2)
        y = self.gn[0](y)
        y = self.act(y)

        y = self.conv[2](y)
        y = self.act(y)

        y = self.conv[3](y)
        y = y - torch.mean(y, dim=(2, 3), keepdim=True)

        if self.loss_type == "mae" or self.loss_type == "mass":
            u = y[:, 0:1, ...]
            v = y[:, 1:2, ...]
            if self.p_pred:
                p = y[:, 2:3, ...]
            else:
                p = None
            del y

            return u[:, 0, ...], v[:, 0, ...], p

        elif self.loss_type == "curl":
            a = y[:, 0:1, ...] * self.a_bound

            if self.blurrer is not None:
                a = F.pad(a, (1, 1, 1, 1), mode="replicate")
                a = F.conv2d(a, self.blurrer)

            if self.p_pred:
                p = y[:, 1, ...]
            else:
                p = None
            del y

            u = F.conv2d(a, self.dy_center_kernel)[:, :, :, 1:-1]  # 128 x 506
            v = -F.conv2d(a, self.dx_center_kernel)[:, :, 1:-1, :]  # 128 x 506

            return u[:, 0, ...], v[:, 0, ...], p


class Unet(nn.Module):
    """
    U-Net architecture for physics-based machine learning in mantle convection.

    This model is designed for fluid simulations with physics-informed constraints. It supports
    various activation functions, boundary conditions, spectral convolutions, and optional
    pressure prediction.

    Args:
        levels (int): Number of downsampling levels in the U-Net.
        c_i (int): Number of input channels.
        c_h (int): Number of hidden channels.
        c_o (int): Number of output channels.
        device (torch.device, optional): Device for model computations. Defaults to CPU.
        act_fn (str, optional): Activation function ("gelu", "relu", "sine", etc.). Defaults to "gelu".
        r_p (str, optional): Padding mode ("replicate", "zeros", etc.). Defaults to "replicate".
        loss_type (str, optional): Loss type, either "curl" or "mae". Defaults to "curl".
        use_symm (bool, optional): Whether to enforce symmetry in convolutions. Defaults to False.
        dilation (int, optional): Dilation factor for convolutions. Defaults to 1.
        a_bound (float, optional): Scaling factor for curl loss. Defaults to 10.0.
        use_cosine (bool, optional): Whether to use cosine similarity. Defaults to False.
        repeats (int, optional): Number of repeated convolutional layers at each level. Defaults to 2.
        use_skip (bool, optional): Whether to include skip connections. Defaults to False.
        f (int, optional): Kernel size for convolutions. Defaults to 5.
        p_pred (bool, optional): Whether to predict pressure. Defaults to False.
        spectral_conv (bool, optional): Whether to use spectral convolutions. Defaults to False.
        blurr (bool, optional): Apply Gaussian blur to curl loss. Defaults to False.
        drop_rate (float, optional): Dropout rate for regularization. Defaults to 0.0.

    Attributes:
        conv (nn.ModuleList): Convolutional layers at the initial level.
        gn (nn.ModuleList): Group normalization layers.
        pool (nn.Module): Downsampling (average pooling) layer.
        upconvs (nn.ModuleList): Convolutional layers for upsampling path.
        dx_center_kernel (torch.Tensor): Finite difference kernel for x-direction.
        dy_center_kernel (torch.Tensor): Finite difference kernel for y-direction.
        blurrer (nn.Module or None): Optional Gaussian blur filter.

    Methods:
        forward(inputs):
            Forward pass of the U-Net model.
            - Applies downsampling, convolutions, and upsampling.
            - Computes velocity (u, v) from curl if loss_type is "curl".
            - Returns velocity components, pressure (if p_pred=True), and temperature.

    Returns:
        If loss_type == "mae" or "mass":
            Tuple of (u, v, p, T) where:
            - u (torch.Tensor): x-velocity component.
            - v (torch.Tensor): y-velocity component.
            - p (torch.Tensor or None): Pressure field.
            - T (torch.Tensor): Temperature field.

        If loss_type == "curl":
            Tuple of (u, v, p, T) where:
            - u, v computed from curl operation.
            - p (torch.Tensor or None): Pressure field.
            - T (torch.Tensor): Temperature field.

    Example:
        >>> model = Unet(levels=3, c_i=4, c_h=64, c_o=3).double()
        >>> input_tensor = torch.randn(1, 4, 128, 128).double()
        >>> u, v, p, T = model(input_tensor)
    """

    def __init__(
        self,
        levels: int,
        c_i: int,
        c_h: int,
        c_o: int,
        device=torch.device("cpu"),
        act_fn: str = "gelu",
        r_p="replicate",
        loss_type="curl",
        use_symm=False,
        dilation=1,
        a_bound=10.0,
        use_cosine=False,
        repeats=2,
        use_skip=False,
        f=5,
        p_pred=False,
        spectral_conv=False,
        blurr=False,
        drop_rate=0.0,
    ):
        super().__init__()

        self.conv = nn.ModuleList()
        self.gn = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.unpool = nn.ModuleList()
        self.levels = levels
        self.loss_type = loss_type
        self.a_bound = a_bound
        self.use_cosine = use_cosine
        self.repeats = repeats
        self.use_skip = use_skip
        self.p_pred = p_pred
        if blurr:
            self.blurrer = v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))
        else:
            self.blurrer = None

        if r_p == "zeros":
            self.r_p = "constant"
        else:
            self.r_p = r_p

        if act_fn == "selu":
            self.act = nn.SELU()
        elif act_fn == "sine":
            self.act = Sine(30.0)
        elif act_fn == "tanh":
            self.act = nn.Tanh()
        elif act_fn == "elu":
            self.act = nn.ELU()
        elif act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "relu":
            self.act = nn.ReLU()
        elif act_fn == "gelu":
            self.act = nn.GELU()

        self.dx_center_kernel = (
            torch.Tensor([-0.5, 0, 0.5])
            .double()
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(2)
            .to(device)
        )
        self.dy_center_kernel = (
            torch.Tensor([-0.5, 0, 0.5])
            .double()
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(3)
            .to(device)
        )

        for r in range(self.repeats):
            if r == 0:
                c_in = c_i
            else:
                c_in = c_h
            if spectral_conv:
                self.conv.append(
                    SpectralFluidLayer(c_in, c_h, act_fn, r_p, use_symm, dilation, f=f)
                )
            else:
                self.conv.append(
                    FluidLayer(
                        c_in,
                        c_h,
                        act_fn,
                        r_p,
                        use_symm,
                        dilation,
                        f=f,
                        drop_rate=drop_rate,
                    )
                )

        self.pool = nn.AvgPool2d((2, 2), stride=2)
        # self.unpool = nn.Upsample(scale_factor=2, mode='bicubic')

        self.convs = nn.ModuleList()

        for l in range(1, levels):
            self.convs.append(nn.ModuleList())
            for r in range(self.repeats):
                if r == 0 and l > 1:
                    c_in = int(c_h / 2)
                else:
                    c_in = c_h

                if spectral_conv:
                    self.convs[-1].append(
                        SpectralFluidLayer(
                            c_in, c_h, act_fn, r_p, use_symm, dilation, f=f
                        )
                    )
                else:
                    self.convs[-1].append(
                        FluidLayer(
                            c_in,
                            c_h,
                            act_fn,
                            r_p,
                            use_symm,
                            dilation,
                            f=f,
                            drop_rate=drop_rate,
                        )
                    )
            c_h *= 2
        c_h = int(c_h / 2)

        self.upconvs = nn.ModuleList()
        for l in range(levels - 2, 0, -1):
            self.upconvs.append(nn.ModuleList())
            for r in range(self.repeats):
                if r == 0:
                    c_in = c_h + int(c_h / 2)
                    c_out = int(c_h / 2)
                else:
                    c_in = int(c_h / 2)
                    c_out = int(c_h / 2)

                if spectral_conv:
                    self.upconvs[-1].append(
                        SpectralFluidLayer(
                            c_in, c_out, act_fn, r_p, use_symm, dilation, f=f
                        )
                    )
                else:
                    self.upconvs[-1].append(
                        FluidLayer(
                            c_in,
                            c_out,
                            act_fn,
                            r_p,
                            use_symm,
                            dilation,
                            f=f,
                            drop_rate=drop_rate,
                        )
                    )

            c_h = int(c_h / 2)

        if self.r_p != "learned":
            self.conv.append(
                nn.Conv2d(
                    int(c_h * 2),
                    c_h,
                    kernel_size=f,
                    padding="same",
                    dilation=dilation,
                    padding_mode=r_p,
                    stride=1,
                )
            )
        else:
            self.conv.append(
                BoundaryLearnedConvolution2D(int(c_h * 2), c_h, k=f, use_symm=use_symm)
            )
        self.gn.append(torch.nn.GroupNorm(int(c_h / 4), c_h))

        if self.r_p != "learned":
            self.conv.append(
                nn.Conv2d(
                    c_h,
                    c_h,
                    kernel_size=f,
                    padding="same",
                    dilation=1,
                    padding_mode=r_p,
                    stride=1,
                )
            )
        else:
            self.conv.append(
                BoundaryLearnedConvolution2D(c_h, c_h, k=f, use_symm=use_symm)
            )

        if self.r_p != "learned":
            self.conv.append(
                nn.Conv2d(
                    c_h,
                    c_o,
                    kernel_size=f,
                    padding="same",
                    dilation=1,
                    padding_mode=r_p,
                    stride=1,
                )
            )
        else:
            self.conv.append(
                BoundaryLearnedConvolution2D(c_h, c_o, k=f, use_symm=use_symm)
            )

    def forward(self, inputs):

        x = {}
        s = {}

        if self.r_p != "learned":
            inputs = F.pad(inputs, (3, 3, 0, 0), mode=self.r_p)
        x[0] = inputs
        for r in range(self.repeats):
            if self.r_p == "learned" and r == 0:
                x[0] = self.conv[r](x[0], bc_x=4, bc_y=1)
            else:
                x[0] = self.conv[r](x[0])

        sizes = {}
        sizes[0] = (x[0].shape[-2], x[0].shape[-1])
        for l in range(1, self.levels):
            x[l] = self.pool(x[l - 1])
            sizes[l] = (x[l].shape[-2], x[l].shape[-1])
            for r in range(self.repeats):
                x[l] = self.convs[l - 1][r](x[l])
        xu = x[l]

        for l_i, l in enumerate(np.arange(self.levels - 2, 0, -1)):
            xu = nn.Upsample(size=sizes[l], mode="bicubic")(xu)
            xu = torch.cat((x[l], xu), dim=1)
            for r in range(self.repeats):
                xu = self.upconvs[l_i][r](xu)

        xu = nn.Upsample(size=sizes[0], mode="bicubic")(xu)
        y = torch.cat((xu, x[0]), axis=1)
        y = self.conv[-3](y)  # [:,:,:,3:-3])
        y = self.gn[0](y)
        y = self.act(y)

        y = self.conv[-2](y)
        y = self.act(y)

        y = self.conv[-1](y)
        y = (y - torch.mean(y, dim=(2, 3), keepdim=True))[..., 3:-3]

        if self.loss_type == "mae" or self.loss_type == "mass":
            u = y[:, 0:1, ...]
            v = y[:, 1:2, ...]
            T = y[:, 2:3, ...]
            if self.p_pred:
                p = y[:, 3:4, ...]
            else:
                p = None
            del y

            return u, v, p, T

        elif self.loss_type == "curl":
            a = y[:, 0:1, ...] * self.a_bound
            T = torch.clip(y[:, 1, ...], 0.0, 1.5)

            if self.blurrer is not None:
                a = self.blurrer(a)

            if self.p_pred:
                p = y[:, 2, ...]
            else:
                p = None
            del y

            u = F.conv2d(a, self.dy_center_kernel)[:, :, :, 1:-1]  # 126 x 506
            v = -F.conv2d(a, self.dx_center_kernel)[:, :, 1:-1, :]  # 128 x 504

            u = F.pad(u, (1, 1, 1, 1), mode="replicate")
            u[:, :, :, 0] = -u[:, :, :, 1]
            u[:, :, :, -1] = -u[:, :, :, -2]
            u[:, :, 0, 0] = 0
            u[:, :, 0, -1] = 0
            u[:, :, -1, 0] = 0
            u[:, :, -1, -1] = 0

            v = F.pad(v, (1, 1, 1, 1), mode="replicate")
            v[:, :, 0, :] = -v[:, :, 1, :]
            v[:, :, -1, :] = -v[:, :, -2, :]
            v[:, :, 0, 0] = 0
            v[:, :, 0, -1] = 0
            v[:, :, -1, 0] = 0
            v[:, :, -1, -1] = 0

            return u[:, 0, ...], v[:, 0, ...], p, T
