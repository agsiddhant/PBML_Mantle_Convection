import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

#WENO has bugs; use upwind for now

dx_right_kernel = torch.Tensor([0,-1,1]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2)
def dx_right(v, device):
	return F.conv2d(v,dx_right_kernel.to(device))#,padding=(0,1))

dy_bottom_kernel = torch.Tensor([0,-1,1]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3)
def dy_bot(v, device):
	return F.conv2d(v,dy_bottom_kernel.to(device))#,padding=(1,0))

dx_left_kernel = torch.Tensor([-1,1,0]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2)
def dx_left(v, device):
	return F.conv2d(v,dx_left_kernel.to(device))#,padding=(0,1))

dy_top_kernel = torch.Tensor([-1,1,0]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3)
def dy_top(v, device):
	return F.conv2d(v,dy_top_kernel.to(device))#,padding=(1,0))
    

class ADNetWENO(nn.Module):
    def __init__(self, device, r_p="zeros", CN_max=0.1):
        super().__init__()

        self.device     = device
        self.CN_max = CN_max

    def compute_weno_weights(self,stencils, epsilon=1e-6):
        """
        Compute WENO nonlinear weights for 5th-order scheme.
        Args:
            stencils: Tensor containing candidate stencils (shape: [batch, channel, Nx, Ny, 3]).
            epsilon: Small value to prevent division by zero (float).
        Returns:
            weights: Normalized nonlinear weights (same shape as stencils).
        """
        # Smoothness indicators (beta terms)
        beta = torch.zeros_like(stencils)
        beta[..., 0] = 13/12 * (stencils[..., 0] - 2 * stencils[..., 1] + stencils[..., 2])**2 \
                       + 1/4 * (stencils[..., 0] - 4 * stencils[..., 1] + 3 * stencils[..., 2])**2
        beta[..., 1] = 13/12 * (stencils[..., 1] - 2 * stencils[..., 2] + stencils[..., 3])**2 \
                       + 1/4 * (stencils[..., 1] - stencils[..., 3])**2
        beta[..., 2] = 13/12 * (stencils[..., 2] - 2 * stencils[..., 3] + stencils[..., 4])**2 \
                       + 1/4 * (3 * stencils[..., 2] - 4 * stencils[..., 3] + stencils[..., 4])**2
    
        # Linear weights
        gamma = torch.tensor([0.1, 0.6, 0.3], device=stencils.device, dtype=stencils.dtype)
    
        # Nonlinear weights
        alpha = gamma / (epsilon + beta)**2
        weights = alpha / alpha.sum(dim=-1, keepdim=True)
        return weights
    
    def compute_flux_weno5(self,u, dx):
        """
        Compute 5th-order WENO flux for advection using convolutions.
        Args:
            u: Input scalar field (2D tensor, shape: [Nx, Ny]).
            dx: Grid spacing (float).
        Returns:
            flux: WENO-5 flux (2D tensor, shape: [Nx, Ny]).
        """
        u = u.unsqueeze(0).unsqueeze(0)  # Reshape for convolution
    
        # Define stencils for flux calculations
        stencil_kernels = torch.tensor([[-1, 7, -15, 19, -11, 2]], dtype=torch.float32) / (12 * dx)
        stencil_kernels = stencil_kernels.view(1, 1, 1, -1)

        print(u.shape)
        # Compute candidate stencils
        stencils = F.conv2d(F.pad(u, (3, 3, 0, 0), mode='replicate'), stencil_kernels, stride=1)
    
        # Split into 5 stencils for WENO weights
        stencils = torch.cat([stencils[..., i:i+3] for i in range(3)], dim=-1)
    
        # Compute WENO weights
        weights = self.compute_weno_weights(stencils)
    
        # Compute flux
        flux = (weights * stencils).sum(dim=-1)
    
        return flux.squeeze(0).squeeze(0)
    
    def compute_high_order_diffusion(self,u, dx, dy, nu=1):
        """
        Compute higher-order diffusion term using convolutional stencils.
        Args:
            u: Input scalar field (2D tensor, shape: [Nx, Ny]).
            dx: Grid spacing in x-direction (float).
            dy: Grid spacing in y-direction (float).
            nu: Diffusion coefficient (float).
        Returns:
            diffusion: Higher-order diffusion term (2D tensor, shape: [Nx, Ny]).
        """
        kernel_x = torch.tensor([[1, -4, 6, -4, 1]], dtype=torch.float64).view(1, 1, 1, 5) / dx**4
        kernel_y = kernel_x.transpose(-1, -2)

        print(kernel_x)
        print(kernel_y)
    
        u_pad_x = F.pad(u, (2,2, 0, 0), mode='replicate')
        u_pad_y = F.pad(u, (0, 0, 2,2), mode='replicate')
    
        d4u_dx4 = F.conv2d(u_pad_x, kernel_x, stride=1)
        d4u_dy4 = F.conv2d(u_pad_y, kernel_y, stride=1)
    
        return nu * (d4u_dx4 + d4u_dy4)
    
    def forward(self, inputs, dt=None, T=None, dx=1/126, dy=1/126):

        u      = inputs[:,0:1,1:-1,1:-1]
        v      = inputs[:,1:2,1:-1,1:-1]
        if T is None:
            T = inputs[:,2:3,:,:]
        RaQ_Ra = inputs[:,3:4,1:-1,1:-1]

        #flux_x = self.compute_flux_weno5(T, dx)  # Flux in x
        #flux_y = self.compute_flux_weno5(T.T, dy).T  # Flux in y

        dT_dx_l = dx_left(T,self.device)[...,1:-1,:]
        dT_dx_r = dx_right(T,self.device)[...,1:-1,:]
        dT_dy_t = dy_top(T,self.device)[...,:,1:-1]
        dT_dy_b = dy_bot(T,self.device)[...,:,1:-1]

        flux_x = dT_dx_l/dx* (u>0) + dT_dx_r/dx * (u<0)
        flux_y = dT_dy_t/dx* (v>0) + dT_dy_b/dx * (v<0)

        # Compute higher-order diffusion term
        diffusion = self.compute_high_order_diffusion(T, dx, dy)[:,:,1:-1,1:-1]

        if dt is None:
            dx_min = min(dx,dy)
            uv_mag = torch.max(torch.amax(torch.abs(u)), torch.amax(torch.abs(v)))
            dt_advect = 0.5*self.CN_max * dx_min/uv_mag   #
            dt_diffuse = 0.5*((dx_min*dx_min)**2)/(dx_min**2 + dx_min**2)   #dx**2/1e-6   0.5*
            dt = torch.min(dt_advect,dt_diffuse)
            
        # Update scalar field
        T = T[:,:,1:-1,1:-1] - dt * (u * flux_x + v * flux_y) + dt * (diffusion + RaQ_Ra)
        T = F.pad(T, (1,1,1,1), mode="replicate")
        T[:,:,0,:]  = 1.
        T[:,:,-1,:] = 0.
        return T, dt