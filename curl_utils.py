import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_trubitsyn(x, y, eta, deta_dy, Ra=10, T0=0.1):
    
    v = Ra * T0 / (4 * np.pi) * np.sin(np.pi * y) * np.cos(np.pi * x)
    u = -Ra * T0 / (4 * np.pi) * np.cos(np.pi * y) * np.sin(np.pi * x)
    p = -Ra * (
        (y**2 / 2 - y + 1 / 2)
        + T0/ (2 * np.pi)
        * (eta * np.cos(np.pi * y * np.cos(np.pi * x) + eta[:,:,-1:, 0:1]))
    )
    T = (
        1
        - y
        + T0
        * np.cos(np.pi * x)
        * (eta * np.sin(np.pi * y) - (1 / np.pi) * deta_dy * np.cos(np.pi * y))
    )
    return u, v, p, T


class GEGLU(nn.Module):
    def forward(self, x, num_splits=2):
        x, gate = x.chunk(num_splits, dim=1)
        return x * F.gelu(gate)


class Net(nn.Module):
    def __init__(
        self,
        device,
        padding_mode="zeros",
        c_h=8,
        layers=2,
        k=5,
        conv="normal",
        act="geglu",
        c_i = 4,
        c_o = 1
    ):
        super().__init__()

        

        if act == "geglu":
            self.act = GEGLU()
            c_h_o = int(c_h * 2)
        else:
            self.act = F.gelu
            c_h_o = c_h

        self.convs = nn.ModuleList()
        init_pad = 2 if k == 5 else 1
        if conv == "normal":
            self.convs.append(
                nn.Conv2d(
                    c_i,
                    c_h_o,
                    k,
                    padding=(init_pad, init_pad),
                    padding_mode=padding_mode,
                )
            )
        else:
            self.convs.append(BoundaryLearnedConvolution2D(c_i, c_h_o, k))

        for l in range(layers):
            if conv == "normal":
                self.convs.append(
                    nn.Conv2d(c_h, c_h_o, k, padding="same", padding_mode=padding_mode)
                )
            else:
                self.convs.append(BoundaryLearnedConvolution2D(c_h, c_h_o, k))

        if conv == "normal":
            self.convs.append(
                nn.Conv2d(c_h, c_o, 3, padding="same", padding_mode=padding_mode)
            )
        else:
            self.convs.append(BoundaryLearnedConvolution2D(c_h, c_o, k))

        self.dx = (
            torch.Tensor([-0.5, 0, 0.5])
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(2)
            .float()
            .to(device)
        )
        self.dy = (
            torch.Tensor([-0.5, 0, 0.5])
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(3)
            .float()
            .to(device)
        )

    def forward(self, x):
        u_t = x[:, 2:3, ...]
        v_t = x[:, 3:4, ...]

        maxs = torch.amax(u_t, axis=0)

        for c in self.convs[:-1]:
            x = self.act(c(x))
        x = self.convs[-1](x)

        x = x - torch.mean(x)

        u = F.conv2d(x, self.dy)[:, :, :, 1:-1]
        v = -F.conv2d(x, self.dx)[:, :, 1:-1, :]

        u = F.pad(u, (1, 1, 1, 1), mode="replicate")
        v = F.pad(v, (1, 1, 1, 1), mode="replicate")

        u[:, :, :, 0] = -u[:, :, :, 1]
        u[:, :, :, -1] = -u[:, :, :, -2]
        v[:, :, 0, :] = -v[:, :, 1, :]
        v[:, :, -1, :] = -v[:, :, -2, :]

        u[:, :, 0, 0] = 0
        u[:, :, 0, -1] = 0
        u[:, :, -1, 0] = 0
        u[:, :, -1, -1] = 0

        v[:, :, 0, 0] = 0
        v[:, :, 0, -1] = 0
        v[:, :, -1, 0] = 0
        v[:, :, -1, -1] = 0

        mass_t = (
            F.conv2d(u_t, self.dx)[:, :, 1:-1, :]
            + F.conv2d(v_t, self.dy)[:, :, :, 1:-1]
        )

        mass = F.conv2d(u, self.dx)[:, :, 1:-1, :] + F.conv2d(v, self.dy)[:, :, :, 1:-1]

        return u, v, mass, mass_t
    
    
class FluidNet(nn.Module):
    def __init__(
        self,
        device,
        padding_mode="zeros",
        c_h=8,
        layers=5,
        k=5,
        conv="normal",
        act="geglu",
        c_i = 4,
        c_o = 2,
        x_n=128,
        y_n=128,
        repeats=6
    ):
        super().__init__()

        if act == "geglu":
            self.act = GEGLU()
            c_h_o = int(c_h * 2)
        else:
            self.act = F.gelu
            c_h_o = c_h

        self.pool   = nn.AvgPool2d(2, stride=2)
        self.unpool = nn.Upsample(size=(x_n,y_n), mode='bilinear') 
        
        self.convs = nn.ModuleList()
        self.convs1 = nn.ModuleList()
        self.repeats = repeats
        self.layers = layers
        
        if conv == "normal":
            self.convs.append(
                nn.Conv2d(c_i, c_h_o, k, padding="same", padding_mode=padding_mode)
            )
        else:
            self.convs.append(BoundaryLearnedConvolution2D(c_i, c_h_o, k))

        for l in range(layers):
            self.convs1.append(nn.ModuleList())
            for r in range(repeats):
                if conv == "normal":
                    self.convs1[-1].append(
                        nn.Conv2d(c_h, c_h_o, k, padding="same", padding_mode=padding_mode)
                    )
                else:
                    self.convs1[-1].append(BoundaryLearnedConvolution2D(c_h, c_h_o, k))

        for i in range(3):
            c_i_o = int(c_h*layers)+c_i if i==0 else c_h
            c_h_o = c_o if i==2 else c_h
            if conv == "normal":
                self.convs.append(
                    nn.Conv2d(c_i_o, c_h_o, 3, padding="same", padding_mode=padding_mode)
                )
            else:
                self.convs.append(BoundaryLearnedConvolution2D(c_i_o, c_h_o, k))

        self.dx = (
            torch.Tensor([-0.5, 0, 0.5])
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(2)
            .float()
            .to(device)
        )
        self.dy = (
            torch.Tensor([-0.5, 0, 0.5])
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(3)
            .float()
            .to(device)
        )

    def forward(self, x):
        u_t = x[:, 2:3, ...]
        v_t = x[:, 3:4, ...]

        inputs = x

        x = self.act(self.convs[0](x))
        
        y = x
        for r in range(self.repeats):
            y = self.act(self.convs1[0][r](y))
        
        for l in range(1,self.layers):
            yy = self.pool(x)
            for _ in range(1,l):
                yy = self.pool(yy)
            for r in range(self.repeats):
                yy = self.act(self.convs1[l][r](yy))
            yy = self.unpool(yy)
            y = torch.cat((y, yy), axis=1)
            
        y = torch.cat((inputs,y), axis=1)
        del inputs, x 
        
        y = self.act(self.convs[1](y))
        y = self.act(self.convs[2](y))
        y = self.convs[-1](y)

        a = y[:,0:1,...]
        #p = y[:,1:2,...]
        a = a - torch.mean(a)
        #p = p - torch.mean(p)

        u = F.conv2d(a, self.dy)[:, :, :, 1:-1]
        v = -F.conv2d(a, self.dx)[:, :, 1:-1, :]

        u = F.pad(u, (1, 1, 1, 1), mode="replicate")
        v = F.pad(v, (1, 1, 1, 1), mode="replicate")

        u[:, :, :, 0]  = 0
        u[:, :, :, -1] = 0
        v[:, :, 0, :]  = 0
        v[:, :, -1, :] = 0

        mass_t = (
            F.conv2d(u_t, self.dx)[:, :, 1:-1, :]
            + F.conv2d(v_t, self.dy)[:, :, :, 1:-1]
        )

        mass = F.conv2d(u, self.dx)[:, :, 1:-1, :] + F.conv2d(v, self.dy)[:, :, :, 1:-1]

        return u, v, None, mass, mass_t


class BoundaryLearnedConvolution2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Define convolution layers for corners and edges
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="valid",
            bias=False,
        )

        self.conv_top_left = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="valid",
            bias=False,
        )
        self.conv_top_right = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="valid",
            bias=False,
        )
        self.conv_bottom_left = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="valid",
            bias=False,
        )
        self.conv_bottom_right = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="valid",
            bias=False,
        )

        self.conv_top = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="valid",
            bias=False,
        )
        self.conv_bottom = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="valid",
            bias=False,
        )
        self.conv_left = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="valid",
            bias=False,
        )
        self.conv_right = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="valid",
            bias=False,
        )

        # Define learnable bias (initialized to zeros)
        self.learnable_bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        # Corners
        pad = self.kernel_size + 1 if self.kernel_size == 5 else self.kernel_size
        top_left = x[:, :, :pad, :pad]  # Top-left corner
        top_left = self.conv_top_left(top_left)

        bottom_left = x[:, :, -pad:, :pad]  # Bottom-left corner
        bottom_left = self.conv_bottom_left(bottom_left)

        top_right = x[:, :, :pad, -pad:]  # Top-right corner
        top_right = self.conv_top_right(top_right)

        bottom_right = x[:, :, -pad:, -pad:]  # Bottom-right corner
        bottom_right = self.conv_bottom_right(bottom_right)

        # Edges
        top = x[:, :, :pad, :]  # Top edge
        top = self.conv_top(top)

        left = x[:, :, :, :pad]  # Left edge
        left = self.conv_left(left)

        bottom = x[:, :, -pad:, :]  # Bottom edge
        bottom = self.conv_bottom(bottom)

        right = x[:, :, :, -pad:]  # Right edge
        right = self.conv_right(right)

        x = self.conv(x)  # Apply main convolution

        x = torch.cat([left, x, right], dim=3)

        top = torch.cat([top_left, top, top_right], dim=3)
        bottom = torch.cat([bottom_left, bottom, bottom_right], dim=3)

        x = torch.cat([bottom, x, top], dim=2)

        # Add learnable bias
        x = x + self.learnable_bias  # Learnable bias

        return x


def make_plot(u, u_p, v, v_p, x, y, p=None, p_p=None):
    
    vars_t = [u,v,p] if p is not None else [u,v]
    vars_p = [u_p,v_p,p_p] if p is not None else [u_p,v_p]
    
    fig, axs = plt.subplots(nrows=len(vars_t), ncols=3, figsize=(10, 2*len(vars_t)))

    for i in range(len(vars_t)):
        cax = axs[i, 0].contour(x, y, vars_t[i])
        fig.colorbar(cax)

        cax = axs[i, 1].contour(x, y, vars_p[i][0, 0, ...])
        fig.colorbar(cax)

        cax = axs[i, 2].contour(x, y, vars_t[i] - vars_p[i][0, 0, ...])
        fig.colorbar(cax)

    plt.tight_layout()
    plt.show()

    colors = ["b", "g", "r"]
    colors_p = ["k", "m", "y"]
    fig, axs = plt.subplots(nrows=len(vars_t), ncols=4, figsize=(12, int(3*len(vars_t))))

    for i in range(len(vars_t)):
        for ind in [0, 1, 2]:
            axs[i, 0].plot(np.arange(8), vars_t[i][:8, 0 + ind], colors[ind] + "-")
            axs[i, 0].plot(
                np.arange(8), vars_p[i][0, 0, ...][:8, 0 + ind], colors_p[ind] + "--"
            )
            axs[i, 1].plot(np.arange(8), vars_t[i][-1 - ind, :8], "b-")
            axs[i, 1].plot(
                np.arange(8), vars_p[i][0, 0, ...][-1 - ind, :8], colors_p[ind] + "--"
            )
            axs[i, 2].plot(np.arange(8), vars_t[i][:8, -1 - ind][::-1], "b-")
            axs[i, 2].plot(
                np.arange(8), vars_p[i][0, 0, ...][:8, -1 - ind][::-1], colors_p[ind] + "--"
            )
            axs[i, 3].plot(np.arange(8), vars_t[i][0 + ind, :8][::-1], "b-")
            axs[i, 3].plot(
                np.arange(8), vars_p[i][0, 0, ...][0 + ind, :8][::-1], colors_p[ind] + "--"
            )

            axs[i, 0].plot(np.arange(8, 16), vars_t[i][-8:, 0 + ind], colors[ind] + "-")
            axs[i, 0].plot(
                np.arange(8, 16), vars_p[i][0, 0, ...][-8:, 0 + ind], colors_p[ind] + "--"
            )
            axs[i, 1].plot(np.arange(8, 16), vars_t[i][-1 - ind, -8:], "b-")
            axs[i, 1].plot(
                np.arange(8, 16), vars_p[i][0, 0, ...][-1 - ind, -8:], colors_p[ind] + "--"
            )
            axs[i, 2].plot(np.arange(8, 16), vars_t[i][-8:, -1 - ind][::-1], "b-")
            axs[i, 2].plot(
                np.arange(8, 16),
                vars_p[i][0, 0, ...][-8:, -1 - ind][::-1],
                colors_p[ind] + "--",
            )
            axs[i, 3].plot(np.arange(8, 16), vars_t[i][0 + ind, -8:][::-1], "b-")
            axs[i, 3].plot(
                np.arange(8, 16),
                vars_p[i][0, 0, ...][0 + ind, -8:][::-1],
                colors_p[ind] + "--",
            )

    plt.tight_layout()
    plt.show()
