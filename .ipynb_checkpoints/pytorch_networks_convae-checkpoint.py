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
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
    
def get_mass(u,v,bc=False):
    u = u.view(-1,1,128,506)
    v = v.view(-1,1,128,506)
    du_dx = F.conv2d(u, dx_center_kernel)[...,1:-1,:]
    dv_dy = F.conv2d(v, dy_center_kernel)[...,:,1:-1]

    if bc:
        du_dx[:,:,:,0]  *= 2./1.5
        du_dx[:,:,:,-1] *= 2./1.5
    
        dv_dy[:,:,0,:]  *= 2./1.5
        dv_dy[:,:,-1,:] *= 2./1.5

    return du_dx + dv_dy
    
def pad_grad(x, p=(1,1,1,1)):
    for _ in range(p[0]):
        x_b = 2*x[:,:,:,0:1] - x[:,:,:,1:2]
        x = torch.cat((x_b, x), axis=-1)

    for _ in range(p[1]):
        x_b = 2*x[:,:,:,-1:] - x[:,:,:,-2:-1]
        x = torch.cat((x, x_b), axis=-1)

    for _ in range(p[2]):
        x_b = 2*x[:,:,-1:,:] - x[:,:,-2:-1,:]
        x = torch.cat((x, x_b), axis=-2)

    for _ in range(p[3]):
        x_b = 2*x[:,:,0:1,:] - x[:,:,1:2,:]
        x = torch.cat((x_b, x), axis=-2)
    
    return x
    
def eta_torch(gamma, beta, z, T, Tref=0, zref=0):
    eta = torch.exp( torch.log(gamma)*(Tref-T) + torch.log(beta)*(z-zref) )
    return eta
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def exists(val):
    return val is not None

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def pad_uvp(u,v,p=None):
    u = F.pad(u, (0,0,1,1), mode="replicate")
    u = torch.cat((-u[:,:,:,0:1], u, -u[:,:,:,-1:]), axis=3)
    u[:,:,0,0]   = 0.
    u[:,:,0,-1]  = 0.
    u[:,:,-1,0]  = 0.
    u[:,:,-1,-1] = 0.

    v = F.pad(v, (1,1,0,0), mode="replicate")
    v = torch.cat((-v[:,:,0:1,:], v, -v[:,:,-1:,:]), axis=2)
    v[:,:,0,0]   = 0.
    v[:,:,0,-1]  = 0.
    v[:,:,-1,0]  = 0.
    v[:,:,-1,-1] = 0.

    if p is not None:
        p = F.pad(p, (1,1,1,1), mode="replicate")
        p[:,:,0,0]   = 0.
        p[:,:,0,-1]  = 0.
        p[:,:,-1,0]  = 0.
        p[:,:,-1,-1] = 0.

    return u,v,p

def one_epoch_AD(model_uvp, model_T, epoch, loader, loader_init, optimizer, device, loss_type, 
                 is_train=False, weigh_loss=True, p_pred=True, loss_scale=False):
    
    running_loss    = [0., 0., 0., 0., 0.]
    
    counter = 1
    loss_fn = torch.nn.L1Loss() 

    for i, data in enumerate(loader):
        t0 = time.time()
        #optimizer.zero_grad()

        if is_train:
            for param in model_uvp.parameters():
                param.grad = None

        if loader_init is not None:
            data_init = next(iter(loader_init))
            gVTp     = torch.cat((data[0], data_init[0]), axis=0).to(device)
            uvp      = torch.cat((data[1], data_init[1]), axis=0).to(device)
            scaler   = torch.cat((data[3], data_init[3]), axis=0).to(device)
        else:
            gVTp     = data[0].to(device)
            uvp      = data[1].to(device)
            scaler   = data[3].to(device)

        r      = torch.randperm(gVTp.shape[0])
        gVTp   = gVTp[r, ...]
        uvp    = uvp[r, ...]
        scaler = scaler[r, ...].view(-1,1,1,1)

        #t_weight = torch.cat((data[2].view(1,-1,1,1), data_init[2].view(1,-1,1,1)), axis=0).to(device)
        #print(uvp.shape)
        for t_step in range(1):
            #inp = torch.cat((gVTp,
            #                 paras.expand(-1,-1,gVTp.shape[-2],gVTp.shape[-1]),
            #                 ), axis=1)

            if loss_type == "curl":
                u,v,p = model_uvp(gVTp)
                #,a,u_u_bc,v_v_bc,du_dy_bc,dv_dx_bc 
            else:
                u,v,p  = model_uvp(gVTp)

        channel_last = uvp.shape[1]==64768
        if channel_last:
            uvp = uvp.view(uvp.shape[0],128,506,uvp.shape[-1])
            u_true = uvp[:,1:-1,1:-1,0]
            v_true = uvp[:,1:-1,1:-1,1]
            if p_pred:
                p_true = uvp[:,1:-1,1:-1,2]
        else:
            u_true = uvp[:,0,...]
            v_true = uvp[:,1,...]
            if p_pred:
                p_true = uvp[:,2,...]
            else:
                p_true = None

        u_max = torch.amax(u_true, dim=(1,2), keepdim=True)
        v_max = torch.amax(v_true, dim=(1,2), keepdim=True)
        u_min = torch.amin(u_true, dim=(1,2), keepdim=True)
        v_min = torch.amin(v_true, dim=(1,2), keepdim=True)
        
        loss_u = loss_fn(u_true, u) 
        loss_v = loss_fn(v_true, v)
        
            
        if p_pred:
            loss_p = loss_fn(p_true, p)
        else:
            loss_p = torch.tensor(0, dtype=torch.float64)

        u = u.view(-1,1,128,512)
        v = v.view(-1,1,128,512)
        u_true = u_true.view(-1,1,128,512)
        v_true = v_true.view(-1,1,128,512)

        if model_T is not None:
            raq_nd = gVTp[:,3:4,...].to(device)
            raq    = raq_nd * (9.70723344-0.12624371) + 0.12624371
            T_prev = gVTp[:,6:7,...].to(device)
            
            inp = torch.cat((
                             (u*scaler).to(device),
                             (v*scaler).to(device),
                             T_prev.to(device),
                             torch.zeros_like(u.to(device))+raq), axis=1)
            
            T_new_pred, dt = model_T(inp)
            T_new_pred[:,:,0,:]   = 1
            T_new_pred[:,:,-1,:]  = 0
            T_new_pred[:,:,:,0:1] = T_new_pred[:,:,:,1:2]
            T_new_pred[:,:,:,-1:] = T_new_pred[:,:,:,-2:-1]

            inp = torch.cat((
                             (u_true*scaler).to(device),
                             (v_true*scaler).to(device),
                             T_prev.to(device),
                             torch.zeros_like(u.to(device))+raq), axis=1)
            
            T_new, dt = model_T(inp)
            T_new[:,:,0,:]   = 1
            T_new[:,:,-1,:]  = 0
            T_new[:,:,:,0:1] = T_new[:,:,:,1:2]
            T_new[:,:,:,-1:] = T_new[:,:,:,-2:-1]
            
            loss_T = loss_fn(T_new, T_new_pred)
        else:
            loss_T = torch.tensor(0, dtype=torch.float64)
        
        mass_consv = torch.mean(
                     torch.abs(dx_center(u[...,3:-3]*scaler,device)[...,1:-1,:] + dy_center(v[...,3:-3]*scaler,device)[...,:,1:-1]
                              ))

        if model_T is not None:
            if p_pred:
                loss   = (loss_u + loss_v + loss_T*1e+4)/3.
            else:
                loss   = (loss_u + loss_v + loss_T*1e+4 + loss_p)/4.
        else:
            if p_pred:
                loss   = (loss_u + loss_v)/2.
            else:
                loss   = (loss_u + loss_v + loss_p)/3.
        
        if is_train:
            loss.backward()
            optimizer.step()

        running_loss   = [running_loss[ii]    +  [loss_u,
                                                  loss_v,
                                                  loss_p,
                                                  loss_T,
                                                  mass_consv][ii].item() for ii in range(5)]
        
        counter += 1
        t1 = time.time()
        if i%100 == 0:
            print(epoch, 
                  i, 
                  running_loss[0]/(counter-1),
                  running_loss[1]/(counter-1),
                  running_loss[2]/(counter-1),
                  running_loss[3]/(counter-1),
                  running_loss[4]/(counter-1),
                  t1-t0)
            
    return [running_loss[ii]/(counter-1) for ii in range(5)]


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

dx_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2)
def dx_center(v, device):
	return F.conv2d(v,dx_center_kernel.to(device))#,padding=(0,1))

dy_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3)
def dy_center(v, device):
	return F.conv2d(v,dy_center_kernel.to(device))#,padding=(1,0))

du_dy_kernel = torch.Tensor([1,-1,-1,1]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3)
def du_dy(v, device):
	return F.conv2d(v,du_dy_kernel.to(device))
    
u_u_kernel   = torch.Tensor([[1,1],[0,0],[-1,-1]]).double().unsqueeze(0).unsqueeze(1)
def u_u(v, device):
	return F.conv2d(v,u_u_kernel.to(device))

dv_dx_kernel = torch.Tensor([1,-1,-1,1]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2)
def dv_dx(v, device):
	return F.conv2d(v,dv_dx_kernel.to(device))
    
v_v_kernel   = torch.Tensor([[1,0,-1],[1,0,-1]]).double().unsqueeze(0).unsqueeze(1)
def v_v(v, device):
	return F.conv2d(v,v_v_kernel.to(device))

def rot_mac(a, device):
	return torch.cat((dy_bottom(a,device), -dx_right(a,device)),dim=1)

#laplace_kernel = 0.25*torch.Tensor([[1,2,1],[2,-12,2],[1,2,1]]).double().unsqueeze(0).unsqueeze(1) # isotropic 9 point stencil
laplace_kernel = torch.Tensor([[0,1,0],[1,-4,1],[0,1,0]]).double().unsqueeze(0).unsqueeze(1) # isotropic 9 point stencil
def laplace(v, device):
	return F.conv2d(v,laplace_kernel.to(device))

class TS(nn.Module):
    def __init__(self, stokes, ad, device, ts=8, advection_scheme=2, scale=True, p_pred=True, net="fluidnet"):
        super().__init__()

        self.stokes = stokes
        self.ad     = ad
        self.ts     = ts
        self.device = device
        self.advection_scheme = advection_scheme
        self.scale = scale
        self.p_pred = p_pred
        self.net = net

    @staticmethod
    def __eta_torch(gamma, beta, z, T, Tref=0, zref=0):
        eta = torch.exp( torch.log(gamma)*(Tref-T) + torch.log(beta)*(z-zref) )
        return eta

    @staticmethod
    def __unscale_var(x, raq, fkt, fkp, var):
        if var=="uprev" or var=="vprev":
            scaler = torch.exp((raq/10)*1.80167667 + torch.log(fkt)*0.4330392 + torch.log(fkp)*-0.46052953 )*5 
            x *= scaler 
        return x 

    def forward(self, T_prev, sdf, sdf2, ycc, raq_nd, fkt_nd, fkp_nd, raq, fkt, fkp, xc, yc, u_prev=None, v_prev=None, dt=None):

        x = {}
        dts = {}
        x[0] = T_prev.to(self.device)

        for i in range(1,self.ts+1):
            
            V  = TS.__eta_torch(fkt.to(self.device), 
                                fkp.to(self.device), 
                                1.0-ycc.to(self.device), 
                                x[i-1]
                                , 0, 0)
            
            
            if self.net=="newfluidnet":
                V = torch.clip(V,1e-08,1)
                inp = torch.cat((
                                 xc.to(self.device)/4.,
                                 yc.to(self.device)/4.,
                                 torch.log10(V)/8,
                                 raq_nd.expand(1,1,T_prev.shape[-2],T_prev.shape[-1]).to(self.device),
                                 fkt_nd.expand(1,1,T_prev.shape[-2],T_prev.shape[-1]).to(self.device),
                                 fkp_nd.expand(1,1,T_prev.shape[-2],T_prev.shape[-1]).to(self.device),
                                 x[i-1]), axis=1
                               )
    
                u, v, p = self.stokes(inp)
    
                u = TS.__unscale_var(u, raq, fkt, fkp, "uprev") 
                v = TS.__unscale_var(v, raq, fkt, fkp, "vprev")
    
                u = u.view(-1,1,128,506)
                v = v.view(-1,1,128,506)
                if self.p_pred:
                    p = p.view(-1,1,128,506)
                    
                #u,v,p = pad_uvp(u,v,p)
                
            else:
                V = torch.log10(torch.clip(V, 1e-8, 1.0))/8.0
                inp = torch.cat((
                           xc.to(self.device)/4.,
                           yc.to(self.device)/4.,
                           dt.to(self.device),
                           raq_nd.expand(1,1,u_prev.shape[-2],u_prev.shape[-1]).to(self.device),
                           fkt_nd.expand(1,1,u_prev.shape[-2],u_prev.shape[-1]).to(self.device),
                           fkp_nd.expand(1,1,u_prev.shape[-2],u_prev.shape[-1]).to(self.device),
                           V.to(self.device),
                           x[i-1].to(self.device),
                           u_prev.to(self.device),
                           v_prev.to(self.device)
                          ), axis=1)

                u,v,_,T = self.stokes(inp)
                u = u.view(1,1,128,506)
                v = v.view(1,1,128,506)
                x[i]= T.view(1,1,128,506)
                x[i][:,:,0,:]   = 1
                x[i][:,:,-1,:]  = 0
                x[i][:,:,:,0:1] = x[i][:,:,:,1:2]
                x[i][:,:,:,-1:] = x[i][:,:,:,-2:-1]
                p = None
            
            if self.ad is not None:
                inp = torch.cat((
                                 u.to(self.device),
                                 v.to(self.device),
                                 x[i-1].to(self.device),
                                 torch.zeros_like(u.to(self.device))+raq,
                                 xc.to(self.device),
                                 yc.to(self.device)
                                ), axis=1)
                
                t0 = time.time()
                x[i], dt = self.ad(inp)
                x[i][:,:,0,:]   = 1
                x[i][:,:,-1,:]  = 0
                x[i][:,:,:,0:1] = x[i][:,:,:,1:2]
                x[i][:,:,:,-1:] = x[i][:,:,:,-2:-1]
    
                dts[i] = dt
            
        return x, dts, u, v, p, V


class ADNet(nn.Module):
    def __init__(self, device, r_p="zeros", CN_max=0.1):
        super().__init__()

        self.device     = device
        self.CN_max = CN_max
    
    def forward(self, inputs, dt=None, T_prev=None):

        u      = inputs[:,0:1,1:-1,1:-1]
        v      = inputs[:,1:2,1:-1,1:-1]
        if T_prev is None:
            T_prev = inputs[:,2:3,...]
        RaQ_Ra = inputs[:,3:4,1:-1,1:-1]
        xc = inputs[:,4:5,...]
        yc = inputs[:,5:6,...]

        #T_prev  = F.pad(T_prev, (1,1,1,1), mode="replicate")
        #xc      = F.pad(xc, (1,1,1,1), mode="replicate")
        #yc      = F.pad(yc, (1,1,1,1), mode="replicate")
        
        xc[:,:,:,0]  = 0.0
        xc[:,:,:,-1] = 4.0
        yc[:,:,0,:]  = 0.0
        yc[:,:,-1,:] = 1.0

        dx_l = dx_left(xc,self.device)[...,1:-1,:]
        dx_r = dx_right(xc,self.device)[...,1:-1,:]
        dy_t = dy_top(yc,self.device)[...,1:-1]
        dy_b = dy_bot(yc,self.device)[...,1:-1]
        
        dT_l = dx_left(T_prev,self.device)[...,1:-1,:]
        dT_r = dx_right(T_prev,self.device)[...,1:-1,:]
        dT_t = dy_top(T_prev,self.device)[...,1:-1]
        dT_b = dy_bot(T_prev,self.device)[...,1:-1]

        dT_dx = (dT_l/dx_l)* (u>0) + (dT_r/dx_r) * (u<0)
        dT_dy = (dT_t/dy_t)* (v>0) + (dT_b/dy_b) * (v<0)
        
        T_laplace = (dT_r/dx_r - dT_l/dx_l)/(0.5*dx_r + 0.5*dx_l) + (dT_b/dy_b - dT_t/dy_t)/(0.5*dy_b + 0.5*dy_t)
        
        if dt is None:
            dx_min = torch.amin(dx_l)
            uv_mag = torch.max(torch.amax(torch.abs(u)), torch.amax(torch.abs(v)))
            dt_advect = 0.5*self.CN_max * dx_min/uv_mag   
            dt_diffuse = 0.5*((dx_min*dx_min)**2)/(dx_min**2 + dx_min**2)  
            dt = torch.min(dt_advect,dt_diffuse)
            
        T_prev = T_prev[...,1:-1,1:-1] + dt*(- u*dT_dx - v*dT_dy + T_laplace  + RaQ_Ra) 

        T_prev     = F.pad(T_prev, (1,1,1,1), mode="replicate")
        T_prev[:,:,0,:]  = 1.
        T_prev[:,:,-1,:] = 0.
        return T_prev, dt
    
    
    def forward_old(self, inputs, dt=None, T_prev=None):

        dx     = torch.tensor(1./126., dtype=torch.float64)
        dy     = torch.tensor(1./126., dtype=torch.float64)
        u      = inputs[:,0:1,...]
        v      = inputs[:,1:2,...]
        if T_prev is None:
            T_prev = inputs[:,2:3,...]
        RaQ_Ra = inputs[:,3:4,...]

        # Think about padding 
        #T_prev  = F.pad(T_prev, (1,1,0,0), mode="replicate")
        #T_prev  = pad_grad(T_prev[:,:,1:-1,:], (0,0,2,2))
        T_prev  = F.pad(T_prev, (1,1,1,1), mode="replicate")
        
        dT_dx_l = dx_left(T_prev,self.device)[...,1:-1,:]
        dT_dx_r = dx_right(T_prev,self.device)[...,1:-1,:]
        dT_dy_t = dy_top(T_prev,self.device)[...,:,1:-1]
        dT_dy_b = dy_bot(T_prev,self.device)[...,:,1:-1]
        T_laplace = laplace(T_prev,self.device)/dx**2

        dT_dx = dT_dx_l/dx* (u>0) + dT_dx_r/dx * (u<0)
        dT_dy = dT_dy_t/dx* (v>0) + dT_dy_b/dx * (v<0)

        if dt is None:
            uv_mag = torch.max(torch.amax(torch.abs(u)), torch.amax(torch.abs(v)))
            dt_advect = 0.5*self.CN_max * dx/uv_mag   #
            dt_diffuse = 0.5*((dx*dx)**2)/(dx**2 + dx**2)   #dx**2/1e-6   0.5*
            dt = torch.min(dt_advect,dt_diffuse)
            
        T_new = T_prev[...,1:-1,1:-1] + dt*(-u*dT_dx - v*dT_dy + T_laplace  + RaQ_Ra) 

        return T_new, dt



class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = 4 #modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = 4 #modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cdouble))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cdouble))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cdouble, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class SpectralFluidLayer(nn.Module):
    def __init__(self, c_i: int, c_o: int,  act_fn: str = "selu", r_p="zeros", use_symm=False, 
                 dilation=1, f=3):
        super().__init__()

        self.layers   = nn.ModuleList()
        
        if r_p == "zeros":
            self.r_p    = "constant"
        else:
            self.r_p    = r_p

        if act_fn == "selu":
            self.act = nn.SELU()
        elif act_fn == "sine":
            self.act  = Sine(30.) 
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

        h_s  = int(c_o/4) #int(c_o/8)
        v_s  = 0 #int(c_o/8)
        hv_s = 0# int(c_o/4)

        self.layers.append(SpectralConv2d(c_i, c_o, c_o, c_o))
        self.layers.append(torch.nn.GroupNorm(int(c_o/4), c_o))

    def forward(self, inputs):

        x = self.layers[0](inputs)
        x = self.layers[1](x)
        x = self.act(x)
        return x
        
class FluidLayer(nn.Module):
    def __init__(self, c_i: int, c_o: int,  act_fn: str = "selu", r_p="zeros", use_symm=False, 
                 dilation=1, f=3, drop_rate=0.):
        super().__init__()

        self.layers   = nn.ModuleList()
        
        if r_p == "zeros":
            self.r_p    = "constant"
        else:
            self.r_p    = r_p

        if act_fn == "selu":
            self.act = nn.SELU()
        elif act_fn == "sine":
            self.act  = Sine(30.) 
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
        
        h_s  = int(c_o/4) if c_o>4 else int(c_o/2) #int(c_o/8)
        v_s  = 0 #int(c_o/8)
        hv_s = 0# int(c_o/4)

        if r_p == "learned":
            self.layers.append(BoundaryLearnedConvolution2D(c_i, c_o, k=f, use_symm=use_symm))
        else:
            if use_symm:
                self.layers.append(SymmetricConv2d(c_i, c_o, kernel_size=f, padding='same', 
                                                 dilation=dilation, padding_mode=r_p,
                symmetry={'h':h_s, 'v':v_s, 'hv':hv_s}))
            else:
                self.layers.append(nn.Conv2d(c_i, c_o, kernel_size=f, padding='same', 
                                           dilation=dilation, padding_mode=r_p))

        self.layers.append(torch.nn.GroupNorm(int(c_o/min(4,c_o)), c_o))

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
    def __init__(self, c_i, c_o, k, stride=1, use_symm=False):
        super().__init__()
        self.c_i = c_i
        self.c_o = c_o
        self.k = k

        h_s  = int(c_o/4) if c_o>4 else int(c_o/2) #int(c_o/8)
        v_s  = 0 #int(c_o/8)
        hv_s = 0# int(c_o/4)
        
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
            self.conv = SymmetricConv2d(c_i, c_o, k, bias=False, padding='valid', symmetry={'h':h_s, 'v':v_s, 'hv':hv_s})

        if not use_symm:
            self.conv_top_left = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_top_left = SymmetricConv2d(c_i, c_o, k, bias=False, padding='valid', symmetry={'h':h_s, 'v':v_s, 'hv':hv_s})

        if not use_symm:
            self.conv_top_right = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_top_right = SymmetricConv2d(c_i, c_o, k, bias=False, padding='valid', symmetry={'h':h_s, 'v':v_s, 'hv':hv_s})

        if not use_symm:
            self.conv_bottom_left = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_bottom_left = SymmetricConv2d(c_i, c_o, k, bias=False, padding='valid', symmetry={'h':h_s, 'v':v_s, 'hv':hv_s})
            
        if not use_symm:
            self.conv_bottom_right = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_bottom_right = SymmetricConv2d(c_i, c_o, k, bias=False, padding='valid', symmetry={'h':h_s, 'v':v_s, 'hv':hv_s})


        if not use_symm:
            self.conv_top = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_top = SymmetricConv2d(c_i, c_o, k, bias=False, padding='valid', symmetry={'h':h_s, 'v':v_s, 'hv':hv_s})

        if not use_symm:
            self.conv_bottom = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_bottom = SymmetricConv2d(c_i, c_o, k, bias=False, padding='valid', symmetry={'h':h_s, 'v':v_s, 'hv':hv_s})

        if not use_symm:
            self.conv_left = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_left = SymmetricConv2d(c_i, c_o, k, bias=False, padding='valid', symmetry={'h':h_s, 'v':v_s, 'hv':hv_s})

        if not use_symm:
            self.conv_right = nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=k,
                padding="valid",
                bias=False,
            )
        else:
            self.conv_right = SymmetricConv2d(c_i, c_o, k, bias=False, padding='valid', symmetry={'h':h_s, 'v':v_s, 'hv':hv_s})

        # Define learnable bias (initialized to zeros)
        self.learnable_bias = nn.Parameter(torch.zeros(1, c_o, 1, 1))

    def forward(self, x, bc_x=1, bc_y=1):
        # Corners
        #pad = self.k + 1 if self.k == 5 else self.k
        pad_x = self.k + 1 + (bc_x-1) if self.k == 5 else self.k + (bc_x-1)
        pad_y = self.k + 1 + (bc_y-1) if self.k == 5 else self.k + (bc_y-1)
        
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
    def __init__(self, levels: int, c_i: int, c_h: int, c_o: int, device, 
                 act_fn: str = "selu", r_p="zeros", loss_type="mae", use_symm=False, 
                 dilation=1, a_bound=4., use_cosine=False, repeats=3, use_skip=False, f=3, 
                p_pred=True, spectral_conv=False, blurr=False, drop_rate=0.0, factor=2):
        super().__init__()

        self.conv   = nn.ModuleList()
        self.gn     = nn.ModuleList()
        self.pool   = nn.ModuleList()
        self.unpool = nn.ModuleList()
        self.levels = levels
        #self.device = device
        self.loss_type  = loss_type
        self.a_bound = a_bound
        self.use_cosine = use_cosine
        self.repeats = repeats
        self.use_skip = use_skip
        self.p_pred = p_pred
        self.c_h = c_h
        self.c_i = c_i
        self.c_o = c_o
        if blurr:
            self.blurrer = torch.Tensor([[1/9]*3]*3).double().unsqueeze(0).unsqueeze(1).to(device)
        else:
            self.blurrer = None

        if r_p == "zeros":
            self.r_p    = "constant"
        else:
            self.r_p    = r_p

        if act_fn == "selu":
            self.act = nn.SELU()
        elif act_fn == "sine":
            self.act  = Sine(30.) 
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

        self.dx_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2).to(device)
        self.dy_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3).to(device)

        if spectral_conv:
            self.conv.append(SpectralFluidLayer(c_i, c_h, act_fn, r_p, use_symm, dilation, f=f))
        else:
            self.conv.append(FluidLayer(c_i, c_h, act_fn, r_p, use_symm, dilation, f=f, drop_rate=drop_rate))

        xs = [128]
        ys = [506]

        self.pool = nn.AvgPool2d((factor,factor), stride=factor)
        for l in range(1, levels):
            self.unpool.append(nn.Upsample(size=(xs[0],ys[0]), mode='bicubic')) #, align_corners=True

        self.convs = nn.ModuleList()
        
        for l in range(levels):
            self.convs.append(nn.ModuleList())
            for r in range(self.repeats):
                if spectral_conv:
                    self.convs[l].append(SpectralFluidLayer(c_h, c_h, act_fn, r_p, use_symm, dilation, f=f))
                else:
                    self.convs[l].append(FluidLayer(c_h, c_h, act_fn, r_p, use_symm, dilation, f=f, drop_rate=drop_rate))

        if self.loss_type == "curl":
            padding = (1,1)
        else:
            padding = (1,1)

        if self.r_p != "learned":
            self.conv.append(nn.Conv2d(c_h*levels+c_i, c_h, kernel_size=3, padding=padding, 
                                           dilation=dilation, padding_mode=r_p, stride=1))
        else:
            self.conv.append(BoundaryLearnedConvolution2D(c_h*levels+c_i, c_h, k=f, use_symm=use_symm))
        self.gn.append(torch.nn.GroupNorm(int(c_h/4), c_h))

        if self.r_p != "learned":
            self.conv.append(nn.Conv2d(c_h, c_h, kernel_size=3, padding=(1,1), 
                                   dilation=1, padding_mode=r_p, stride=1))
        else:
            self.conv.append(BoundaryLearnedConvolution2D(c_h, c_h, k=f, use_symm=use_symm))

        if self.r_p != "learned":
            self.conv.append(nn.Conv2d(c_h, c_o, kernel_size=3, padding=(1,1), 
                                       dilation=1, padding_mode=r_p, stride=1))
        else:
            self.conv.append(BoundaryLearnedConvolution2D(c_h, c_o, k=f, use_symm=use_symm))

    def forward(self, inputs):

        x_in = self.conv[0](inputs)

        for l in range(self.levels):
            y1 = x_in
            for _ in range(l):
                y1 = self.pool(y1)
            for r in range(self.repeats):
                y1 = self.convs[l][r](y1)
            if l>0:
                y1 = self.unpool[l-1](y1)
                y = torch.cat((y,y1), axis=1)
            else:
                y = y1

        del x_in, y1
        y = torch.cat((y,inputs), axis=1)
        del inputs
        
        y = self.conv[1](y) 
        y = self.gn[0](y)
        y = self.act(y)

        y = self.conv[2](y)
        y = self.act(y)

        y = self.conv[3](y)
        y = y - torch.mean(y, dim=(2,3), keepdim=True) 
        
        if self.loss_type == "mae" or self.loss_type == "mass":
            u = y[:,0:1,...]
            v = y[:,1:2,...]
            if self.p_pred:
                p = y[:,2:3,...]
            else:
                p = None
            del y
            
            return u[:,0,...],v[:,0,...],p 
            
        elif self.loss_type == "curl":
            a = y[:,0:1,...] * self.a_bound

            if self.blurrer is not None:
                a = F.pad(a, (1,1,1,1), mode="replicate")
                a = F.conv2d(a, self.blurrer)

            if self.p_pred:
                p = y[:,1,...]
            else:
                p = None
            del y

            u =  F.conv2d(a, self.dy_center_kernel)[:,:,:,1:-1]   #126 x 506
            v = -F.conv2d(a, self.dx_center_kernel)[:,:,1:-1,:]   #128 x 504

            u = F.pad(u, (1,1,1,1), mode="replicate")
            u[:,:,:,0]  = -u[:,:,:,1]
            u[:,:,:,-1] = -u[:,:,:,-2]
            u[:,:,0,0]   = 0
            u[:,:,0,-1]  = 0
            u[:,:,-1,0]  = 0
            u[:,:,-1,-1] = 0

            v = F.pad(v, (1,1,1,1), mode="replicate")
            v[:,:,0,:]  = -v[:,:,1,:]
            v[:,:,-1,:] = -v[:,:,-2,:]
            v[:,:,0,0]   = 0
            v[:,:,0,-1]  = 0
            v[:,:,-1,0]  = 0
            v[:,:,-1,-1] = 0
            
            return u[:,0,...],v[:,0,...],p #,a,u_u_bc,v_v_bc,du_dy_bc,dv_dx_bc 


class HalfNewFluidNet(nn.Module):
    def __init__(self, levels: int, c_i: int, c_h: int, c_o: int, device, 
                 act_fn: str = "selu", r_p="zeros", loss_type="mae", use_symm=False, 
                 dilation=1, a_bound=4., use_cosine=False, repeats=3, use_skip=False, f=3, 
                p_pred=True, spectral_conv=False, blurr=False, drop_rate=0.0, factor=2):
        super().__init__()

        self.conv   = nn.ModuleList()
        self.gn     = nn.ModuleList()
        self.pool   = nn.ModuleList()
        self.unpool = nn.ModuleList()
        self.levels = levels
        #self.device = device
        self.loss_type  = loss_type
        self.a_bound = a_bound
        self.use_cosine = use_cosine
        self.repeats = repeats
        self.use_skip = use_skip
        self.p_pred = p_pred
        self.c_h = c_h
        self.c_i = c_i
        self.c_o = c_o
        if blurr:
            self.blurrer = torch.Tensor([[1/9]*3]*3).double().unsqueeze(0).unsqueeze(1).to(device)
        else:
            self.blurrer = None

        if r_p == "zeros":
            self.r_p    = "constant"
        else:
            self.r_p    = r_p

        if act_fn == "selu":
            self.act = nn.SELU()
        elif act_fn == "sine":
            self.act  = Sine(30.) 
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

        self.dx_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2).to(device)
        self.dy_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3).to(device)

        if spectral_conv:
            self.conv.append(SpectralFluidLayer(c_i, c_h, act_fn, r_p, use_symm, dilation, f=f))
        else:
            self.conv.append(FluidLayer(c_i, c_h, act_fn, r_p, use_symm, dilation, f=f, drop_rate=drop_rate))

        xs = [128]
        ys = [506]

        self.pool = nn.AvgPool2d((factor,factor), stride=factor)
        for l in range(1, levels):
            self.unpool.append(nn.Upsample(size=(xs[0],ys[0]), mode='bicubic')) #, align_corners=True

        self.convs = nn.ModuleList()
        
        for l in range(levels):
            self.convs.append(nn.ModuleList())
            for r in range(self.repeats):
                if spectral_conv:
                    self.convs[l].append(SpectralFluidLayer(c_h, c_h, act_fn, r_p, use_symm, dilation, f=f))
                else:
                    self.convs[l].append(FluidLayer(c_h, c_h, act_fn, r_p, use_symm, dilation, f=f, drop_rate=drop_rate))

        if self.loss_type == "curl":
            padding = (1,1)
        else:
            padding = (1,1)

        if self.r_p != "learned":
            self.conv.append(nn.Conv2d(c_h*levels+c_i, c_h, kernel_size=3, padding=padding, 
                                           dilation=dilation, padding_mode=r_p, stride=1))
        else:
            self.conv.append(BoundaryLearnedConvolution2D(c_h*levels+c_i, c_h, k=f, use_symm=use_symm))
        self.gn.append(torch.nn.GroupNorm(int(c_h/4), c_h))

        if self.r_p != "learned":
            self.conv.append(nn.Conv2d(c_h, c_h, kernel_size=3, padding=(1,1), 
                                   dilation=1, padding_mode=r_p, stride=1))
        else:
            self.conv.append(BoundaryLearnedConvolution2D(c_h, c_h, k=f, use_symm=use_symm))

        if self.r_p != "learned":
            self.conv.append(nn.Conv2d(c_h, c_o, kernel_size=3, padding=(1,1), 
                                       dilation=1, padding_mode=r_p, stride=1))
        else:
            self.conv.append(BoundaryLearnedConvolution2D(c_h, c_o, k=f, use_symm=use_symm))

    def forward(self, inputs):

        x_in = self.conv[0](inputs)

        for l in range(self.levels):
            y1 = x_in
            for _ in range(l):
                y1 = self.pool(y1)
            for r in range(self.repeats):
                y1 = self.convs[l][r](y1)
            if l>0:
                y1 = self.unpool[l-1](y1)
                y = torch.cat((y,y1), axis=1)
            else:
                y = y1

        del x_in, y1
        y = torch.cat((y,inputs), axis=1)
        del inputs
        
        y = self.conv[1](y) 
        y = self.gn[0](y)
        y = self.act(y)

        y = self.conv[2](y)
        y = self.act(y)

        y = self.conv[3](y)
        y = y - torch.mean(y, dim=(2,3), keepdim=True) 
        
        return y
            
class MultiScaleNewFluidNet(nn.Module):
    def __init__(self, nets, loss_type, device, scales=[1e-5,1e-4,1e-3,1e-2,1e-1,1e+0,1e+1], p_pred=False):
        super().__init__()

        self.nets   = nets
        self.scales = scales
        self.dx_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2).to(device)
        self.dy_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3).to(device)
        self.loss_type = loss_type
        self.p_pred = p_pred
        
    def forward(self, x):

        #y = 0
        #for i, scale in enumerate(self.scales):
        #    y = y + scale * self.nets[i](x)

        # Assuming self.scales is a tensor and self.nets is a list of modules
        scales = torch.tensor(self.scales, device=x.device).double()  # Ensure it's a tensor on the right device
        # Stack all outputs into a tensor and perform batch multiplication with scales
        outputs = torch.stack([net(x) for net in self.nets])  # Shape: (num_nets, batch_size, features...)
        y = torch.einsum("i,ib...->b...", scales, outputs)  # Efficient weighted sum

        if self.loss_type == "mae" or self.loss_type == "mass":
            u = y[:,0:1,...]
            v = y[:,1:2,...]
            if self.p_pred:
                p = y[:,2:3,...]
            else:
                p = None
            del y
            
            return u[:,0,...],v[:,0,...],p 
            
        elif self.loss_type == "curl":
            #a = y[:,0:1,...] * self.a_bound
            a = y[:,0:1,...] 

            #if self.blurrer is not None:
            #    a = F.pad(a, (1,1,1,1), mode="replicate")
            #    a = F.conv2d(a, self.blurrer)

            if self.p_pred:
                p = y[:,1,...]
            else:
                p = None
            del y

            u =  F.conv2d(a, self.dy_center_kernel)[:,:,:,1:-1]   #126 x 506
            v = -F.conv2d(a, self.dx_center_kernel)[:,:,1:-1,:]   #128 x 504

            u = F.pad(u, (1,1,1,1), mode="replicate")
            u[:,:,:,0]  = -u[:,:,:,1]
            u[:,:,:,-1] = -u[:,:,:,-2]
            u[:,:,0,0]   = 0
            u[:,:,0,-1]  = 0
            u[:,:,-1,0]  = 0
            u[:,:,-1,-1] = 0

            v = F.pad(v, (1,1,1,1), mode="replicate")
            v[:,:,0,:]  = -v[:,:,1,:]
            v[:,:,-1,:] = -v[:,:,-2,:]
            v[:,:,0,0]   = 0
            v[:,:,0,-1]  = 0
            v[:,:,-1,0]  = 0
            v[:,:,-1,-1] = 0
            
            return u[:,0,...],v[:,0,...],p #,a,u_u_bc,v_v_bc,du_dy_bc,dv_dx_bc 
            
        return u, v


class Unet(nn.Module):
    def __init__(self, levels: int, c_i: int, c_h: int, c_o: int, device = torch.device("cpu"), 
                 act_fn: str = "gelu", r_p="replicate", loss_type="curl", use_symm=False, 
                 dilation=1, a_bound=10., use_cosine=False, repeats=2, use_skip=False, f=5, 
                 p_pred=False, spectral_conv=False, blurr=False, drop_rate=0.0):
        super().__init__()

        self.conv   = nn.ModuleList()
        self.gn     = nn.ModuleList()
        self.pool   = nn.ModuleList()
        self.unpool = nn.ModuleList()
        self.levels = levels
        #self.device = device
        self.loss_type  = loss_type
        self.a_bound = a_bound
        self.use_cosine = use_cosine
        self.repeats = repeats
        self.use_skip = use_skip
        self.p_pred = p_pred
        if blurr:
            self.blurrer = v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
        else:
            self.blurrer = None

        if r_p == "zeros":
            self.r_p    = "constant"
        else:
            self.r_p    = r_p

        if act_fn == "selu":
            self.act = nn.SELU()
        elif act_fn == "sine":
            self.act  = Sine(30.) 
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

        self.dx_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2).to(device)
        self.dy_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3).to(device)

        for r in range(self.repeats):
            if r == 0:
                c_in = c_i
            else:
                c_in =c_h
            if spectral_conv:
                self.conv.append(SpectralFluidLayer(c_in, c_h, act_fn, r_p, use_symm, dilation, f=f))
            else:
                self.conv.append(FluidLayer(c_in, c_h, act_fn, r_p, use_symm, dilation, f=f, drop_rate=drop_rate))

        self.pool   = nn.AvgPool2d((2,2), stride=2)
        #self.unpool = nn.Upsample(scale_factor=2, mode='bicubic')

        self.convs = nn.ModuleList()

        for l in range(1,levels):
            self.convs.append(nn.ModuleList())
            for r in range(self.repeats):
                if r==0 and l>1:
                    c_in = int(c_h/2)
                else:
                    c_in = c_h

                if spectral_conv:
                    self.convs[-1].append(SpectralFluidLayer(c_in, c_h, act_fn, r_p, use_symm, dilation, f=f))
                else:
                    self.convs[-1].append(FluidLayer(c_in, c_h, act_fn, r_p, use_symm, dilation, f=f, drop_rate=drop_rate))
            c_h *= 2
        c_h = int(c_h/2)

        self.upconvs       = nn.ModuleList()
        for l in range(levels-2,0,-1):
            self.upconvs.append(nn.ModuleList())
            for r in range(self.repeats):
                if r==0:
                    c_in  = c_h + int(c_h/2)
                    c_out = int(c_h/2)
                else:
                    c_in  = int(c_h/2)
                    c_out = int(c_h/2)

                if spectral_conv:
                    self.upconvs[-1].append(SpectralFluidLayer(c_in, c_out, act_fn, r_p, use_symm, dilation, f=f))
                else:
                    self.upconvs[-1].append(FluidLayer(c_in, c_out, act_fn, r_p, use_symm, dilation, f=f, drop_rate=drop_rate))

            c_h = int(c_h/2)
        
        if self.r_p != "learned":
            self.conv.append(nn.Conv2d(int(c_h*2), c_h, kernel_size=f, padding="same", 
                                           dilation=dilation, padding_mode=r_p, stride=1))
        else:
            self.conv.append(BoundaryLearnedConvolution2D(int(c_h*2), c_h, k=f, use_symm=use_symm))
        self.gn.append(torch.nn.GroupNorm(int(c_h/4), c_h))

        if self.r_p != "learned":
            self.conv.append(nn.Conv2d(c_h, c_h, kernel_size=f, padding="same", 
                                   dilation=1, padding_mode=r_p, stride=1))
        else:
            self.conv.append(BoundaryLearnedConvolution2D(c_h, c_h, k=f, use_symm=use_symm))

        if self.r_p != "learned":
            self.conv.append(nn.Conv2d(c_h, c_o, kernel_size=f, padding="same", 
                                       dilation=1, padding_mode=r_p, stride=1))
        else:
            self.conv.append(BoundaryLearnedConvolution2D(c_h, c_o, k=f, use_symm=use_symm))
    
    def forward(self, inputs):

        x = {}
        s = {}

        if self.r_p != "learned":
            inputs = F.pad(inputs, (3,3,0,0), mode=self.r_p)
        x[0] = inputs
        for r in range(self.repeats):
            if self.r_p == "learned" and r==0:
                x[0] = self.conv[r](x[0], bc_x=4, bc_y=1)
            else:
                x[0] = self.conv[r](x[0])

        sizes = {}
        sizes[0] = (x[0].shape[-2], x[0].shape[-1])
        for l in range(1, self.levels):
            x[l] = self.pool(x[l-1])
            sizes[l] = (x[l].shape[-2], x[l].shape[-1])
            for r in range(self.repeats):
                x[l] = self.convs[l-1][r](x[l])
        xu = x[l]
        
        for l_i, l in enumerate(np.arange(self.levels-2,0,-1)):
            xu = nn.Upsample(size=sizes[l], mode='bicubic')(xu)
            xu = torch.cat((x[l], xu), dim=1)
            for r in range(self.repeats):
                xu = self.upconvs[l_i][r](xu)

        xu = nn.Upsample(size=sizes[0], mode='bicubic')(xu)
        y = torch.cat((xu, x[0]), axis=1)
        y = self.conv[-3](y) #[:,:,:,3:-3])
        y = self.gn[0](y)
        y = self.act(y)

        y = self.conv[-2](y)
        y = self.act(y)

        y = self.conv[-1](y)
        y = (y - torch.mean(y, dim=(2,3), keepdim=True))[...,3:-3] 

        if self.loss_type == "mae" or self.loss_type == "mass":
            u = y[:,0:1,...]
            v = y[:,1:2,...]
            T = y[:,2:3,...]
            if self.p_pred:
                p = y[:,3:4,...]
            else:
                p = None
            del y
            
            return u,v,p,T 
            
        elif self.loss_type == "curl":
            a = y[:,0:1,...]*self.a_bound 
            T = torch.clip(y[:,1,...], 0.0, 1.5)
            
            if self.blurrer is not None:
                a = self.blurrer(a)

            if self.p_pred:
                p = y[:,2,...]
            else:
                p = None
            del y

            u =  F.conv2d(a, self.dy_center_kernel)[:,:,:,1:-1]   #126 x 506
            v = -F.conv2d(a, self.dx_center_kernel)[:,:,1:-1,:]   #128 x 504

            u = F.pad(u, (1,1,1,1), mode="replicate")
            u[:,:,:,0]  = -u[:,:,:,1]
            u[:,:,:,-1] = -u[:,:,:,-2]
            u[:,:,0,0]   = 0
            u[:,:,0,-1]  = 0
            u[:,:,-1,0]  = 0
            u[:,:,-1,-1] = 0

            v = F.pad(v, (1,1,1,1), mode="replicate")
            v[:,:,0,:]  = -v[:,:,1,:]
            v[:,:,-1,:] = -v[:,:,-2,:]
            v[:,:,0,0]   = 0
            v[:,:,0,-1]  = 0
            v[:,:,-1,0]  = 0
            v[:,:,-1,-1] = 0

            return u[:,0,...],v[:,0,...],p,T #,a,u_u_bc,v_v_bc,du_dy_bc,dv_dx_bc  