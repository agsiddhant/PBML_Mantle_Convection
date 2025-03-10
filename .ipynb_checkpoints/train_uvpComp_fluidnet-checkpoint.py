import glob, os, sys

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
from tabulate import tabulate
from pytorch_networks_convae import *
import argparse
from datasetio import *

import time


# In[2]:


data_dir = "/plp_scr3/agar_sh/data/TPH/"


# In[3]:


var_vec = [["uprev"], ["vprev"], ["pprev"], ["Tprev"]]
encoder = {}
decoder = {}

for var in var_vec:
    x_w = 16
    y_w = 16
    overlap = 0
    b = 8
    act_fn = "selu"
    epoch = 8
    rep = 0
    
    device = torch.device("cpu")
    
    nn_dir = "/plp_user/agar_sh/PBML/pytorch/TPH/CONVAE/trained_networks/"
    f_nn = "convae_d4" + "_" + str(var) + "_" + str(x_w)+ "_" + str(y_w) + "_b" + str(b) + "_" + act_fn 
    
    c_i = len(var)
    
    w_i = min(x_w,y_w)
    w_e = 2
    
    if "nofnn" in f_nn:
        c_h = 4
        levels = int(np.emath.logn(w_e, w_i) - 1)
        f = []
    else:
        c_h = 8
        levels = int(np.emath.logn(w_e, w_i) - 1)
        f = [int(y_w*c_h/4)] 
        while f[-1] != b:
            add = int(f[-1]/4)
            if add < b:
                add = b
            f += [add]
    
    act_fn = nn.Tanh if act_fn == "tanh" else nn.SELU
    
    nn_dir = nn_dir + f_nn + "/"
    print(nn_dir)
    
    with open(nn_dir + "encoder_decoder.txt") as fw:
        lines = fw.readlines()
    fw.close()
    loss    = []
    loss_cv = []
    
    for l in lines[1:]:
        ll = l.split(",")
        loss.append(float(ll[1]))
        loss_cv.append(float(ll[2]))
    
    plt.figure()
    plt.plot(loss, 'bx-')
    plt.plot(loss_cv, 'rx-')
    plt.yscale("log")
    plt.show()
    
    encoder[var[0]] = Encoder(levels=levels, c_i=c_i, c_h=c_h, latent_dim=b, f=f, 
                      act_fn=act_fn, rep=rep).double().to(device)
    encoder[var[0]].load_state_dict(torch.load(nn_dir + "encoder_" + str(epoch) + ".pt", map_location=device))
    encoder[var[0]].eval()
    
    decoder[var[0]] = Decoder(levels=levels, c_i=c_i, c_h=c_h, latent_dim=b,  act_fn=act_fn,
                      f=f, rep=rep).double().to(device)
    decoder[var[0]].load_state_dict(torch.load(nn_dir + "decoder_" + str(epoch) + ".pt", map_location=device))
    decoder[var[0]].eval()


# In[4]:


run_cell = True
if run_cell:
    parser = argparse.ArgumentParser(description='Train autodecoders')
    parser.add_argument("-gpu", "--gpu_number", type=int, help="specify gpu number")
    parser.add_argument("-a", "--act_fn", type=str, help ="activation function")
    parser.add_argument("-l", "--levels", type=int, help ="levels")
    parser.add_argument("-f", "--c_h", type=int, help ="filters")
    parser.add_argument("-p", "--r_p", type=str, help="padding type")
    parser.add_argument("-lt", "--loss_type", type=str, help="loss type")
    parser.add_argument("-d", "--dilation", type=int, help="loss type")
    parser.add_argument("-b", "--batch_size", type=int, help="batch size")
    parser.add_argument("-s", "--use_symm", type=int, help="use symmetries")
    args = parser.parse_args()
    
    gpu_number = args.gpu_number
    act_fn = args.act_fn
    levels = args.levels
    c_h = args.c_h
    r_p = args.r_p
    loss_type = args.loss_type
    dilation = args.dilation
    batch_size = args.batch_size
    use_symm = True if args.use_symm == 1 else False
else:
    gpu_number = 0
    act_fn = "selu"
    levels = 3
    c_h = 16
    r_p = "replicate"
    loss_type = "mae"
    dilation = 1
    batch_size = 16


# In[5]:


device = torch.device("cuda:" + str(gpu_number)) if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")
print(device)

torch.cuda.empty_cache()
nn_dir = "/plp_user/agar_sh/PBML/pytorch/TPH/CONVNN/trained_networks/"
f_nn = "fluidnet_uvpComp_levels" + str(levels) + "_" + act_fn + \
            "_" + str(c_h) + "_" + r_p + "_" + loss_type + "_dil_" + str(dilation)

if not os.path.isdir(nn_dir + f_nn):
    os.mkdir(nn_dir + f_nn)

epoch       = 0
start_lr    = 1e-3
milestones  = [40, 80, 100, 120, 140]
epochs      = 160
best_vloss  = 1e+16
scale       = True #if loss_type == "mae" else False

if loss_type == "curl":
    c_i = 10
    c_o = 24
else:
    c_i = 10
    c_o = 24

model_uvp = FluidNetComp(levels, c_i, c_h, c_o, device, act_fn, r_p, loss_type, 
                     use_symm=use_symm, dilation=dilation).double().to(device)
print(count_parameters(model_uvp))
print(model_uvp)
#c_i = 6
#c_o = 1
#model_T = FluidNet(levels, c_i, int(c_h/2), c_o, act_fn).double().to(device)
#print(count_parameters(model_T))

nn_dir = nn_dir + f_nn + "/"
with open(nn_dir + "fluidnet.txt", 'w') as writer:
    writer.write('Epoch, train loss, val loss, learning rate \n')


# In[6]:


dataset = {}
loader = {}
batches = {}

for an in ["train", "cv"]:
    
    dataset[an] = ADDatasetCompressed(data_dir, encoder, an, var, x_w, y_w, overlap, b)
    batches[an] = int(len(dataset[an])/batch_size)
    loader[an]  = DataLoader(dataset[an], batch_size=batch_size, shuffle=True)
    print(an, batches[an])


# In[7]:


optimizer = torch.optim.Adam([
                    {"params": model_uvp.parameters(),
                     "lr": start_lr,
                     #"weight_decay": 1e-6
                    }
])

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)


# In[ ]:


while epoch<epochs:
    model_uvp.train(True)
    avg_loss = one_epoch_ADComp(model_uvp, epoch, loader["train"], optimizer, device, loss_type, is_train=True)
    
    model_uvp.eval()
    avg_vloss = one_epoch_ADComp(model_uvp, epoch, loader["cv"], optimizer, device, loss_type, is_train=False)

    print("-------------------------------------------")
    print(epoch, avg_loss, avg_vloss, get_lr(optimizer))
    print("-------------------------------------------")
    
    #if avg_vloss < best_vloss:
    #    best_vloss = avg_vloss

    torch.save(model_uvp.state_dict(), nn_dir + str(epoch) + "_fluidnet.pt")
    
    with open(nn_dir + "fluidnet.txt", "a") as writer:
        writer.write(str(epoch) + "," + str(avg_loss) 
                     + "," + str(avg_vloss) + "," + str(get_lr(optimizer)) + "\n")
    scheduler.step()
    epoch += 1


# In[ ]:




