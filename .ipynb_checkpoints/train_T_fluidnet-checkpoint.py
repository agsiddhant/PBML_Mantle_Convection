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
    
    args = parser.parse_args()
    
    gpu_number = args.gpu_number
    act_fn = args.act_fn
    levels = args.levels
    c_h = args.c_h
    r_p = args.r_p
    loss_type = args.loss_type
    dilation = args.dilation
    batch_size = args.batch_size
    
else:
    gpu_number = 3
    act_fn = "selu"
    levels = 4
    c_h = 8
    r_p = "replicate"
    loss_type = "mae"
    dilation = 1
    batch_size = 8


# In[8]:


device = torch.device("cuda:" + str(gpu_number)) if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")
print(device)

torch.cuda.empty_cache()
nn_dir = "/plp_user/agar_sh/PBML/pytorch/TPH/CONVNN/trained_networks/"
f_nn = "fluidnet_T_levels" + str(levels) + "_" + act_fn + \
            "_" + str(c_h) + "_" + r_p + "_" + loss_type + "_dil_" + str(dilation)

if not os.path.isdir(nn_dir + f_nn):
    os.mkdir(nn_dir + f_nn)

epoch       = 0
start_lr    = 1e-3
milestones  = [5, 10, 15, 20, 25]
epochs      = 50
best_vloss  = 1e+16
scale       = True #if loss_type == "mae" else False
use_symm    = True if "symm" in f_nn else False

c_i = 5
c_o = 1
model_T = FluidNetT(levels, c_i, c_h, c_o, device, act_fn, r_p, loss_type, 
                     use_symm=use_symm, dilation=dilation).double().to(device)
print(count_parameters(model_T))
print(model_T)
#c_i = 6
#c_o = 1
#model_T = FluidNet(levels, c_i, int(c_h/2), c_o, act_fn).double().to(device)
#print(count_parameters(model_T))

nn_dir = nn_dir + f_nn + "/"
with open(nn_dir + "fluidnet_T.txt", 'w') as writer:
    writer.write('Epoch, train loss, val loss, learning rate \n')


# In[5]:


dataset = {}
loader = {}
batches = {}

for an in ["train", "cv"]:
    
    dataset[an] = TDataset(data_dir, an, scale)
    batches[an] = int(len(dataset[an])/batch_size)
    loader[an]  = DataLoader(dataset[an], batch_size=batch_size, shuffle=True)
    print(an, batches[an])


# In[9]:


optimizer = torch.optim.Adam([
                    {"params": model_T.parameters(),
                     "lr": start_lr,
                     "weight_decay": 1e-6
                    }
])

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)


# In[10]:


while epoch<epochs:
    model_T.train(True)
    avg_loss = one_epoch_T(model_T, epoch, loader["train"], optimizer, device, loss_type, is_train=True)
    
    model_T.eval()
    avg_vloss = one_epoch_T(model_T, epoch, loader["cv"], optimizer, device, loss_type, is_train=False)

    print("-------------------------------------------")
    print(epoch, avg_loss, avg_vloss, get_lr(optimizer))
    print("-------------------------------------------")
    
    #if avg_vloss < best_vloss:
    #    best_vloss = avg_vloss

    torch.save(model_T.state_dict(), nn_dir + str(epoch) + "_fluidnet_T.pt")
    
    with open(nn_dir + "fluidnet_T.txt", "a") as writer:
        writer.write(str(epoch) + "," + str(avg_loss) 
                     + "," + str(avg_vloss) + "," + str(get_lr(optimizer)) + "\n")
    scheduler.step()
    epoch += 1
