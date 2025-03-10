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


data_dir = "/plp_scr1/agar_sh/data/TPH/"


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
    parser.add_argument("-s", "--use_symm", type=int, help="use symmetries")
    parser.add_argument("-ab", "--a_bound", type=int, help="bound of a")
    parser.add_argument("-r", "--repeats", type=int, help="repeats in hidden layers")
    parser.add_argument("-rst", "--restart", type=int, help="restart")
    parser.add_argument("-sk", "--use_skip", type=int, help="use skip connections")
    parser.add_argument("-k", "--kernel", type=int, help="kernel size")
    args = parser.parse_args()
    
    gpu_number = args.gpu_number
    act_fn = args.act_fn
    levels = args.levels
    c_h = args.c_h
    r_p = args.r_p
    loss_type = args.loss_type
    dilation = args.dilation
    batch_size = args.batch_size
    a_bound = args.a_bound
    use_symm = True if args.use_symm == 1 else False
    restart = True if args.restart == 1 else False
    use_skip = True if args.use_skip == 1 else False
    repeats = args.repeats
    kernel = args.kernel
    
else:
    gpu_number = 5
    act_fn = "gelu"
    levels = 6
    c_h = 16
    r_p = "replicate"
    loss_type = "curl"
    dilation = 1
    batch_size = 16
    use_symm = False
    a_bound = 10
    repeats = 3
    restart = True
    kernel = 3


device = torch.device("cuda:" + str(gpu_number)) if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")

torch.cuda.empty_cache()
nn_dir = "/plp_user/agar_sh/PBML/pytorch/TPH/CONVNN/trained_networks/"
f_nn   = "fluidnet_uvpT_levels_aCnS_test_" + str(levels) + "_" + act_fn + \
            "_" + str(c_h) + "_" + r_p + "_" + loss_type + "_dil_" + str(dilation) +  \
            "_" + str(use_symm) + "_ab" + str(a_bound) + "_b" + str(batch_size) + \
            "_r" + str(repeats) + "_skip" + str(use_skip) + "_k" + str(kernel)

epochs      = 25000
scale       = True 
milestones  = [1000, 5000, 10000, 15000, 20000]

if loss_type == "curl":
    c_i = 6
    c_o = 2
else:
    c_i = 6
    c_o = 3

model_uvp = FluidNet(levels, c_i, c_h, c_o, device, act_fn, r_p, loss_type, 
                     use_symm=use_symm, dilation=dilation, a_bound=a_bound,
                    repeats=repeats, use_skip=use_skip, f=kernel).double().to(device)
torch.compile(model_uvp)
print(count_parameters(model_uvp))
print(model_uvp)

nn_dir = nn_dir + f_nn + "/"

if restart:                
    with open(nn_dir + "fluidnet_uvpT.txt") as fw:
        lines = fw.readlines()
    fw.close()
    loss_u       = []
    loss_v       = []
    loss_p       = []
    loss_mass    = []
    loss_cv_u    = []
    loss_cv_v    = []
    loss_cv_p    = []
    loss_cv_mass = []
    
    for l in lines[1:]:
        ll    = l[l.index("[")+1:l.index("],[")].split(",")
        l_r   = l[l.index("],[")+3:]
        ll_cv = l_r[:l_r.index("],")].split(",")                   

        loss_u.append([float(ll[0])])
        loss_v.append([float(ll[1])])
        loss_p.append([float(ll[2])])    
        loss_mass.append([float(ll[3])+1e-16])
        
        loss_cv_u.append([float(ll_cv[0])])
        loss_cv_v.append([float(ll_cv[1])])
        loss_cv_p.append([float(ll_cv[2])])
        loss_cv_mass.append([float(ll_cv[3])+1e-16])

    epoch       = int(lines[-1].split(",")[0])
    start_lr    = float(lines[-1].split(",")[-1])
    best_vloss  = 1e+16
    start_milestone_ind = np.where(np.asarray(milestones)>epoch)[0][0]
    milestones = [milestones[start_milestone_ind]-epoch] + milestones[start_milestone_ind+1:]
    print("Restarting from epoch, lr, milestones")
    print(epoch, start_lr, milestones)
    model_uvp.load_state_dict(torch.load(nn_dir + str(epoch) + "_fluidnet_uvp.pt", map_location=device))
else:
    if not os.path.isdir(nn_dir):
        os.mkdir(nn_dir)
        
    epoch       = 0
    start_lr    = 1e-3
    best_vloss  = 1e+16
    
    with open(nn_dir + "fluidnet_uvpT.txt", 'w') as writer:
        writer.write('Epoch, train loss, val loss, learning rate \n')


dataset = {}
loader = {}
batches = {}

dataset_init = {}
loader_init = {}
batches_init = {}

init_samples = 1

sims = torch.load(data_dir + "/sims.pt")

sim = sims[0]
ignr, ignr, raq, fkt, fkp, gr, ar, ignr = sim
py_dir = data_dir + "/" + sim[1] + "/sim_" + str(sim[0])
times = torch.load(py_dir + "/times.pt")
print(tabulate([["num", "dataset", "raq", "fkt", "fkp", "gr", "ar"],
                    sim[:-1]
                   ]))
    
raq_nd = torch.tensor((raq-0.12624371)/(9.70723344-0.12624371), dtype=torch.float64)
fkt_nd = torch.tensor((np.log10(fkt)-6.00352841978384)/(
         9.888820429862925-6.00352841978384), dtype=torch.float64)
fkp_nd = torch.tensor((np.log10(fkp)-0.005251646002323797)/(
        1.9927988938926755-0.005251646002323797), dtype=torch.float64)

fkt = torch.tensor(fkt, dtype=torch.float64)
fkp = torch.tensor(fkp, dtype=torch.float64)

xc    = torch.load(py_dir + "/xc.pt")
yc    = torch.load(py_dir + "/yc.pt") #.reshape(xc.shape[0],xc.shape[1])
xc    = xc.view(1,1,xc.shape[0],xc.shape[1])
yc    = yc.view(1,1,yc.shape[0],yc.shape[1])

sdf = torch.zeros_like(yc)
sdf[:,:,0,:]  = 1.
sdf[:,:,-1,:] = 1.
sdf[:,:,:,0]  = 1.
sdf[:,:,:,-1] = 1.

take_every = 1
u  = torch.load(py_dir + "/e" + str(take_every) + "_uprev_data.pt")[[40,50,60,70,80,90,200,1000],...]
v  = torch.load(py_dir + "/e" + str(take_every) + "_vprev_data.pt")[[40,50,60,70,80,90,200,1000],...]

uv_max = max(torch.amax(torch.abs(u)), torch.amax(torch.abs(v)))
u /= uv_max
v /= uv_max

p  = torch.load(py_dir + "/e" + str(take_every) + "_pprev_data.pt")[[40,50,60,70,80,90,200,1000],...]
Tp = torch.load(py_dir + "/e" + str(take_every) + "_Tprev_data.pt")[[40,50,60,70,80,90,200,1000],...]

V  = eta_torch(fkt, fkp, 1.-yc, Tp)

x = torch.cat((sdf.expand(8,1,-1,-1),
               V,
               raq_nd.expand(8,1,Tp.shape[-2],Tp.shape[-1]),
               fkt_nd.expand(8,1,Tp.shape[-2],Tp.shape[-1]),
               fkp_nd.expand(8,1,Tp.shape[-2],Tp.shape[-1]),
               Tp), axis=1)
x = F.pad(x, (3,3,0,0), mode="replicate")
y = torch.cat((u,
               v,
               p), axis=1)
y = F.pad(y, (3,3,0,0), mode="replicate")

print(x.shape, y.shape)


for an in ["train"]:
    loader[an] = DataLoader(TensorDataset(x, y),  batch_size=8, shuffle=True)

# In[6]:


optimizer = torch.optim.Adam([
                    {"params": model_uvp.parameters(),
                     "lr": start_lr,
                     "weight_decay": 1e-5
                    },

                    #{"params": model_T.parameters(),
                    # "lr": start_lr,
                    # #"weight_decay": 1e-7
                    #}
])

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)


# In[7]:


while epoch<epochs:
    t0 = time.time()
    model_uvp.train(True)
    #model_T.train(True)
    avg_loss = one_epoch_AD(model_uvp, {}, epoch, loader["train"], None, 
                            optimizer, device, loss_type, is_train=True, weigh_loss=False)
    
    #model_T.eval()
    avg_vloss = avg_loss

    print("-------------------------------------------")
    print(epoch, avg_loss, avg_vloss, get_lr(optimizer))
    print("-------------------------------------------")
    
    torch.save(model_uvp.state_dict(), nn_dir + "fluidnet_uvp.pt")
    
    with open(nn_dir + "fluidnet_uvpT.txt", "a") as writer:
        writer.write(str(epoch) + "," + str(avg_loss) 
                     + "," + str(avg_vloss) + "," + str(get_lr(optimizer)) + "\n")
    scheduler.step()
    epoch += 1

    t1 = time.time()
    print("------------------------------")
    print("------------------------------")
    print(t1-t0)
    print("------------------------------")
    print("------------------------------")
