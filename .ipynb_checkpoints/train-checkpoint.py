import glob, os, sys

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
from tabulate import tabulate
from pytorch_networks_convae import *
import argparse
from datasetio import *
from multigpu import *
#from Transolver_Structured_Mesh_2D import Model as transolver_structured
#from Transolver import Model as transolver

import time

torch.backends.cudnn.benchmark = True


# In[2]:


data_dir = "/plp_scr1/agar_sh/data/TPH/"


# In[3]:


run_cell = True
if run_cell:
    parser = argparse.ArgumentParser(description='Train convnet')
    parser.add_argument("-a", "--act_fn", type=str, help ="activation function", default="gelu")
    parser.add_argument("-l", "--levels", type=int, help ="levels", default=6)
    parser.add_argument("-f", "--c_h", type=int, help ="filters")
    parser.add_argument("-p", "--r_p", type=str, help="padding type", default="replicate")
    parser.add_argument("-lt", "--loss_type", type=str, help="loss type")
    parser.add_argument("-d", "--dilation", type=int, help="loss type", default=1)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size")
    parser.add_argument("-s", "--use_symm", type=int, help="use symmetries")
    parser.add_argument("-ab", "--a_bound", type=int, help="bound of a")
    parser.add_argument("-r", "--repeats", type=int, help="repeats in hidden layers")
    parser.add_argument("-rst", "--restart", type=int, help="restart", default=0)
    parser.add_argument("-sk", "--use_skip", type=int, help="use skip connections", default=0)
    parser.add_argument("-k", "--kernel", type=int, help="kernel size")
    parser.add_argument("-sc", "--scale", type=int, help="use scaling", default=1)
    parser.add_argument("-l_sc", "--loss_scale", type=int, help="scale loss", default=0)
    parser.add_argument("-pp", "--p_pred", type=int, help="predict pressure")
    parser.add_argument("-ad", "--advect", type=int, help="add advection to loss")
    parser.add_argument("-n", "--noise", type=float, help="noise level", default=0.0)
    parser.add_argument("-deb", "--debug", type=int, help="debugging")
    parser.add_argument("-net", "--network", type=str, help="neural network model", default="fluidnet")
    parser.add_argument("-spectral", "--spectral_conv", type=int, help="use spectral conv", default=False)
    parser.add_argument("-gpu", "--gpu_nums", type=str)
    args = parser.parse_args()
    
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
    scale = True if args.scale == 1 else False
    p_pred = True if args.p_pred == 1 else False
    debug = True if args.debug == 1 else False
    spectral_conv = True if args.spectral_conv == 1 else False
    advect = True if args.advect == 1 else False
    loss_scale = True if args.loss_scale == 1 else False
    repeats = args.repeats
    kernel = args.kernel
    noise = args.noise
    network = args.network
    
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
    scale = True
    p_pred = True
    noise = 0.0
    debug = True
    network = "fluidnet"


os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_nums
device = torch.device('cuda', torch.cuda.current_device())
#device = torch.device("cpu")

torch.cuda.empty_cache()
nn_dir = "/plp_user/agar_sh/PBML/pytorch/TPH/CONVNN/trained_networks/"
f_nn   =    network + "_levels_" + str(levels) + "_" + act_fn + \
            "_" + str(c_h) + "_" + r_p + "_" + loss_type + "_dil_" + str(dilation) +  \
            "_" + str(use_symm) + "_ab" + str(a_bound) + "_b" + str(batch_size) + \
            "_r" + str(repeats) + "_k" + str(kernel) + \
            "_ad" + str(advect) + "_p_pred" + str(p_pred) + "_deb" + str(debug) 

if torch.cuda.device_count()>1:
    f_nn += "_multi" 

if debug:
    epochs      = 1000
    milestones  = [25, 200, 400, 600, 800]
else:
    epochs      = 80
    milestones  = [5, 20, 35, 50, 65]

if loss_type == "curl":
    c_i = 7
    c_o = 2
else:
    c_i = 7
    c_o = 3

if not p_pred:
    c_o -= 1
    
if network == "fluidnet":
    model_uvp = FluidNet(levels, c_i, c_h, c_o, device, act_fn, r_p, loss_type, 
                         use_symm=use_symm, dilation=dilation, a_bound=a_bound,
                        repeats=repeats, use_skip=use_skip, f=kernel, p_pred=p_pred,
                        spectral_conv=spectral_conv).double().to(device)
elif network == "transolver":
    model_uvp = transolver(n_hidden=256, n_layers=8, space_dim=2,
                  fun_dim=c_i-2,
                  n_head=8,
                  mlp_ratio=2, out_dim=c_o,
                  slice_num=32,
                  unified_pos=0).double().to(device)
elif network == "transolver_structured":
    model_uvp = transolver_structured(device=device,
                                  space_dim=2,
                                  n_layers=repeats,
                                  n_hidden=c_h,
                                  dropout=0.05,
                                  n_head=8,
                                  Time_Input=False,
                                  mlp_ratio=1,
                                  fun_dim=c_i-2,
                                  out_dim=c_o,
                                  slice_num=32,
                                  ref=8,
                                  unified_pos=1,
                                  H=128, 
                                  W=506,
                                  kernel=kernel).double().to(device)
else:
    raise Exception("Model not available")
    
torch.compile(model_uvp)
print(count_parameters(model_uvp))
print(model_uvp)

if advect:
    model_AD = ADNet(device=device, CN_max=0.9).double().to(device)
    model_AD.eval()
else:
    model_AD = None

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
    if epoch>milestones[-1]:
        milestones = []
    else:
        start_milestone_ind = np.where(np.asarray(milestones)>epoch)[0][0]
        milestones = [milestones[start_milestone_ind]-epoch] + [m-epoch for m in milestones[start_milestone_ind+1:]]
    
    model_uvp.load_state_dict(torch.load(nn_dir + str(epoch) + "_fluidnet_uvp.pt", map_location=device))
    epoch += 1
    print("Restarting from epoch, lr, milestones")
    print(epoch, start_lr, milestones)
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

for an in ["train", "cv"]:
    if network == "fluidnet":
        dataset_pytorch = ADDataset
    elif network == "transolver" or network == "transolver_structured":
        dataset_pytorch = UnstructuredDataset
        
    dataset[an] = dataset_pytorch(data_dir, an, scale, is_init=False, p_pred=p_pred, noise=noise, debug=debug)
    batches[an] = int(len(dataset[an])/batch_size)
    
    if debug:
        loader[an]  = DataLoader(dataset[an], batch_size=batch_size, shuffle=True, pin_memory=True)
        print(an, batches[an])
        loader_init[an] = None
    else:
        loader[an]  = DataLoader(dataset[an], batch_size=batch_size-init_samples, shuffle=True, pin_memory=True)
        print(an, batches[an])
        
        dataset_init[an] = dataset_pytorch(data_dir, an, scale, is_init=True, p_pred=p_pred, noise=noise, debug=debug)
        batches_init[an] = int(len(dataset_init[an])/init_samples)
        loader_init[an]  = DataLoader(dataset_init[an], batch_size=init_samples, shuffle=True, pin_memory=True)
        print("init samples ", an, batches_init[an])

# In[6]:


optimizer = torch.optim.Adam([
                    {"params": model_uvp.parameters(),
                     "lr": start_lr,
                     #"weight_decay": 1e-5
                    },

                    #{"params": model_T.parameters(),
                    # "lr": start_lr,
                    # #"weight_decay": 1e-7
                    #}
])

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    


while epoch<epochs:
    t0 = time.time()
    model_uvp.train(True)
    #model_T.train(True)
    avg_loss = one_epoch_AD(model_uvp, model_AD, epoch, loader["train"], loader_init["train"], 
                            optimizer, device, loss_type, is_train=True, p_pred=p_pred, loss_scale=loss_scale)
    
    model_uvp.eval()
    #model_T.eval()
    if not debug:
        with torch.no_grad():
            avg_vloss = one_epoch_AD(model_uvp, model_AD, epoch, loader["cv"], loader_init["cv"], 
                                     optimizer, device, loss_type, is_train=False, p_pred=p_pred, loss_scale=loss_scale)
    else:
        avg_vloss = [0.0]*5
    
    t1 = time.time()
    
    if debug:
        torch.save(model_uvp.state_dict(), nn_dir + "fluidnet_uvp.pt")
    else:
        torch.save(model_uvp.state_dict(), nn_dir + str(epoch) + "_fluidnet_uvp.pt")
        
        print("-------------------------------------------")
        print(epoch, avg_loss, avg_vloss, get_lr(optimizer))
        print("-------------------------------------------")
        print("------------------------------")
        print("------------------------------")
        print(t1-t0)
        print("------------------------------")
        print("------------------------------")
    
    with open(nn_dir + "fluidnet_uvpT.txt", "a") as writer:
        writer.write(str(epoch) + "," + str(avg_loss) 
                     + "," + str(avg_vloss) + "," + str(get_lr(optimizer)) + "\n")
    scheduler.step()
    epoch += 1
