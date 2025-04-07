import glob, os, sys

import numpy as np
import matplotlib.pyplot as plt

import torch
torch.serialization.add_safe_globals
from torch.utils.data import TensorDataset, DataLoader
#from tabulate import tabulate
from pytorch_networks_convae import *
import argparse
from datasetio import *
import copy
from scaler import *
import time
import pickle 


# define here
data_dir = ""
nn_dir = ""
# or
# import from a file
from paths import *


sims = torch.load(data_dir + "/sims.pt", weights_only=False)

eval_batch_size = 1
gpu_number = 0
device = torch.device("cuda:" + str(gpu_number)) if torch.cuda.is_available() else torch.device("cpu")
for an in ["test"]:
    for si, sim in enumerate(sims):
        ignr, ignr, raq, fkt, fkp, gr, ar, ignr = sim
        check = (sim[1] == an and si==118) # an
        
        d raq in [0.80523448,  9.17743012, 8.75081696]) 
        if check:
            print(tabulate([["num", "dataset", "raq", "fkt", "fkp", "gr", "ar"],
                            sim[:-1]
                           ]))
    
            py_dir = data_dir + "/" + sim[1] + "/sim_" + str(sim[0])
                
            raq_nd = torch.tensor((raq-0.12624371)/(9.70723344-0.12624371), dtype=torch.float64)
            fkt_nd = torch.tensor((np.log10(fkt)-6.00352841978384)/(
                9.888820429862925-6.00352841978384), dtype=torch.float64)
            fkp_nd = torch.tensor((np.log10(fkp)-0.005251646002323797)/(
                1.9927988938926755-0.005251646002323797), dtype=torch.float64)
    
            fkt = torch.tensor(fkt, dtype=torch.float64)
            fkp = torch.tensor(fkp, dtype=torch.float64)
            
            xcc    = torch.load(py_dir + "/xc.pt", weights_only=False)
            ycc    = torch.load(py_dir + "/yc.pt", weights_only=False)
            xcc    = xcc.view(1,1,xcc.shape[0],xcc.shape[1])
            ycc    = ycc.view(1,1,ycc.shape[0],ycc.shape[1])
    
            xcc[:,:,:,0]  = 0.0
            xcc[:,:,:,-1] = 4.0
            ycc[:,:,0,:]  = 0.0
            ycc[:,:,-1,:] = 1.0
    
            sdf = torch.zeros_like(ycc)
            sdf2 = torch.ones_like(ycc)
            
            take_every = 1
            u     = torch.load(py_dir + "/e" + str(take_every) + "_uprev_data.pt", weights_only=False)
            v     = torch.load(py_dir + "/e" + str(take_every) + "_vprev_data.pt", weights_only=False)
            Tprev = torch.load(py_dir + "/e" + str(take_every) + "_Tprev_data.pt", weights_only=False)
            
            i_vec = np.arange(1, u.shape[0], eval_batch_size*5)    
            print(i_vec)
            for r_p, c_h, batch_size, repeats in [ 
                                                   
                                                   ["zeros", 64, 8, 4] ,
                                                    ["zeros", 16, 16, 6],
                                                    ["learned", 16, 16, 6],
                                                ]:
            
                act_fn = "gelu"
                
                network = "newfluidnet"
                levels = 5
                kernel = 5
                epoch  = 81
                blurr = False
                debug = False
                loss_scale = True
                loss_derivative = True
                loss_type = "curl"
                a_bound   = 10
                use_symm  = False 
                factor = 2
                
                l2  = 0.0
                d_r = 0.0
                
                dilation = 1
                use_skip = False
                scale = True
                p_pred = False
                noise = 0.0
                
                advect = False
                spectral_conv = False
                                
                f_nn   =    network + "_levels_" + str(levels) + "_" + act_fn + \
                            "_" + str(c_h) + "_" + r_p + "_" + loss_type +  \
                            "_" + str(use_symm) + "_ab" + str(a_bound) + "_b" + str(batch_size) + \
                            "_r" + str(repeats) + "_k" + str(kernel) + "_fa" + str(factor) + \
                            "_ad" + str(advect) + "_p_pred" + str(p_pred) + \
                            "_l2" + str(l2) + "_l_sc" + str(loss_scale) + "_l_de" + str(loss_derivative) + "_deb" + str(debug)  
                
                if blurr:
                    f_nn += "_blurr"
                    
                _nn_dir = nn_dir + f_nn + "/"
                
                if network=="fluidnet" or network=="newfluidnet":
                    c_i = 7
                    c_o = 3
                elif network=="newfluidnet":
                    c_i = 7
                    c_o = 3
                elif network == "ifluidnet":
                    c_i = 9
                    c_o = 3
                elif network == "convae":
                    c_i = 3
                    c_o = 3
                elif network == "unet":
                    c_i = 11
                    c_o = 4
                    if not p_pred:
                        c_i -= 1
                
                if loss_type == "curl":
                    c_o -= 1
                if not p_pred:
                    c_o -= 1
                
                if network=="fluidnet" or network == "ifluidnet":
                    model_uvp = FluidNet(levels, c_i, c_h, c_o, device, act_fn, r_p, loss_type, 
                                         use_symm=use_symm, dilation=dilation, a_bound=a_bound,
                                         repeats=repeats, use_skip=use_skip, f=kernel, p_pred=p_pred, blurr=blurr).double().to(device)
                
                    ts = 1
                    ts_net = TS(model_uvp, ad=None, device=device, ts=ts, advection_scheme=0, 
                                scale=scale, p_pred=p_pred, net=network).double().to(device)
                
                elif network=="newfluidnet":
                    model_uvp = NewFluidNet(levels, c_i, c_h, c_o, device, act_fn, r_p, loss_type, 
                                         use_symm=use_symm, dilation=dilation, a_bound=a_bound,
                                         repeats=repeats, use_skip=use_skip, f=kernel, p_pred=p_pred, 
                                            blurr=blurr, factor=factor).double().to(device)
                    
                    ts = 1
                    ts_net = TS(model_uvp, ad=None, device=device, ts=ts, advection_scheme=0, 
                                scale=scale, p_pred=p_pred, net=network).double().to(device)
                
                        
                elif network == "unet":
                    model_uvp = Unet(levels, c_i, c_h, c_o, device, act_fn, r_p, loss_type, 
                                         use_symm=use_symm, dilation=dilation, a_bound=a_bound,
                                         repeats=repeats, use_skip=use_skip, f=kernel, p_pred=p_pred).double()
                
                print(count_parameters(model_uvp))
                
                model_uvp.load_state_dict(torch.load(_nn_dir + str(epoch) + "_fluidnet_uvp.pt", map_location=device, weights_only=True))
                
                torch.compile(model_uvp)
                model_uvp.eval()
        
                
                u_diff_l = []
                u_diff_r = []
                u_diff_t = []
                u_diff_b = []
                u_diff_base = []
                v_diff_l = []
                v_diff_r = []
                v_diff_t = []
                v_diff_b = []
                v_diff_base = []
                
        
                for i in i_vec:
                    print(f"Processing batch starting at index {i}/{i_vec[-1]}")
                    Tp = Tprev[i:i+eval_batch_size, ...]  # Slice a batch
                            
                    with torch.no_grad():
                        V = eta_torch(fkt.to(device),
                                      fkp.to(device),
                                  1.0 - ycc.to(device).expand(Tp.shape[0],-1,-1,-1),
                                  Tp.to(device),
                                  0,
                                  0
                                 ).to(device)
                        print(V.shape)
                        inp = torch.cat(
                            (
                                xcc.to(device).expand(Tp.shape[0],-1,-1,-1) / 4.0,
                                ycc.to(device).expand(Tp.shape[0],-1,-1,-1) / 4.0,
                                torch.log10(V) / 8.0,
                                raq_nd.expand(*Tp.shape).to(
                                    device
                                ),
                                fkt_nd.expand(*Tp.shape).to(
                                    device
                                ),
                                fkp_nd.expand(*Tp.shape).to(
                                    device
                                ),
                                Tp.to(device),
                            ),
                            axis=1,
                        )

                        u_pred, v_pred, p_pred = model_uvp(inp)

                        u_pred = unscale_var(u_pred, raq, fkt, fkp, "uprev")
                        v_pred = unscale_var(v_pred, raq, fkt, fkp, "vprev")
    
                        u_pred = u_pred.view(-1, 1, 128, 506)
                        v_pred = v_pred.view(-1, 1, 128, 506)
        
                    u_diff = (u_pred.detach().cpu().numpy()-u[i:i+eval_batch_size,...].numpy())
                    v_diff = (v_pred.detach().cpu().numpy()-v[i:i+eval_batch_size,...].numpy())
                    ub_diff = (u[i-1:i+eval_batch_size-1,...]-u[i:i+eval_batch_size,...]).numpy()
                    vb_diff = (v[i-1:i+eval_batch_size-1,...]-v[i:i+eval_batch_size,...]).numpy()
        
                    [u_diff_l.append(x) for x in np.mean(np.abs(u_diff[:,:,:,0]), axis=(1,2))]
                    [u_diff_r.append(x) for x in np.mean(np.abs(u_diff[:,:,:,-1]), axis=(1,2))]
                    [u_diff_t.append(x) for x in np.mean(np.abs(u_diff[:,:,-1,:]), axis=(1,2))]
                    [u_diff_b.append(x) for x in np.mean(np.abs(u_diff[:,:,0,:]), axis=(1,2))]
                    [u_diff_base.append(x) for x in np.mean(np.abs(ub_diff), axis=(1,2,3))/np.mean(np.abs(u_diff), axis=(1,2,3))]
        
                    [v_diff_l.append(x) for x in np.mean(np.abs(v_diff[:,:,:,0]), axis=(1,2))]
                    [v_diff_r.append(x) for x in np.mean(np.abs(v_diff[:,:,:,-1]), axis=(1,2))]
                    [v_diff_t.append(x) for x in np.mean(np.abs(v_diff[:,:,-1,:]), axis=(1,2))]
                    [v_diff_b.append(x) for x in np.mean(np.abs(v_diff[:,:,0,:]), axis=(1,2))]
                    [v_diff_base.append(x) for x in np.mean(np.abs(vb_diff), axis=(1,2,3))/np.mean(np.abs(v_diff), axis=(1,2,3))]
                    
                with open("Paper/FiguresData/" + network + r_p + str(c_h) + "_mae_sim" + str(sim[0]) + ".pkl", "wb") as file: 
                    pickle.dump([u_diff_l,u_diff_r,u_diff_t,u_diff_b,u_diff_base,
                                 v_diff_l,v_diff_r,v_diff_t,v_diff_b,v_diff_base], file) 

