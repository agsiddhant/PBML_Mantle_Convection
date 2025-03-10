from matplotlib import pyplot as plt
import warnings
import pickle
import numpy as np

def selu(x):
    alpha = 1.6732632423543772848170429916717 
    scale = 1.0507009873554804934193349852946
    return scale*( np.maximum(0,x) + np.minimum(alpha*(np.exp(x)-1), 0) )

def non_dimensionalize_raq(x):
    return (x-0.12624371)/(9.70723344-0.12624371)

def non_dimensionalize_fkt(x):
    return (np.log10(x)-6.00352841978384)/(9.888820429862925-6.00352841978384)

def non_dimensionalize_fkv(x):
    return (np.log10(x)-0.005251646002323797)/(1.9927988938926755-0.005251646002323797)

def dimensionalize_raq(x):
    return x*(9.70723344-0.12624371) + 0.12624371

def dimensionalize_fkt(x):
    return 10**(x*(9.888820429862925-6.00352841978384)+6.00352841978384)

def dimensionalize_fkv(x):
    return 10**(x*(1.9927988938926755-0.005251646002323797)+0.005251646002323797)

def get_input(raq_ra, fkt, fkp, y_prof):

    x = np.zeros((len(raq_ra)*len(y_prof), 4))
    
    cntr = 0
    for i in range(len(raq_ra)):
        for j in range(len(y_prof)):
            x[cntr,0] = non_dimensionalize_raq(raq_ra[i])
            x[cntr,1] = non_dimensionalize_fkt(fkt[i])
            x[cntr,2] = non_dimensionalize_fkv(fkp[i])
            x[cntr,3] = y_prof[j]
            cntr += 1      
            
    return x

def get_profile(inp, mlp, num_sims=1, correction=True, prof_points=128):
    # get predicted profile based on input

    # forward network pass in pure python using saved weights
    num_layers = len(mlp)-1
    y_pred = inp
    res = []
    for l in range(num_layers+1):
        
        y_pred = y_pred @ mlp[l][0].T + mlp[l][1]

        if l in [num_layers-1]:
            y_pred = np.concatenate((inp,y_pred), axis=-1)
            
        if l != num_layers:
            for r in res:
                y_pred += r
                
            y_pred = selu(y_pred)
            res.append(y_pred)
                
    y_pred = y_pred.reshape(num_sims, -1)

    # overwrite points at the boundary
    y_pred[:,0]  = 1.
    y_pred[:,-1] = 0.

    if correction:   # boundary layer corection
        inp = inp.reshape(num_sims, -1, inp.shape[-1])
    
        for sim_ind in range(num_sims):
            inds = np.where(inp[sim_ind,:,3] < 0.04)[0]
            slope = (0 - y_pred[sim_ind,inds[0]])/(0 - inp[sim_ind,inds[0],3:4])
            y_pred[sim_ind,inds] = slope*inp[sim_ind,inds,3]  

            inds = np.where(inp[sim_ind,:,3] > 0.985)[0]
            x_old = [inp[sim_ind,inds[-1],3], 1]
            y_old = [y_pred[sim_ind,inds[-1]], 1]
            y_pred[sim_ind,inds] = np.interp(inp[sim_ind,inds,3:4], x_old, y_old).flatten()

    return y_pred


def calc_mlp_profile(r_list, t_list, v_list, simulation_dir=None, num_points = 128):

    with open('mlp_[128, 128, 128, 128, 128].pkl', 'rb') as file: 
        mlp = pickle.load(file) 
    y_prof = np.concatenate(([1], np.linspace(0+1/(num_points*2),1-1/(num_points*2),num_points-2)[::-1], [0]))
    
    x_in = get_input(r_list, t_list, v_list, y_prof)
    y_pred_nn_pointwise = get_profile(x_in, mlp, num_sims=len(r_list)) 

    if simulation_dir is not None:
        for i in range(len(r_list)):
            fname = simulation_dir + "/ml_prof"
            f = open(fname + ".txt", "wb")
            for j in range(len(y_prof)):
                f.writelines([str(y_prof[j]).encode('ascii'), 
                              "   ".encode('ascii'), 
                              str(y_pred_nn_pointwise[i,j]).encode('ascii'), 
                              "\n".encode('ascii')])
            f.close()

    return y_pred_nn_pointwise, y_prof