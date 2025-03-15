import glob, os, sys
from pathlib import Path

import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
from tabulate import tabulate
from pytorch_networks_convae import *
from prepare_gaia_ini import create_ini_file
import argparse
from datasetio import *
import copy
from scaler import *
import time

pathsave = "/plp_user/agar_sh/gaia-git/python/modules/"
sys.path.insert(0, pathsave)
from gaia import *
import pickle
from matplotlib.animation import FuncAnimation

from calculate_profiles import calc_mlp_profile
from scipy.signal import savgol_filter


# In[ ]:


data_dir = "/plp_scr1/agar_sh/data/TPH/"

pre = "/plp_user/agar_sh/PBML/pytorch/TPH/MLP/Paper/"
with open(pre + "los.pkl", "rb") as file:
    los = pickle.load(file)


# In[ ]:

run_cell = True
if run_cell:
    parser = argparse.ArgumentParser(description="Advect with GAIA")
    parser.add_argument(
        "-gpu", "--gpu_number", type=int, help="specify gpu number", default=-1
    )
    parser.add_argument("-f", "--c_h", type=int, help="filters", default=16)
    parser.add_argument("-s", "--use_symm", type=int, help="use symmetries", default=1)
    parser.add_argument("-l", "--levels", type=int, help="levels", default=6)
    parser.add_argument(
        "-r", "--repeats", type=int, help="repeats in hidden layers", default=4
    )
    parser.add_argument("-k", "--kernel", type=int, help="kernel size", default=5)
    parser.add_argument(
        "-w", "--warm_up_steps", type=int, help="warm_up_steps", default=0
    )
    parser.add_argument(
        "-i", "--intervene_TS", type=int, help="intervene_TS", default=1
    )
    parser.add_argument("-t", "--t_end", type=float, help="t_end", default=10)
    parser.add_argument("-m", "--mode", type=str, help="mode", default="GAIA")
    parser.add_argument(
        "-save", "--save_steps", type=int, help="save_steps", default=200
    )
    parser.add_argument(
        "-write", "--write_steps", type=int, help="write steps", default=200
    )
    parser.add_argument(
        "-ad", "--advection_scheme", type=int, help="advection_scheme", default=2
    )
    parser.add_argument("-raq", "--raq", type=float, help="raq")
    parser.add_argument("-fkt", "--fkt", type=float, help="fkt")
    parser.add_argument("-fkp", "--fkp", type=float, help="fkp")
    parser.add_argument("-pp", "--p_pred", type=int, help="predict pressure", default=0)
    parser.add_argument("-n", "--noise", type=float, help="noise level", default=0.0)
    parser.add_argument(
        "-lt", "--loss_type", type=str, help="loss type", default="curl"
    )
    parser.add_argument("-b", "--batch_size", type=str, help="loss type", default=16)
    parser.add_argument("-ad_loss", "--advect", type=int, help="advect loss", default=0)
    parser.add_argument("-l_sc", "--loss_scale", type=int, help="loss scale", default=1)
    parser.add_argument(
        "-l_de", "--loss_derivative", type=int, help="loss derivative", default=1
    )
    parser.add_argument("-l2", "--l2_reg", type=float, help="l2_reg", default=0.0)
    parser.add_argument(
        "-net", "--network", type=str, help="network name", default="none"
    )
    parser.add_argument("-fac", "--factor", type=int, help="scale factor", default=2)
    parser.add_argument("-pad", "--r_p", type=str, help="padding", default="learned")
    parser.add_argument("-e", "--epoch", type=int, help="load epoch", default=-1)
    parser.add_argument(
        "-cool", "--core_cool", type=int, help="core cooling", default=0
    )
    parser.add_argument(
        "-decay", "--radioactive_decay", type=int, help="radioactive decay", default=0
    )
    parser.add_argument(
        "-init", "--initialization", type=str, help="gaia initialization", default="hot"
    )
    parser.add_argument(
        "-sol", "--solver", type=str, help="momentum solver", default="mumps"
    )
    parser.add_argument(
        "-u", "--urf", type=float, help="urf if iterative momentum solver", default=1
    )
    parser.add_argument("-di", "--Di", type=float, help="dissipation number", default=0)

    args = parser.parse_args()

    gpu_number = args.gpu_number
    c_h = args.c_h
    use_symm = True if args.use_symm == 1 else False
    repeats = args.repeats
    kernel = args.kernel
    warm_up_steps = args.warm_up_steps
    intervene_TS = args.intervene_TS
    t_end = args.t_end
    modes = [args.mode]
    save_steps = args.save_steps
    write_steps = args.write_steps
    advection_scheme = args.advection_scheme
    raq = args.raq
    fkt = args.fkt
    fkp = args.fkp
    p_pred = True if args.p_pred == 1 else False
    noise = args.noise
    loss_type = args.loss_type
    batch_size = args.batch_size
    advect = True if args.advect == 1 else False
    loss_scale = True if args.loss_scale == 1 else False
    loss_derivative = True if args.loss_derivative == 1 else False
    levels = args.levels
    l2_reg = args.l2_reg
    network = args.network
    factor = args.factor
    r_p = args.r_p
    epoch = args.epoch
    initialization = args.initialization
    solver = args.solver
    urf = args.urf
    Di = args.Di
    radioactive_decay = True if args.radioactive_decay == 1 else False
    core_cool = True if args.core_cool == 1 else False

save_every = t_end / save_steps
write_every = t_end / write_steps

if args.mode == "GAIA":
    simulation = (
        "raq_"
        + str(raq)
        + "_fkt_"
        + str(fkt)
        + "_fkv_"
        + str(fkp)
        + "_mmskip"
        + str(intervene_TS)
        + "_sol"
        + solver
        + "_urf"
        + str(urf)
        + "_Di"
        + str(Di)
        + "_start"
        + initialization
    )
else:
    simulation = (
        network
        + "_raq_"
        + str(raq)
        + "_fkt_"
        + str(fkt)
        + "_fkv_"
        + str(fkp)
        + "_f"
        + str(c_h)
        + "_symm"
        + str(use_symm)
        + "_r"
        + str(repeats)
        + "_k"
        + str(kernel)
        + "_p"
        + str(p_pred)
        + "_l"
        + loss_type
        + "_b"
        + str(batch_size)
        + "_ad"
        + str(advect)
        + "_lev"
        + str(levels)
        + "_l_sc"
        + str(loss_scale)
        + "_l_de"
        + str(loss_derivative)
        + "_l2_"
        + str(l2_reg)
        + "_r_p"
        + r_p
        + "_Di"
        + str(Di)
        + "_start"
        + initialization
        + "_sol"
        + solver
    )

if core_cool:
    simulation += "_cool"
if radioactive_decay:
    simulation += "_decay"

CaseID = args.mode

# "GAIA"       : fully gaia
# "ML"         : fully ml with gaia every n steps
# "ML_STOKES"  : ml for stokes solve only
# "ML_PRE"     : ml stokes prediction with iterative solver

gaia_dir = "/plp_scr2/PLAGeS/CONVNN/GAIA_ML_RUNS/" + simulation + "/"
if not os.path.exists(gaia_dir):
    os.makedirs(gaia_dir)

y_pred_nn_pointwise, y_prof = calc_mlp_profile([raq], [fkt], [fkp], gaia_dir)

if solver == "mumps":
    fs = ["/gaiam/GaiaM", "/gaiam/libgaia.so", "ini"]
else:
    fs = ["/gaiac/GaiaC", "/gaiac/libgaia.so", "ini"]

for f in fs:
    f_o = f.split("/")[-1] if "/" in f else f

    if os.path.exists(gaia_dir + f_o):
        os.remove(gaia_dir + f_o)

    if f == "ini":
        os.symlink("/plp_scr2/PLAGeS/CONVNN/GAIA_ML_RUNS/" + f, gaia_dir + f_o)
    else:
        shutil.copyfile("/plp_scr2/PLAGeS/CONVNN/GAIA_ML_RUNS/" + f, gaia_dir + f_o)

f_gaia_ini = gaia_dir + "/Gaia.ini"

act_fn = "gelu"
dilation = 1
a_bound = 10
use_skip = False
scale = True
debug = False
spectral_conv = False
d_r = 0.0
blurr = False
dropout = 0.0

nn_dir = "/plp_user/agar_sh/PBML/pytorch/TPH/CONVNN/trained_networks/"

f_nn = (
    network
    + "_levels_"
    + str(levels)
    + "_"
    + act_fn
    + "_"
    + str(c_h)
    + "_"
    + r_p
    + "_"
    + loss_type
    + "_"
    + str(use_symm)
    + "_ab"
    + str(a_bound)
    + "_b"
    + str(batch_size)
    + "_r"
    + str(repeats)
    + "_k"
    + str(kernel)
    + "_fa"
    + str(factor)
    + "_ad"
    + str(advect)
    + "_p_pred"
    + str(p_pred)
    + "_l2"
    + str(l2_reg)
    + "_l_sc"
    + str(loss_scale)
    + "_l_de"
    + str(loss_derivative)
    + "_deb"
    + str(debug)
)

if network == "unet":
    f_nn += "_roll1"

nn_dir = nn_dir + f_nn + "/"


# In[ ]:


def get_model(c_h, repeats, kernel, use_symm, p_pred, network, factor=2):

    scale = True  # if loss_type == "mae" else False
    if "fluidnet" in network:
        c_i = 7
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

    if network == "newfluidnet":
        model_uvp = NewFluidNet(
            levels,
            c_i,
            c_h,
            c_o,
            device,
            act_fn,
            r_p,
            loss_type,
            use_symm=use_symm,
            dilation=dilation,
            a_bound=a_bound,
            repeats=repeats,
            use_skip=use_skip,
            f=kernel,
            p_pred=p_pred,
            spectral_conv=spectral_conv,
            blurr=blurr,
            drop_rate=dropout,
            factor=factor,
        ).double()
    elif network == "unet":
        model_uvp = Unet(
            levels,
            c_i,
            c_h,
            c_o,
            device,
            act_fn,
            r_p,
            loss_type,
            use_symm=use_symm,
            dilation=dilation,
            a_bound=a_bound,
            repeats=repeats,
            use_skip=use_skip,
            f=kernel,
            p_pred=p_pred,
            spectral_conv=spectral_conv,
            blurr=blurr,
            drop_rate=dropout,
        ).double()

    return model_uvp


# In[ ]:


colors = ["r-", "g-", "b-", "c-", "y-", "m-"]

nns = [f_nn]

if args.mode != "GAIA":
    last_epochs = []

    for counter, f_nn in enumerate(nns):

        nn_dir = "/plp_user/agar_sh/PBML/pytorch/TPH/CONVNN/trained_networks/"
        nn_dir = nn_dir + f_nn + "/"

        with open(nn_dir + "fluidnet_uvpT.txt") as fw:
            lines = fw.readlines()
        fw.close()
        loss_u = []
        loss_v = []
        loss_p = []
        loss_mass = []
        loss_cv_u = []
        loss_cv_v = []
        loss_cv_p = []
        loss_cv_mass = []

        for l in lines[1:]:
            ll = l[l.index("[") + 1 : l.index("],[")].split(",")
            l_r = l[l.index("],[") + 3 :]
            ll_cv = l_r[: l_r.index("],")].split(",")

            loss_u.append([float(ll[0])])
            loss_v.append([float(ll[1])])
            loss_p.append([float(ll[2])])
            loss_mass.append([float(ll[3]) + 1e-16])

            loss_cv_u.append([float(ll_cv[0])])
            loss_cv_v.append([float(ll_cv[1])])
            loss_cv_p.append([float(ll_cv[2])])
            loss_cv_mass.append([float(ll_cv[3]) + 1e-16])

        last_epochs.append(len(loss_u) - 1)


# In[ ]:

if "ML" in modes or "ML_STOKES" in modes or "ML_PRE" in modes:
    if torch.cuda.is_available() and gpu_number > -1:
        device = torch.device("cuda:" + str(gpu_number))
    else:
        device = torch.device("cpu")

    epoch = last_epochs[-1] - 1 if epoch == -1 else epoch

    model_uvp = (
        get_model(c_h, repeats, kernel, use_symm, p_pred, network).double().to(device)
    )
    print(count_parameters(model_uvp))

    model_uvp.load_state_dict(
        torch.load(nn_dir + str(epoch) + "_fluidnet_uvp.pt", map_location=device)
    )
    model_uvp.eval()
    print("loaded epoch " + str(epoch))


# In[ ]:

raq = torch.tensor(raq, dtype=torch.float64)
fkt = torch.tensor(fkt, dtype=torch.float64)
fkp = torch.tensor(fkp, dtype=torch.float64)
raq_nd = (raq - 0.12624371) / (9.70723344 - 0.12624371)
fkt_nd = (torch.log10(fkt) - 6.00352841978384) / (9.888820429862925 - 6.00352841978384)
fkp_nd = (torch.log10(fkp) - 0.005251646002323797) / (
    1.9927988938926755 - 0.005251646002323797
)
snapshots = {}
T_vec = {}
t_vec = {}
TS_vec = {}
scaler = (
    np.exp(
        (raq / 10) * 1.80167667 + np.log(fkt) * 0.4330392 + np.log(fkp) * -0.46052953
    )
    * 5
)


# In[ ]:

for mode in modes:

    if mode == "ML":
        model_AD = ADNet(device=device, CN_max=0.99).double().to(device)
        model_AD.eval()
        ts_net = (
            TS(
                model_uvp,
                model_AD,
                device=device,
                ts=1,
                advection_scheme=advection_scheme,
                scale=scale,
                p_pred=p_pred,
                net=network,
            )
            .double()
            .to(device)
        )
        print(ts_net)
        torch.compile(ts_net)
        ts_net.eval()

    elif mode == "ML_STOKES" or mode == "ML_PRE":
        model_AD = None
        ts_net = (
            TS(
                model_uvp,
                model_AD,
                device=device,
                ts=1,
                advection_scheme=advection_scheme,
                scale=scale,
                p_pred=p_pred,
                net=network,
            )
            .double()
            .to(device)
        )
        print(ts_net)
        torch.compile(ts_net)
        ts_net.eval()

    create_ini_file(
        f_gaia_ini,
        mode,
        raq.item(),
        fkt.item(),
        fkp.item(),
        advection_scheme,
        intervene_TS,
        warm_up_steps,
        solver=solver,
        initialization=initialization,
        urf=urf,
        Di=Di,
        core_cool=core_cool,
        radioactive_decay=radioactive_decay,
        CaseID=CaseID,
    )

    T_vec[mode] = []
    t_vec[mode] = []
    TS_vec[mode] = []

    snapshots[mode] = {}
    for var in ["v", "P", "T"]:
        snapshots[mode][var] = []
    t = 0
    n_step = 0

    os.chdir(gaia_dir)

    def attempt(t, n_step):

        sim = Direct()
        sim.init1()
        sim.iniLoad("ini/default.ini")
        sim.iniLoad(f_gaia_ini)
        sim.init2()

        state = sim.getState()
        T_vec[mode].append(np.copy(state["T"].mean()))
        t_vec[mode].append(np.copy(t))
        save_t = 0
        write_t = 0

        while n_step < warm_up_steps:
            n_step += 1
            dt = sim.doTimestep()
            state = sim.getState()

        for var in ["v", "P", "T"]:
            snapshots[mode][var].append(np.copy(state[var]))

        xcc = torch.tensor(np.copy(state["pos"][:, 0]), dtype=torch.float64).view(
            1, 1, 128, 506
        )
        ycc = torch.tensor(np.copy(state["pos"][:, 1]), dtype=torch.float64).view(
            1, 1, 128, 506
        )
        sdf = torch.zeros_like(xcc)
        sdf[:, :, 0, :] = 1.0
        sdf[:, :, -1, :] = 1.0
        sdf[:, :, :, 0] = 1.0
        sdf[:, :, :, -1] = 1.0
        sdf2 = torch.ones_like(xcc)
        sdf2[:, :, 0, :] = 0.0
        sdf2[:, :, -1, :] = 0.0
        sdf2[:, :, :, 0] = 0.0
        sdf2[:, :, :, -1] = 0.0

        snapshots[mode]["xcc"] = xcc
        snapshots[mode]["ycc"] = ycc

        state = sim.getState()
        Tp = torch.tensor(state["T"], dtype=torch.float64).view(1, 1, 128, 506)

        while t < t_end:
            n_step += 1
            t0 = time.time()

            if mode != "GAIA" and mode != "GAIA_PRECONDITIONER":

                t0_nn = time.time()
                with torch.no_grad():
                    T_new, dts, u, v, p, V = ts_net(
                        Tp, sdf, sdf2, ycc, raq_nd, fkt_nd, fkp_nd, raq, fkt, fkp, xcc, ycc
                    )

                u = u.detach().cpu().numpy()
                v = v.detach().cpu().numpy()
                V = V.detach().cpu().numpy()

                # u[0,0,:,1]  = savgol_filter(u[0,0,:,1],  11, 3)
                # u[0,0,:,-2] = savgol_filter(u[0,0,:,-2], 11, 3)
                # v[0,0,1,:]  = savgol_filter(v[0,0,1,:],  21, 3)

                state["v"][:, :] = np.copy(
                    np.concatenate(
                        (
                            u.reshape(-1, 1),
                            v.reshape(-1, 1),
                            np.zeros_like(u.reshape(-1, 1)),
                        ),
                        axis=1,
                    )
                )
                if p_pred:
                    p = p.detach().cpu().numpy().flatten()
                    state["P"][:] = np.copy(p)
                state["V"][:] = np.copy(V.flatten())

                if mode != "ML" or n_step % intervene_TS == 0:
                    dt = sim.doTimestep()
                    state = sim.getState()
                    Tp = torch.tensor(state["T"], dtype=torch.float64).view(
                        1, 1, 128, 506
                    )
                    if not core_cool:
                        Tp[:, :, 0, :] = 1.0
                    Tp[:, :, -1, :] = 0.0
                    Tp[:, :, :, 0] = Tp[:, :, :, 1]
                    Tp[:, :, :, -1] = Tp[:, :, :, -2]
                    Tp = torch.clip(Tp, 0.0, 2.0)
                    state["T"][:] = np.copy(Tp.detach().cpu().numpy().flatten())

                if mode == "ML" and n_step % intervene_TS != 0:
                    Tnew = T_new[1].clone().detach().cpu().numpy().flatten()
                    dt = dts[1].clone().detach().cpu().numpy()
                    state["T"][:] = np.copy(Tnew)

                state["raw"].time = np.copy(t)
                t1_nn = time.time()

                # print("NN inference took ", t1_nn-t0_nn)
            else:
                dt = sim.doTimestep()
                state = sim.getState()

            t += dt

            T_vec[mode].append(np.copy(state["T"].mean()))
            t_vec[mode].append(np.copy(t))

            t1 = time.time()
            TS = t1 - t0
            TS_vec[mode].append(TS)

            if t > save_t:
                save_t = t + save_every
                for var in ["v", "P", "T"]:
                    snapshots[mode][var].append(np.copy(state[var]))

            if t > write_t:
                write_t = t + write_every
                with open(gaia_dir + "snapshots_" + mode + ".pkl", "wb") as file:
                    pickle.dump(snapshots[mode], file)
                with open(gaia_dir + "TS_vec_" + mode + ".pkl", "wb") as file:
                    pickle.dump(TS_vec[mode], file)
                with open(gaia_dir + "t_vec_" + mode + ".pkl", "wb") as file:
                    pickle.dump(t_vec[mode], file)
                with open(gaia_dir + "T_vec_" + mode + ".pkl", "wb") as file:
                    pickle.dump(T_vec[mode], file)

        with open(gaia_dir + "snapshots_" + mode + ".pkl", "wb") as file:
            pickle.dump(snapshots[mode], file)
        with open(gaia_dir + "TS_vec_" + mode + ".pkl", "wb") as file:
            pickle.dump(TS_vec[mode], file)
        with open(gaia_dir + "t_vec_" + mode + ".pkl", "wb") as file:
            pickle.dump(t_vec[mode], file)
        with open(gaia_dir + "T_vec_" + mode + ".pkl", "wb") as file:
            pickle.dump(T_vec[mode], file)

        return t, n_step

    def attempt_unet(t, n_step):

        sim = Direct()
        sim.init1()
        sim.iniLoad("ini/default.ini")
        sim.iniLoad(f_gaia_ini)
        sim.init2()

        state = sim.getState()
        T_vec[mode].append(np.copy(state["T"].mean()))
        t_vec[mode].append(np.copy(t))
        save_t = 0
        write_t = 0

        while n_step < warm_up_steps:
            n_step += 1
            dt = sim.doTimestep()
            state = sim.getState()

        for var in ["v", "P", "T"]:
            snapshots[mode][var].append(np.copy(state[var]))

        xcc = torch.tensor(np.copy(state["pos"][:, 0]), dtype=torch.float64).view(
            1, 1, 128, 506
        )
        ycc = torch.tensor(np.copy(state["pos"][:, 1]), dtype=torch.float64).view(
            1, 1, 128, 506
        )
        sdf = torch.zeros_like(xcc)
        sdf[:, :, 0, :] = 1.0
        sdf[:, :, -1, :] = 1.0
        sdf[:, :, :, 0] = 1.0
        sdf[:, :, :, -1] = 1.0
        sdf2 = torch.ones_like(xcc)
        sdf2[:, :, 0, :] = 0.0
        sdf2[:, :, -1, :] = 0.0
        sdf2[:, :, :, 0] = 0.0
        sdf2[:, :, :, -1] = 0.0

        snapshots[mode]["xcc"] = xcc
        snapshots[mode]["ycc"] = ycc

        state = sim.getState()
        Tp = torch.tensor(state["T"], dtype=torch.float64).view(1, 1, 128, 506)
        up = (
            torch.tensor(state["v"][:, 0], dtype=torch.float64).view(1, 1, 128, 506)
            / scaler
        )
        vp = (
            torch.tensor(state["v"][:, 1], dtype=torch.float64).view(1, 1, 128, 506)
            / scaler
        )

        while t < t_end:
            n_step += 1
            t0 = time.time()

            state = sim.getState()
            dx_min = 0.5 / 126.0
            uv_mag = torch.max(
                torch.amax(torch.abs(up * scaler)), torch.amax(torch.abs(vp * scaler))
            )
            dt_advect = 0.5 * 100.0 * dx_min / uv_mag  # 0.5 * CN_max
            dt_diffuse = 0.5 * ((dx_min * dx_min) ** 2) / (dx_min**2 + dx_min**2)
            dt = torch.tensor(
                np.min([dt_advect.item(), dt_diffuse]), dtype=torch.float64
            ).expand(1, 1, vp.shape[-2], vp.shape[-1])

            t0_nn = time.time()
            with torch.no_grad():
                T_new, dts, u, v, p, V = ts_net(
                    Tp,
                    sdf,
                    sdf2,
                    ycc,
                    raq_nd,
                    fkt_nd,
                    fkp_nd,
                    raq,
                    fkt,
                    fkp,
                    xcc,
                    ycc,
                    up,
                    vp,
                    dt,
                )

            u = u.detach().cpu().numpy()
            v = v.detach().cpu().numpy()
            V = V.detach().cpu().numpy()
            T_new = T_new[1]

            state["v"][:, :] = np.copy(
                np.concatenate(
                    (
                        u.reshape(-1, 1),
                        v.reshape(-1, 1),
                        np.zeros_like(u.reshape(-1, 1)),
                    ),
                    axis=1,
                )
            )
            state["V"][:] = np.copy(V.flatten())

            if not core_cool:
                T_new[:, :, 0, :] = 1.0
            T_new[:, :, -1, :] = 0.0
            T_new[:, :, :, 0] = T_new[:, :, :, 1]
            T_new[:, :, :, -1] = T_new[:, :, :, -2]
            T_new = torch.clip(T_new, 0.0, 2.0)
            state["T"][:] = np.copy(T_new.detach().cpu().numpy().flatten())

            state["raw"].time = np.copy(t)
            t1_nn = time.time()

            t += dt[0, 0, 0, 0].item()

            T_vec[mode].append(np.copy(state["T"].mean()))
            t_vec[mode].append(np.copy(t))

            t1 = time.time()
            TS = t1 - t0
            TS_vec[mode].append(TS)

            print(t_vec[mode][-1], T_vec[mode][-1])

            if t > save_t:
                save_t = t + save_every
                for var in ["v", "P", "T"]:
                    snapshots[mode][var].append(np.copy(state[var]))

            if t > write_t:
                write_t = t + write_every
                with open(gaia_dir + "snapshots_" + mode + ".pkl", "wb") as file:
                    pickle.dump(snapshots[mode], file)
                with open(gaia_dir + "TS_vec_" + mode + ".pkl", "wb") as file:
                    pickle.dump(TS_vec[mode], file)
                with open(gaia_dir + "t_vec_" + mode + ".pkl", "wb") as file:
                    pickle.dump(t_vec[mode], file)
                with open(gaia_dir + "T_vec_" + mode + ".pkl", "wb") as file:
                    pickle.dump(T_vec[mode], file)

        with open(gaia_dir + "snapshots_" + mode + ".pkl", "wb") as file:
            pickle.dump(snapshots[mode], file)
        with open(gaia_dir + "TS_vec_" + mode + ".pkl", "wb") as file:
            pickle.dump(TS_vec[mode], file)
        with open(gaia_dir + "t_vec_" + mode + ".pkl", "wb") as file:
            pickle.dump(t_vec[mode], file)
        with open(gaia_dir + "T_vec_" + mode + ".pkl", "wb") as file:
            pickle.dump(T_vec[mode], file)

        return t, n_step

    if network == "unet":
        t, n_step = attempt_unet(t, n_step)
    else:
        t, n_step = attempt(t, n_step)
