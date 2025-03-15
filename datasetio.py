import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scaler import *
from tabulate import tabulate
import random
import torch.nn.functional as F
from pytorch_networks_convae import *
import copy


def get_sdf(x, y):
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    sdf = torch.minimum(
        torch.minimum(torch.abs(x - x_min), torch.abs(x - x_max)),
        torch.minimum(torch.abs(y - y_min), torch.abs(y - y_max)),
    )
    return sdf


def eta(gamma, beta, z, T, Tref=0, zref=0):
    eta = np.exp(np.log(gamma) * (Tref - T) + np.log(beta) * (z - zref))
    return eta


def get_indices_time(data_dir, an, is_init=False, debug=True, roll_forward=1):

    sims = torch.load(data_dir + "/sims.pt")
    ignore_inds = [8, 39]
    sims_vec = []
    times_vec = []
    for si, sim in enumerate(sims):
        if an == "train":
            check = sim[1] == "train"
        else:
            check = sim[1] == "cv"
        ignr, ignr, raq, fkt, fkp, gr, ar, ignr = sim
        py_dir = data_dir + "/" + sim[1] + "/sim_" + str(sim[0])
        if check and si not in ignore_inds:  # and (si==0 or si == 110):
            take_every = 1
            if debug:
                u = torch.load(
                    py_dir + "/e" + str(take_every) + "_uprev_data_select_init.pt"
                ).repeat(roll_forward * 2, 1, 1, 1)
                times = torch.load(py_dir + "/times.pt")[: u.shape[0]]
            else:
                # u      = torch.load(py_dir + "/e" + str(take_every) + "_uprev_data.pt")[:500,...]
                times = torch.load(py_dir + "/times.pt")[:750, ...]
                times = times[:-2]

            for i, t in enumerate(times):
                if i < len(times) - roll_forward - 1:
                    sims_vec.append(sim[0])
                    times_vec.append(t)

    return sims_vec, times_vec


class ADTimeDataset(Dataset):

    def __init__(
        self,
        data_dir,
        an,
        scale=True,
        load=False,
        is_init=False,
        p_pred=True,
        noise=0.0,
        debug=True,
        sims_vec=[],
        times_vec=[],
        roll_forward=1,
    ):

        sims = torch.load(data_dir + "/sims.pt")

        self.y_data = []
        self.x_data = []
        self.t_data = []
        self.t = []
        self.paras = []
        self.paras_nd = []
        self.indices = []
        self.indices_init = []

        self.scale = scale
        self.p_pred = p_pred
        self.noise = noise
        self.up_layer = nn.Upsample(scale_factor=(4, 1), mode="bilinear")

        ignore_inds = [8, 39]

        # sims = sims[:1] + sims[110:]

        cntr = 0
        for si, sim in enumerate(sims):

            if len(sims_vec) > 0:
                check = (sim[1] == an) & (sim[0] in sims_vec)
            else:
                check = sim[1] == an

            ignr, ignr, raq, fkt, fkp, gr, ar, ignr = sim
            py_dir = data_dir + "/" + sim[1] + "/sim_" + str(sim[0])
            times = torch.load(py_dir + "/times.pt")
            if (
                check and si not in ignore_inds and len(times) > 1
            ):  # and (si==0 or si == 110):
                if not debug:
                    print(
                        tabulate(
                            [
                                ["num", "dataset", "raq", "fkt", "fkp", "gr", "ar"],
                                sim[:-1],
                            ]
                        )
                    )

                raq_nd = torch.tensor(
                    (raq - 0.12624371) / (9.70723344 - 0.12624371), dtype=torch.float64
                )
                fkt_nd = torch.tensor(
                    (np.log10(fkt) - 6.00352841978384)
                    / (9.888820429862925 - 6.00352841978384),
                    dtype=torch.float64,
                )
                fkp_nd = torch.tensor(
                    (np.log10(fkp) - 0.005251646002323797)
                    / (1.9927988938926755 - 0.005251646002323797),
                    dtype=torch.float64,
                )

                fkt = torch.tensor(fkt, dtype=torch.float64)
                fkp = torch.tensor(fkp, dtype=torch.float64)

                self.xc = torch.load(py_dir + "/xc.pt")
                self.yc = torch.load(
                    py_dir + "/yc.pt"
                )  # .reshape(xc.shape[0],xc.shape[1])
                self.xc = self.xc.view(1, self.xc.shape[0], self.xc.shape[1])
                self.yc = self.yc.view(1, self.yc.shape[0], self.yc.shape[1])

                take_every = 1
                self.xc[:, :, 0] = 0
                self.xc[:, :, -1] = 4
                self.yc[:, 0, :] = 0
                self.yc[:, -1, :] = 1

                paras = torch.tensor([raq, fkt, fkp], dtype=torch.float64).view(3, 1, 1)
                paras_nd = torch.tensor(
                    [raq_nd, fkt_nd, fkp_nd], dtype=torch.float64
                ).view(3, 1, 1)

                if debug:
                    reps = max(1, int(roll_forward / 2 * 2))
                    u = torch.load(
                        py_dir + "/e" + str(take_every) + "_uprev_data_select_init.pt"
                    ).repeat(reps, 1, 1, 1)
                    v = torch.load(
                        py_dir + "/e" + str(take_every) + "_vprev_data_select_init.pt"
                    ).repeat(reps, 1, 1, 1)
                    Tprev = torch.load(
                        py_dir + "/e" + str(take_every) + "_Tprev_data_select_init.pt"
                    ).repeat(reps, 1, 1, 1)
                    times = torch.load(py_dir + "/times.pt")[: u.shape[0]]
                    if p_pred:
                        raise ValueError("p_pred is not implemented in debug mode")
                else:
                    u = torch.load(py_dir + "/e" + str(take_every) + "_uprev_data.pt")[
                        :760, ...
                    ]
                    v = torch.load(py_dir + "/e" + str(take_every) + "_vprev_data.pt")[
                        :760, ...
                    ]
                    if p_pred:
                        p = torch.load(
                            py_dir + "/e" + str(take_every) + "_pprev_data.pt"
                        )[:760, ...]
                    Tprev = torch.load(
                        py_dir + "/e" + str(take_every) + "_Tprev_data.pt"
                    )[:760, ...]
                    times = torch.load(py_dir + "/times.pt")[: u.shape[0]]

                for i, t in enumerate(times):
                    if len(sims_vec) > 0:
                        check = (
                            t in np.asarray(times_vec)[np.asarray(sims_vec) == sim[0]]
                        )
                    else:
                        check = True

                    if check and i < len(times) - roll_forward - 1:
                        self.indices.append([cntr, cntr + roll_forward])
                        if i == 0:
                            self.indices_init.append([cntr, cntr + roll_forward])
                    cntr += 1

                    self.paras.append(paras)
                    self.paras_nd.append(paras_nd)

                    u_t = u[i, ...]
                    v_t = v[i, ...]
                    T_t = Tprev[i, ...]

                    if p_pred:
                        p_t = p[i, ...]
                        y = torch.cat((u_t, v_t, p_t), axis=0)
                    else:
                        y = torch.cat((u_t, v_t), axis=0)

                    self.x_data.append(T_t)
                    self.y_data.append(y)
                    self.t_data.append(torch.tensor(t, dtype=torch.float64))
                    self.t.append(t)
                    del u_t, v_t, T_t

                del u, v, Tprev

        self.num_examples = len(self.indices)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx0, idx1 = self.indices[idx]
        if idx0 % 8 == 0:
            idx0, idx1 = random.choice(self.indices_init)

        assert torch.all(self.paras[idx0] == self.paras[idx1])

        scaler = (
            np.exp(
                (self.paras[idx0][0:1, ...] / 10) * 1.80167667
                + np.log(self.paras[idx0][1:2, ...]) * 0.4330392
                + np.log(self.paras[idx0][2:3, ...]) * -0.46052953
            )
            * 5
        )

        t_weight = self.t_data[idx0].double()
        Tp = self.x_data[idx0].double()
        y = self.y_data[idx0].double()
        V = eta_torch(
            self.paras[idx0][1:2, ...], self.paras[idx0][2:3, ...], 1.0 - self.yc, Tp
        )

        Tp1 = self.x_data[idx1].double()
        y1 = self.y_data[idx1].double()

        x = torch.cat(
            (
                self.xc,
                self.yc,
                torch.tensor((self.t[idx1] - self.t[idx0]), dtype=torch.float64).expand(
                    1, Tp.shape[-2], Tp.shape[-1]
                ),
                self.paras_nd[idx0][0:1, ...].expand(1, Tp.shape[-2], Tp.shape[-1]),
                self.paras_nd[idx0][1:2, ...].expand(1, Tp.shape[-2], Tp.shape[-1]),
                self.paras_nd[idx0][2:3, ...].expand(1, Tp.shape[-2], Tp.shape[-1]),
                torch.log10(torch.clip(V, 1e-8, 1.0)) / 8.0,
                Tp,
                y[0:1, ...] / scaler,
                y[1:2, ...] / scaler,
            ),
            axis=0,
        )

        y_new = torch.cat((y1[0:1, ...] / scaler, y1[1:2, ...] / scaler, Tp1), axis=0)

        # x = F.pad(x, (3,3,0,0), mode="replicate")
        # y_new = F.pad(y_new, (3,3,0,0), mode="replicate")
        return x, y_new, scaler, self.paras[idx0], self.yc


def get_indices(data_dir, an, is_init=False, debug=True, roll_forward=1):

    sims = torch.load(data_dir + "/sims.pt")
    ignore_inds = [8, 39]
    sims_vec = []
    times_vec = []
    # if debug:
    #    sims = [sims[0], sims[110]]
    for si, sim in enumerate(sims):
        if an == "train":
            check = sim[1] == "train"
        else:
            check = sim[1] == "cv"
        ignr, ignr, raq, fkt, fkp, gr, ar, ignr = sim
        py_dir = data_dir + "/" + sim[1] + "/sim_" + str(sim[0])
        if check and si not in ignore_inds:  # and (si==0 or si == 110):
            take_every = 1
            if is_init:
                i_vec = torch.load(
                    py_dir + "/e" + str(take_every) + "_i_vec_select_init.pt"
                )
            else:
                if debug:
                    u = torch.load(
                        py_dir + "/e" + str(take_every) + "_uprev_data_select_snaps.pt"
                    )
                    i_vec = np.arange(u.shape[0])
                else:
                    i_vec = torch.load(
                        py_dir + "/e" + str(take_every) + "_i_vec_select.pt"
                    )
            for i_prev in i_vec:
                sims_vec.append(sim[0])
                times_vec.append(i_prev)
    return sims_vec, times_vec


class NewADDataset(Dataset):

    def __init__(
        self,
        data_dir,
        an,
        scale=True,
        load=False,
        is_init=False,
        p_pred=True,
        noise=0.0,
        debug=True,
        sims_vec=[],
        times_vec=[],
        max_examples_percent_per_epoch=100,
    ):

        sims = torch.load(data_dir + "/sims.pt")

        self.y_data = []
        self.x_data = []
        self.t_data = []
        self.paras = []
        self.paras_nd = []
        self.scale = scale
        self.p_pred = p_pred
        self.noise = noise
        self.up_layer = nn.Upsample(scale_factor=(4, 1), mode="bilinear")

        ignore_inds = [8, 39]

        # if debug:
        #     sims = [sims[0], sims[110]]

        for si, sim in enumerate(sims):

            if len(sims_vec) > 0:
                check = (sim[1] == an) & (sim[0] in sims_vec)
            else:
                check = sim[1] == an

            ignr, ignr, raq, fkt, fkp, gr, ar, ignr = sim
            py_dir = data_dir + "/" + sim[1] + "/sim_" + str(sim[0])
            times = torch.load(py_dir + "/times.pt")
            if (
                check and si not in ignore_inds and len(times) > 1
            ):  # and (si==0 or si == 110):
                if not debug:
                    print(
                        tabulate(
                            [
                                ["num", "dataset", "raq", "fkt", "fkp", "gr", "ar"],
                                sim[:-1],
                            ]
                        )
                    )

                raq_nd = torch.tensor(
                    (raq - 0.12624371) / (9.70723344 - 0.12624371), dtype=torch.float64
                )
                fkt_nd = torch.tensor(
                    (np.log10(fkt) - 6.00352841978384)
                    / (9.888820429862925 - 6.00352841978384),
                    dtype=torch.float64,
                )
                fkp_nd = torch.tensor(
                    (np.log10(fkp) - 0.005251646002323797)
                    / (1.9927988938926755 - 0.005251646002323797),
                    dtype=torch.float64,
                )

                fkt = torch.tensor(fkt, dtype=torch.float64)
                fkp = torch.tensor(fkp, dtype=torch.float64)

                self.xc = torch.load(py_dir + "/xc.pt")
                self.yc = torch.load(
                    py_dir + "/yc.pt"
                )  # .reshape(xc.shape[0],xc.shape[1])
                self.xc = self.xc.view(1, self.xc.shape[0], self.xc.shape[1])
                self.yc = self.yc.view(1, self.yc.shape[0], self.yc.shape[1])

                self.xc[:, :, 0] = 0.0
                self.xc[:, :, -1] = 4.0
                self.yc[:, 0, :] = 0.0
                self.yc[:, -1, :] = 1.0

                take_every = 1

                self.sdf = torch.zeros_like(self.yc)
                self.sdf[:, 0, :] = 1.0
                self.sdf[:, -1, :] = 1.0
                self.sdf[:, :, 0] = 1.0
                self.sdf[:, :, -1] = 1.0

                self.sdf2 = torch.ones_like(self.yc)
                self.sdf2[:, 0, :] = 0.0
                self.sdf2[:, -1, :] = 0.0
                self.sdf2[:, :, 0] = 0.0
                self.sdf2[:, :, -1] = 0.0

                paras = torch.tensor([raq, fkt, fkp], dtype=torch.float64).view(3, 1, 1)
                paras_nd = torch.tensor(
                    [raq_nd, fkt_nd, fkp_nd], dtype=torch.float64
                ).view(3, 1, 1)

                if load:
                    u = torch.load(py_dir + "/e" + str(take_every) + "_uprev_data.pt")
                    if scale:
                        u = scale_var(u, raq, fkt, fkp, "uprev")
                    v = torch.load(py_dir + "/e" + str(take_every) + "_vprev_data.pt")
                    if scale:
                        v = scale_var(v, raq, fkt, fkp, "vprev")

                    if p_pred:
                        p = torch.load(
                            py_dir + "/e" + str(take_every) + "_pprev_data.pt"
                        )
                    Tprev = torch.load(
                        py_dir + "/e" + str(take_every) + "_Tprev_data.pt"
                    )

                    times = torch.load(py_dir + "/times.pt")[:-2]

                    if len(times) > 200:
                        rest_steps = np.arange(200, len(times)).tolist()

                    if len(times) > 700:
                        rest_steps = random.choices(
                            rest_steps, k=min(500, rest_steps[-1] - 200)
                        )
                        i_vec = np.arange(1, 200).tolist() + rest_steps
                    else:
                        i_vec = np.arange(1, len(times)).tolist()

                    if is_init:
                        i_vec = i_vec[:5]
                    else:
                        i_vec = i_vec[5:]

                    if debug:
                        i_vec = i_vec[-8:]
                    for i in i_vec:
                        self.paras.append(paras)
                        self.paras_nd.append(paras_nd)
                        if p_pred:
                            y = torch.cat((u[i, ...], v[i, ...], p[i, ...]), axis=0)
                        else:
                            y = torch.cat((u[i, ...], v[i, ...]), axis=0)

                        self.x_data.append(Tprev[i, ...])
                        self.y_data.append(y)
                        self.t_data.append(
                            torch.tensor(6 / (i + 1) ** 0.25, dtype=torch.float64)
                        )

                else:
                    if is_init:
                        u = torch.load(
                            py_dir
                            + "/e"
                            + str(take_every)
                            + "_uprev_data_select_init.pt"
                        )
                        v = torch.load(
                            py_dir
                            + "/e"
                            + str(take_every)
                            + "_vprev_data_select_init.pt"
                        )
                        if p_pred:
                            p = torch.load(
                                py_dir
                                + "/e"
                                + str(take_every)
                                + "_pprev_data_select_init.pt"
                            )
                        Tprev = torch.load(
                            py_dir
                            + "/e"
                            + str(take_every)
                            + "_Tprev_data_select_init.pt"
                        )
                        i_vec = torch.load(
                            py_dir + "/e" + str(take_every) + "_i_vec_select_init.pt"
                        )
                    else:
                        if debug:
                            u = torch.load(
                                py_dir
                                + "/e"
                                + str(take_every)
                                + "_uprev_data_select_snaps.pt"
                            )
                            v = torch.load(
                                py_dir
                                + "/e"
                                + str(take_every)
                                + "_vprev_data_select_snaps.pt"
                            )
                            Tprev = torch.load(
                                py_dir
                                + "/e"
                                + str(take_every)
                                + "_Tprev_data_select_snaps.pt"
                            )
                            i_vec = np.arange(u.shape[0])
                            if p_pred:
                                raise ValueError(
                                    "p_pred is not implemented in debug mode"
                                )
                        else:
                            u = torch.load(
                                py_dir
                                + "/e"
                                + str(take_every)
                                + "_uprev_data_select.pt"
                            )
                            v = torch.load(
                                py_dir
                                + "/e"
                                + str(take_every)
                                + "_vprev_data_select.pt"
                            )
                            if p_pred:
                                p = torch.load(
                                    py_dir
                                    + "/e"
                                    + str(take_every)
                                    + "_pprev_data_select.pt"
                                )
                            Tprev = torch.load(
                                py_dir
                                + "/e"
                                + str(take_every)
                                + "_Tprev_data_select.pt"
                            )
                            i_vec = torch.load(
                                py_dir + "/e" + str(take_every) + "_i_vec_select.pt"
                            )

                    for i, i_prev in enumerate(i_vec):
                        if len(sims_vec) > 0:
                            # print(sim[0], np.asarray(times_vec).shape, np.asarray(sims_vec).shape)
                            check = (
                                i_prev
                                in np.asarray(times_vec)[np.asarray(sims_vec) == sim[0]]
                            )
                        else:
                            check = True
                        if check:
                            self.paras.append(paras)
                            self.paras_nd.append(paras_nd)
                            if p_pred:
                                y = torch.cat((u[i, ...], v[i, ...], p[i, ...]), axis=0)
                            else:
                                y = torch.cat((u[i, ...], v[i, ...]), axis=0)

                            self.x_data.append(Tprev[i, ...])
                            self.y_data.append(y)
                            self.t_data.append(
                                torch.tensor(
                                    6 / (i_prev + 1) ** 0.25, dtype=torch.float64
                                )
                            )
                    del u, v, Tprev

        self.num_examples = min(
            int(len(self.y_data) * max_examples_percent_per_epoch / 100),
            len(self.y_data),
        )
        print("using ", self.num_examples, " out of ", len(self.y_data), " per epoch")

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        t0 = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        t_weight = self.t_data[idx].double()
        Tp = self.x_data[idx].double()
        # Tp2      = self.up_layer(Tp[:,-48:-16,:].view(1,1,32,506)).view(1,128,506)

        if self.noise > 0:
            n = torch.tensor(
                np.random.uniform(
                    low=-1e-5,
                    high=1e-5,  # self.noise, high=self.noise,
                    size=(1, Tp.shape[-2] - 4, Tp.shape[-1] - 4),
                ),
                dtype=torch.float64,
            )
            Tp[:, 2:-2, 2:-2] = torch.clip(Tp[:, 2:-2, 2:-2] + n, 0.0, 1.35)

        y = self.y_data[idx].double()
        V = eta_torch(
            self.paras[idx][1:2, ...], self.paras[idx][2:3, ...], 1.0 - self.yc, Tp
        )
        V = torch.clip(V, 1e-08, 1)

        if self.scale:
            scaler = (
                np.exp(
                    (self.paras[idx][0:1, ...] / 10) * 1.80167667
                    + np.log(self.paras[idx][1:2, ...]) * 0.4330392
                    + np.log(self.paras[idx][2:3, ...]) * -0.46052953
                )
                * 5
            )
            x = torch.cat(
                (
                    self.xc / 4,
                    self.yc / 4,
                    torch.log10(V) / 8,
                    self.paras_nd[idx][0:1, ...].expand(1, Tp.shape[-2], Tp.shape[-1]),
                    self.paras_nd[idx][1:2, ...].expand(1, Tp.shape[-2], Tp.shape[-1]),
                    self.paras_nd[idx][2:3, ...].expand(1, Tp.shape[-2], Tp.shape[-1]),
                    Tp,
                    # Tp2
                ),
                axis=0,
            )
            if self.p_pred:
                y_new = torch.cat(
                    (y[0:1, ...] / scaler, y[1:2, ...] / scaler, y[2:3, ...]), axis=0
                )
            else:
                y_new = torch.cat((y[0:1, ...] / scaler, y[1:2, ...] / scaler), axis=0)

            # x = F.pad(x, (3,3,0,0), mode="replicate")
            # y_new = F.pad(y_new, (3,3,0,0), mode="replicate")
            t1 = time.time()
            # print(t1-t0)
            return x, y_new, t_weight, scaler
