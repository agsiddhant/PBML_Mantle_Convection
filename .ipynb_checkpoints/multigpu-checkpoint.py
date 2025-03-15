import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasetio import *
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import datetime
import random

import os
import time


def ddp_setup(rank, world_size, master_port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    print(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    if world_size > 1:
        os.environ["MASTER_PORT"] = str(master_port)  # str(65535)
    else:
        os.environ["MASTER_PORT"] = str(random.randint(1, 65535))
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=36000),
    )


class Trainer:
    def __init__(
        self,
        model_uvp: torch.nn.Module,
        model_AD: torch.nn.Module,
        train_data: DataLoader,
        cv_data: DataLoader,
        train_data_init: DataLoader,
        cv_data_init: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        gpu_id: int,
        save_every: int,
        nn_dir,
        p_pred=False,
        debug=False,
        network="fluidnet",
        loss_scale=False,
        loss_derivative=False,
        roll_forward=1,
        epoch=0,
        loss_type="curl",
    ) -> None:
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.cv_data = cv_data
        self.train_data_init = train_data_init
        self.cv_data_init = cv_data_init
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.model_uvp = model_uvp.to(gpu_id)
        self.model_uvp = DDP(model_uvp, device_ids=[gpu_id])
        if model_AD is not None:
            self.model_AD = model_AD.to(gpu_id)
        else:
            self.model_AD = None
        self.p_pred = p_pred
        self.dx_center_kernel = (
            torch.Tensor([-0.5, 0, 0.5])
            .double()
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(2)
            .to(gpu_id)
        )
        self.dy_center_kernel = (
            torch.Tensor([-0.5, 0, 0.5])
            .double()
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(3)
            .to(gpu_id)
        )
        self.dx_left_kernel = (
            torch.Tensor([-1, 1, 0])
            .double()
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(2)
            .to(gpu_id)
        )
        self.dy_top_kernel = (
            torch.Tensor([-1, 1, 0])
            .double()
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(3)
            .to(gpu_id)
        )

        self.nn_dir = nn_dir
        self.debug = debug
        self.net = network
        self.loss_scale = loss_scale
        self.loss_derivative = loss_derivative
        self.roll_forward = roll_forward
        self.start_epoch = epoch
        self.loss_type = loss_type
        self.l1 = torch.nn.L1Loss()

    def get_lr(self, optimizer):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def loss_fn(self, x_true, x_pred):

        if self.loss_scale:
            maxs = torch.amax(x_true, dim=(1, 2), keepdim=True)
            mins = torch.amin(x_true, dim=(1, 2), keepdim=True)
            scaler = torch.clip(1.0 / (maxs - mins), 1.0, 10.0)
            bc_scaler = torch.ones_like(x_true) + 10.0
            bc_scaler[:, 2:-2, 2:-2] = 1.0
            loss = torch.mean(torch.abs(((x_true - x_pred)) * scaler * bc_scaler))
            return loss, self.l1(x_true, x_pred)
        else:
            loss = self.l1(x_true, x_pred)
            return loss, loss

    def get_loss(self, gVTp, uvp, scaler, paras=None, yc=None):

        if "fluidnet" in self.net:
            u, v, p = self.model_uvp(gVTp)
            u_true = uvp[:, 0, ...]
            v_true = uvp[:, 1, ...]
            loss_u, loss_true_u = self.loss_fn(u_true, u)
            loss_v, loss_true_v = self.loss_fn(v_true, v)

            if self.p_pred:
                p_true = uvp[:, 2, ...]
                loss_p, _ = self.loss_fn(p_true, p)
            else:
                loss_p = torch.tensor(0, dtype=torch.float64)

            u = u.view(-1, 1, 128, 506)
            v = v.view(-1, 1, 128, 506)
            u_true = u_true.view(-1, 1, 128, 506)
            v_true = v_true.view(-1, 1, 128, 506)

            if self.model_AD is None:
                loss_T = torch.tensor(0, dtype=torch.float64)

            du_dx = F.conv2d(u, self.dx_center_kernel)[..., 1:-1, :]
            dv_dy = F.conv2d(v, self.dy_center_kernel)[..., :, 1:-1]

            if self.loss_derivative:
                du_dy_true = F.conv2d(u_true, self.dy_top_kernel) * 126
                du_dy_pred = F.conv2d(u, self.dy_top_kernel) * 126
                dv_dx_true = F.conv2d(v_true, self.dx_left_kernel) * 126
                dv_dx_pred = F.conv2d(v, self.dx_left_kernel) * 126
            
                loss_u += self.l1(du_dy_true, du_dy_pred)
                loss_v += self.l1(dv_dx_true, dv_dx_pred)

            mass_consv = torch.abs(du_dx + dv_dy)

            if self.model_AD is None:
                if self.p_pred:
                    loss = (loss_u + loss_v + loss_p) / 3.0
                else:
                    loss = (loss_u + loss_v) / 2.0
            else:
                if self.p_pred:
                    loss = (loss_u + loss_v + loss_p + loss_T) / 4.0
                else:
                    loss = (loss_u + loss_v + loss_T) / 3.0

            if self.loss_type == "mass":
                loss += torch.mean(mass_consv)
            elif self.loss_type == "curl":
                loss += (
                    torch.mean(mass_consv[:, :, :, 0])
                    + torch.mean(mass_consv[:, :, :, -1])
                    + torch.mean(mass_consv[:, :, 0, :])
                    + torch.mean(mass_consv[:, :, -1, :])
                )

            mass_consv = torch.mean(mass_consv)

        elif self.net == "unet":

            if self.p_pred:
                xc, yc, dt, raq_nd, fkt_nd, fkp_nd, V, T, u, v, p = torch.split(
                    gVTp, 1, dim=1
                )
            else:
                xc, yc, dt, raq_nd, fkt_nd, fkp_nd, V, T, u, v = torch.split(
                    gVTp, 1, dim=1
                )

            for r in range(self.roll_forward):
                with torch.no_grad():
                    for _ in range(self.roll_forward - 1):
                        gVTp = torch.cat(
                            (
                                xc / 4.0,
                                yc / 4.0,
                                dt / self.roll_forward,
                                raq_nd,
                                fkt_nd,
                                fkp_nd,
                                V.view(-1, 1, 128, 506),
                                T.view(-1, 1, 128, 506),
                                u.view(-1, 1, 128, 506),
                                v.view(-1, 1, 128, 506),
                            ),
                            axis=1,
                        )
                        u, v, p, T = self.model_uvp(gVTp)
                        V = eta_torch(
                            paras[:, 1:2, ...],
                            paras[:, 2:3, ...],
                            1.0 - yc,
                            T.view(-1, 1, 128, 506).detach(),
                        )
                        V = torch.log10(torch.clip(V, 1e-8, 1.0)) / 8.0

                gVTp = torch.cat(
                    (
                        xc / 4.0,
                        yc / 4.0,
                        dt / self.roll_forward,
                        raq_nd,
                        fkt_nd,
                        fkp_nd,
                        V.view(-1, 1, 128, 506),
                        T.view(-1, 1, 128, 506),
                        u.view(-1, 1, 128, 506),
                        v.view(-1, 1, 128, 506),
                    ),
                    axis=1,
                )
                u, v, p, T = self.model_uvp(gVTp)
                # V = eta_torch(paras[:,1:2,...], paras[:,2:3,...], 1.-yc, T.view(-1,1,128,506).detach())
                # V = torch.log10(torch.clip(V, 1e-8, 1.0))/8.0

            u_true = uvp[:, 0, ...]
            v_true = uvp[:, 1, ...]

            loss_u, loss_true_u = self.loss_fn(u_true, u)
            loss_v, loss_true_v = self.loss_fn(v_true, v)

            if self.p_pred:
                p_true = uvp[:, 2, ...]
                T_true = uvp[:, 3, ...]
                _, loss_p = self.loss_fn(p_true, p)
                _, loss_T = self.loss_fn(T_true, T)
            else:
                loss_p = torch.tensor(0, dtype=torch.float64)
                T_true = uvp[:, 2, ...]
                _, loss_T = self.loss_fn(T_true, T)

            u = u.view(-1, 1, 128, 506)
            v = v.view(-1, 1, 128, 506)
            u_true = u_true.view(-1, 1, 128, 506)
            v_true = v_true.view(-1, 1, 128, 506)

            du_dx = F.conv2d(u, self.dx_center_kernel)[..., 1:-1, :]
            dv_dy = F.conv2d(v, self.dy_center_kernel)[..., :, 1:-1]

            if self.loss_derivative:
                du_dy_true = F.conv2d(u_true, self.dy_top_kernel) * 126
                du_dy_pred = F.conv2d(u, self.dy_top_kernel) * 126
                dv_dx_true = F.conv2d(v_true, self.dx_left_kernel) * 126
                dv_dx_pred = F.conv2d(v, self.dx_left_kernel) * 126
            
                loss_u += self.l1(du_dy_true, du_dy_pred)
                loss_v += self.l1(dv_dx_true, dv_dx_pred)

            mass_consv = torch.abs(du_dx + dv_dy)

            if self.p_pred:
                loss = (loss_u + loss_v + loss_p + loss_T) / 4.0
            else:
                loss = (loss_u + loss_v + loss_T) / 3.0

            if self.loss_type == "mass":
                loss += torch.mean(mass_consv)
            elif self.loss_type == "curl":
                loss += (
                    torch.mean(mass_consv[:, :, :, 0])
                    + torch.mean(mass_consv[:, :, :, -1])
                    + torch.mean(mass_consv[:, :, 0, :])
                    + torch.mean(mass_consv[:, :, -1, :])
                )

            mass_consv = torch.mean(mass_consv)

        return loss, loss_true_u, loss_true_v, loss_p, loss_T, mass_consv

    def _run_batch(self, gVTp, uvp, scaler, is_train, paras=None, yc=None):

        if is_train:
            self.optimizer.zero_grad()
            if self.net == "convae":
                loss, loss_u, loss_v, loss_p, loss_T, mass_consv = self.get_loss_convae(
                    gVTp, scaler, paras, yc
                )
            else:
                loss, loss_u, loss_v, loss_p, loss_T, mass_consv = self.get_loss(
                    gVTp, uvp, scaler, paras, yc
                )
            loss.backward()
            self.optimizer.step()
        else:
            if self.net == "convae":
                loss, loss_u, loss_v, loss_p, loss_T, mass_consv = self.get_loss_convae(
                    gVTp, scaler, paras, yc
                )
            else:
                loss, loss_u, loss_v, loss_p, loss_T, mass_consv = self.get_loss(
                    gVTp, uvp, scaler, paras, yc
                )

        return [
            loss.detach().item(),
            loss_u.detach().item(),
            loss_v.detach().item(),
            loss_p.detach().item(),
            loss_T.detach().item(),
            mass_consv.detach().item(),
        ]

    def _run_epoch(self, epoch):

        paras = None
        yc = None

        b_sz = len(next(iter(self.train_data))[0])
        if self.train_data_init is not None:
            b_sz += len(next(iter(self.train_data_init))[0])
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        for i, data in enumerate(self.train_data):
            t0 = time.time()
            if self.train_data_init is not None:
                data_init = next(iter(self.train_data_init))
                gVTp = torch.cat((data[0], data_init[0]), axis=0)
                uvp = torch.cat((data[1], data_init[1]), axis=0)
                scaler = torch.cat((data[3], data_init[3]), axis=0)
                r = torch.randperm(gVTp.shape[0])
                gVTp = gVTp[r, ...]
                uvp = uvp[r, ...]
                scaler = uvp[r, ...]
            else:
                gVTp = data[0]
                uvp = data[1]
                scaler = data[3]
                if self.net == "unet":
                    scaler = data[2].to(self.gpu_id)
                    paras = data[3].to(self.gpu_id)
                    yc = data[4].to(self.gpu_id)
            gVTp = gVTp.to(self.gpu_id)
            uvp = uvp.to(self.gpu_id)
            losses = self._run_batch(
                gVTp, uvp, scaler, is_train=True, paras=paras, yc=yc
            )
            self.losses = [losses[j] + self.losses[j] for j in range(6)]
            t1 = time.time()
            if i % 100 == 0:
                print(
                    epoch, [float(self.losses[j] / (i + 1)) for j in range(6)], t1 - t0
                )
        self.losses = [float(self.losses[j] / (i + 1)) for j in range(6)]

        with torch.no_grad():
            b_sz = len(next(iter(self.cv_data))[0])
            if self.cv_data_init is not None:
                b_sz += len(next(iter(self.cv_data_init))[0])
            print(
                f"[GPU{self.gpu_id}] Epoch CV {epoch} | Batchsize: {b_sz} | Steps: {len(self.cv_data)}"
            )
            for i_cv, data in enumerate(self.cv_data):
                if self.cv_data_init is not None:
                    data_init = next(iter(self.cv_data_init))
                    gVTp = torch.cat((data[0], data_init[0]), axis=0)
                    uvp = torch.cat((data[1], data_init[1]), axis=0)
                    scaler = torch.cat((data[3], data_init[3]), axis=0)
                else:
                    gVTp = data[0]
                    uvp = data[1]
                    scaler = data[3]
                    if self.net == "unet":
                        scaler = data[2].to(self.gpu_id)
                        paras = data[3].to(self.gpu_id)
                        yc = data[4].to(self.gpu_id)
                gVTp = gVTp.to(self.gpu_id)
                uvp = uvp.to(self.gpu_id)
                losses_cv = self._run_batch(
                    gVTp, uvp, scaler, is_train=False, paras=paras, yc=yc
                )
                self.losses_cv = [losses_cv[j] + self.losses_cv[j] for j in range(6)]
        self.losses_cv = [float(self.losses_cv[j] / (i_cv + 1)) for j in range(6)]

    def _save_checkpoint(self, epoch):
        ckp = self.model_uvp.module.state_dict()

        if self.debug:
            if epoch % 10 == 0:
                torch.save(ckp, self.nn_dir + "fluidnet_uvp.pt")
        else:
            torch.save(ckp, self.nn_dir + str(epoch) + "_fluidnet_uvp.pt")

            print("-------------------------------------------")
            print(epoch, self.losses, self.losses_cv, self.get_lr(self.optimizer))
            print("-------------------------------------------")

        with open(self.nn_dir + "fluidnet_uvpT.txt", "a") as writer:
            writer.write(
                str(epoch)
                + ","
                + str(self.losses[1:])
                + ","
                + str(self.losses_cv[1:])
                + ","
                + str(self.get_lr(self.optimizer))
                + "\n"
            )
        self.scheduler.step()

    def train(self, max_epochs: int):
        for epoch in range(self.start_epoch, max_epochs):
            t0 = time.time()
            self.losses = [0.0] * 6
            self.losses_cv = [0.0] * 6
            if self.net == "convae":
                self._run_epoch_convae(epoch)
            else:
                self._run_epoch(epoch)
            t1 = time.time()
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                print(t1 - t0)


def load_train_objs(
    rank,
    world_size,
    nn_dir,
    data_dir,
    levels,
    c_i,
    c_h,
    c_o,
    act_fn,
    r_p,
    loss_type,
    use_symm,
    repeats,
    kernel,
    milestones,
    sims_vec,
    times_vec,
    sims_vec_init,
    times_vec_init,
    use_skip=False,
    p_pred=False,
    spectral_conv=False,
    dilation=1,
    a_bound=10,
    restart=False,
    advect=False,
    network="fluidnet",
    scale=True,
    noise=0.0,
    debug=False,
    blurr=False,
    l2_reg=0.0,
    dropout=0.0,
    roll_forward=1,
    factor=2,
    multi_scales=[],
):

    if network == "fluidnet" or network == "ifluidnet":
        model_uvp = FluidNet(
            levels,
            c_i,
            c_h,
            c_o,
            rank,
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
    elif network == "newfluidnet":
        model_uvp = NewFluidNet(
            levels,
            c_i,
            c_h,
            c_o,
            rank,
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
    elif network == "multiscalenewfluidnet":
        models = torch.nn.ModuleList()
        for _ in multi_scales:
            models.append(
                HalfNewFluidNet(
                    levels,
                    c_i,
                    c_h,
                    c_o,
                    rank,
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
            )
        model_uvp = MultiScaleNewFluidNet(
            nets=models,
            loss_type=loss_type,
            device=rank,
            scales=multi_scales,
            p_pred=p_pred,
        ).double()
    elif network == "convae":
        model_uvp = ConvAE(
            levels,
            c_i,
            c_h,
            c_o,
            rank,
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
        ).double()
    elif network == "unet" or network == "iunet":
        model_uvp = Unet(
            levels,
            c_i,
            c_h,
            c_o,
            rank,
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

    torch.compile(model_uvp)
    print(count_parameters(model_uvp))
    print(model_uvp)

    if advect:
        model_AD = ADNet(device=rank, CN_max=2e4).double()
        model_AD.eval()
    else:
        model_AD = None

    if restart:
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

        epoch = int(lines[-1].split(",")[0])
        start_lr = float(lines[-1].split(",")[-1])
        best_vloss = 1e16
        if epoch > milestones[-1]:
            milestones = []
        else:
            start_milestone_ind = np.where(np.asarray(milestones) > epoch)[0][0]
            milestones = [milestones[start_milestone_ind] - epoch] + [
                m - epoch for m in milestones[start_milestone_ind + 1 :]
            ]

        device = torch.cuda.current_device()
        model_uvp.load_state_dict(
            torch.load(
                nn_dir + str(epoch) + "_fluidnet_uvp.pt",
                map_location=torch.device("cuda", device),
            )
        )
        epoch += 1

        print("Restarting from epoch, lr, milestones")
        print(epoch, start_lr, milestones)
    else:
        epoch = 0
        start_lr = 1e-3
        best_vloss = 1e16

        with open(nn_dir + "fluidnet_uvpT.txt", "w") as writer:
            writer.write("Epoch, train loss, val loss, learning rate \n")

    dataset = {}
    dataset_init = {}
    for an in ["train", "cv"]:
        max_examples_percent_per_epoch = 100
        if "fluidnet" in network:
            dataset_pytorch = NewADDataset
            if "multiscale" in network:
                max_examples_percent_per_epoch = 100
        elif network == "convae":
            dataset_pytorch = ConvAEDataset
        elif network == "transolver" or network == "transolver_structured":
            dataset_pytorch = UnstructuredDataset
        elif network == "unet" or network == "iunet":
            dataset_pytorch = ADTimeDataset

        dataset_len = int(
            (len(sims_vec[an]) - len(sims_vec[an]) % world_size) / world_size
        )
        start_ind = int(dataset_len * rank)
        end_ind = int(dataset_len * (rank + 1))
        print(
            rank,
            "splitting ",
            len(sims_vec[an]),
            " samples into ",
            world_size,
            " chunks of size ",
            end_ind - start_ind,
        )

        if network == "unet" or network == "iunet":

            dataset[an] = dataset_pytorch(
                data_dir,
                an,
                scale,
                is_init=False,
                p_pred=p_pred,
                noise=noise,
                debug=debug,
                sims_vec=sims_vec[an][start_ind:end_ind],
                times_vec=times_vec[an][start_ind:end_ind],
                roll_forward=roll_forward,
            )

            dataset_init[an] = None
        else:
            dataset[an] = dataset_pytorch(
                data_dir,
                an,
                scale,
                is_init=False,
                p_pred=p_pred,
                noise=noise,
                debug=debug,
                sims_vec=sims_vec[an][start_ind:end_ind],
                times_vec=times_vec[an][start_ind:end_ind],
                max_examples_percent_per_epoch=max_examples_percent_per_epoch,
            )

            if debug:
                dataset_init[an] = None
            else:
                dataset_len = int(
                    (len(sims_vec_init[an]) - len(sims_vec_init[an]) % world_size)
                    / world_size
                )
                start_ind = int(dataset_len * rank)
                end_ind = int(dataset_len * (rank + 1))
                dataset_init[an] = dataset_pytorch(
                    data_dir,
                    an,
                    scale,
                    is_init=True,
                    p_pred=p_pred,
                    noise=noise,
                    debug=debug,
                    sims_vec=sims_vec_init[an][start_ind:end_ind],
                    times_vec=times_vec_init[an][start_ind:end_ind],
                    max_examples_percent_per_epoch=max_examples_percent_per_epoch,
                )

    optimizer = torch.optim.Adam(
        [{"params": model_uvp.parameters(), "lr": start_lr, "weight_decay": l2_reg}]
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5
    )

    return dataset, dataset_init, model_uvp, model_AD, optimizer, scheduler, epoch


def prepare_dataloader(dataset: Dataset, batch_size: int, world_size, rank):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        # sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    )


def main(
    rank: int,
    world_size: int,
    save_every: int,
    total_epochs: int,
    batch_size: int,
    nn_dir,
    data_dir,
    levels,
    c_i,
    c_h,
    c_o,
    act_fn,
    r_p,
    loss_type,
    use_symm,
    repeats,
    kernel,
    milestones,
    sims_vec,
    times_vec,
    sims_vec_init,
    times_vec_init,
    use_skip=False,
    p_pred=False,
    spectral_conv=False,
    dilation=1,
    a_bound=10,
    restart=False,
    advect=False,
    network="fluidnet",
    debug=False,
    scale=True,
    blurr=False,
    master_port=366,
    l2_reg=0.0,
    dropout=0.0,
    loss_scale=False,
    loss_derivative=False,
    roll_forward=1,
    factor=2,
    multi_scales=[],
):

    ddp_setup(rank, world_size, master_port)
    dataset, dataset_init, model_uvp, model_AD, optimizer, scheduler, epoch = (
        load_train_objs(
            rank,
            world_size,
            nn_dir,
            data_dir,
            levels,
            c_i,
            c_h,
            c_o,
            act_fn,
            r_p,
            loss_type,
            use_symm,
            repeats,
            kernel,
            milestones,
            sims_vec,
            times_vec,
            sims_vec_init,
            times_vec_init,
            use_skip=use_skip,
            p_pred=p_pred,
            spectral_conv=spectral_conv,
            dilation=dilation,
            a_bound=a_bound,
            restart=restart,
            advect=advect,
            network=network,
            scale=scale,
            debug=debug,
            blurr=blurr,
            l2_reg=l2_reg,
            dropout=dropout,
            roll_forward=roll_forward,
            factor=factor,
            multi_scales=multi_scales,
        )
    )
    small_batch = 2
    if world_size > 1:
        small_batch = 1
    train_data = prepare_dataloader(
        dataset["train"], batch_size - small_batch, world_size, rank
    )
    cv_data = prepare_dataloader(
        dataset["cv"], batch_size - small_batch, world_size, rank
    )
    if dataset_init["train"] is not None:
        train_data_init = prepare_dataloader(
            dataset_init["train"], small_batch, world_size, rank
        )
        cv_data_init = prepare_dataloader(
            dataset_init["cv"], small_batch, world_size, rank
        )
    else:
        train_data_init = None
        cv_data_init = None

    trainer = Trainer(
        model_uvp,
        model_AD,
        train_data,
        cv_data,
        train_data_init,
        cv_data_init,
        optimizer,
        scheduler,
        rank,
        save_every,
        nn_dir,
        p_pred,
        debug,
        network,
        loss_scale,
        loss_derivative,
        roll_forward,
        epoch=epoch,
        loss_type=loss_type,
    )
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":

    data_dir = "/plp_scr1/agar_sh/data/TPH/"

    import argparse

    parser = argparse.ArgumentParser(description="Train convnet")
    parser.add_argument(
        "-a", "--act_fn", type=str, help="activation function", default="gelu"
    )
    parser.add_argument("-l", "--levels", type=int, help="levels", default=6)
    parser.add_argument("-f", "--c_h", type=int, help="filters")
    parser.add_argument("-fac", "--factor", type=int, help="factor", default=2)
    parser.add_argument(
        "-p", "--r_p", type=str, help="padding type", default="replicate"
    )
    parser.add_argument("-gpu", "--gpu_nums", type=str)
    parser.add_argument(
        "-lt", "--loss_type", type=str, help="loss type", default="curl"
    )
    parser.add_argument("-d", "--dilation", type=int, help="loss type", default=1)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size")
    parser.add_argument("-s", "--use_symm", type=int, help="use symmetries")
    parser.add_argument("-ab", "--a_bound", type=int, help="bound of a")
    parser.add_argument("-r", "--repeats", type=int, help="repeats in hidden layers")
    parser.add_argument("-rst", "--restart", type=int, help="restart", default=0)
    parser.add_argument(
        "-sk", "--use_skip", type=int, help="use skip connections", default=0
    )
    parser.add_argument("-k", "--kernel", type=int, help="kernel size")
    parser.add_argument("-sc", "--scale", type=int, help="use scaling", default=1)
    parser.add_argument("-l_sc", "--loss_scale", type=int, help="scale loss", default=1)
    parser.add_argument(
        "-l_de", "--loss_derivative", type=int, help="scale loss", default=0
    )
    parser.add_argument("-blurr", "--blurr", type=int, help="blurr a", default=0)
    parser.add_argument("-pp", "--p_pred", type=int, help="predict pressure", default=0)
    parser.add_argument(
        "-ad", "--advect", type=int, help="add advection to loss", default=0
    )
    parser.add_argument("-n", "--noise", type=float, help="noise level", default=0.0)
    parser.add_argument("-deb", "--debug", type=int, help="debugging")
    parser.add_argument(
        "-net", "--network", type=str, help="neural network model", default="fluidnet"
    )
    parser.add_argument(
        "-spectral", "--spectral_conv", type=int, help="use spectral conv", default=0
    )
    parser.add_argument(
        "-mp", "--master_port", type=int, help="master_port", default=366
    )
    parser.add_argument("-l2", "--l2_reg", type=float, help="weight decay", default=0.0)
    parser.add_argument(
        "-d_r", "--drop_rate", type=float, help="dropout rate", default=0.0
    )
    parser.add_argument(
        "-roll", "--roll_forward", type=int, help="roll forward", default=1
    )
    parser.add_argument(
        "-scales", "--multi_scales", type=float, nargs="+", help="scales", default=[]
    )
    args = parser.parse_args()
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
    loss_derivative = True if args.loss_derivative == 1 else False
    blurr = True if args.blurr == 1 else False
    repeats = args.repeats
    kernel = args.kernel
    noise = args.noise
    network = args.network
    master_port = args.master_port
    l2_reg = args.l2_reg
    drop_rate = args.drop_rate
    roll_forward = args.roll_forward
    factor = args.factor
    multi_scales = args.multi_scales

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_nums
    if network == "fluidnet":
        torch.backends.cudnn.benchmark = True

    world_size = torch.cuda.device_count()
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

    if "unet" in network:
        f_nn += "_roll" + str(roll_forward) + "_new"

    if blurr:
        f_nn += "_blurr"

    nn_dir = nn_dir + f_nn + "/"
    if not os.path.isdir(nn_dir):
        os.mkdir(nn_dir)

    if debug:
        epochs = 1500
        milestones = [20, 200, 400, 600, 800, 1000]
        if network == "ifluidnet":
            epochs = 80
            milestones = [4, 14, 24, 34, 50]
    else:
        epochs = 150
        milestones = [20, 40, 60, 80, 180, 120]
        if network == "ifluidnet":
            epochs = 40
            milestones = [2, 7, 12, 17, 25]

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

    save_every = 1

    sims_vec = {}
    times_vec = {}
    sims_vec_init = {}
    times_vec_init = {}

    for an in ["train", "cv"]:
        init_dataset = get_indices_time if network == "unet" else get_indices
        sims_vec[an], times_vec[an] = init_dataset(
            data_dir, an, is_init=False, debug=debug, roll_forward=roll_forward
        )

        if debug:
            sims_vec_init[an], times_vec_init[an] = None, None
        else:
            sims_vec_init[an], times_vec_init[an] = init_dataset(
                data_dir, an, is_init=True, debug=debug, roll_forward=roll_forward
            )

    mp.spawn(
        main,
        args=(
            world_size,
            save_every,
            epochs,
            batch_size,
            nn_dir,
            data_dir,
            levels,
            c_i,
            c_h,
            c_o,
            act_fn,
            r_p,
            loss_type,
            use_symm,
            repeats,
            kernel,
            milestones,
            sims_vec,
            times_vec,
            sims_vec_init,
            times_vec_init,
            use_skip,
            p_pred,
            spectral_conv,
            dilation,
            a_bound,
            restart,
            advect,
            network,
            debug,
            scale,
            blurr,
            master_port,
            l2_reg,
            drop_rate,
            loss_scale,
            loss_derivative,
            roll_forward,
            factor,
            multi_scales,
        ),
        nprocs=world_size,
    )
