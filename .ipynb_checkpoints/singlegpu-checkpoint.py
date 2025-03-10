import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasetio import *
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import time
torch.backends.cudnn.benchmark = True

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
        debug=False
    ) -> None:
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.cv_data    = cv_data
        self.train_data_init = train_data_init
        self.cv_data_init    = cv_data_init
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.model_uvp = model_uvp.to(gpu_id)
        if model_AD is not None:
            self.model_AD = model_AD.to(gpu_id)
        else:
            self.model_AD = None
        self.p_pred = p_pred
        self.dx_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2).to(gpu_id)
        self.dy_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3).to(gpu_id)
        self.nn_dir = nn_dir
        self.debug = debug

    def get_lr(self, optimizer):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def get_loss(self, gVTp, uvp):
        u,v,p = self.model_uvp(gVTp)
        loss_fn = torch.nn.L1Loss() 
        u_true = uvp[:,0,...]
        v_true = uvp[:,1,...]
        loss_u = loss_fn(u_true, u) 
        loss_v = loss_fn(v_true, v)
        
        if self.p_pred:
            p_true = uvp[:,2,...]
            loss_p = loss_fn(p_true, p)
        else:
            loss_p = torch.tensor(0, dtype=torch.float64)

        if self.model_AD is None:
            loss_T = torch.tensor(0, dtype=torch.float64)

        u = u.view(-1,1,128,512)
        v = v.view(-1,1,128,512)
        u_true = u_true.view(-1,1,128,512)
        v_true = v_true.view(-1,1,128,512)
        
        mass_consv = torch.mean(
                        
                     torch.abs(F.conv2d(u[...,3:-3], self.dx_center_kernel)[...,1:-1,:] + 
                               F.conv2d(v[...,3:-3], self.dy_center_kernel)[...,:,1:-1]
                              ))
        if self.p_pred:
            loss   = (loss_u + loss_v + loss_p)/3.
        else:
            loss   = (loss_u + loss_v)/2.

        return loss, loss_u, loss_v, loss_p, loss_T, mass_consv
        
    def _run_batch(self, gVTp, uvp, is_train):
        if is_train:
            self.optimizer.zero_grad()
            loss, loss_u, loss_v, loss_p, loss_T, mass_consv = self.get_loss(gVTp, uvp)
            loss.backward()
            self.optimizer.step()
        else:
            loss, loss_u, loss_v, loss_p, loss_T, mass_consv = self.get_loss(gVTp, uvp)

        del gVTp, uvp
        return [loss.detach().item(), 
                loss_u.detach().item(),
                loss_v.detach().item(), 
                loss_p.detach().item(), 
                loss_T.detach().item(), 
                mass_consv.detach().item()]
        
    def _run_epoch(self, epoch):

        #self.train_data.sampler.set_epoch(epoch)
        b_sz = len(next(iter(self.train_data))[0])
        if self.train_data_init is not None:
            #self.train_data_init.sampler.set_epoch(epoch)
            b_sz += len(next(iter(self.train_data_init))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for i, data in enumerate(self.train_data):
            if self.train_data_init is not None:
                data_init = next(iter(self.train_data_init))
                gVTp     = copy.deepcopy(torch.cat((data[0], data_init[0]), axis=0))
                uvp      = copy.deepcopy(torch.cat((data[1], data_init[1]), axis=0))
                r      = torch.randperm(gVTp.shape[0])
                gVTp   = gVTp[r, ...]
                uvp    = uvp[r, ...]
            else:
                gVTp     = copy.deepcopy(data[0])
                uvp      = copy.deepcopy(data[1])
            gVTp   = gVTp.to(self.gpu_id)
            uvp    = uvp.to(self.gpu_id)
            losses = self._run_batch(gVTp, uvp, is_train=True)
            self.losses = [losses[j] + self.losses[j] for j in range(6)]
            if i%100==0:
                print([float(self.losses[j]/(i+1)) for j in range(6)])
        self.losses = [float(self.losses[j]/(i+1)) for j in range(6)]
        

        with torch.no_grad():
            #self.cv_data.sampler.set_epoch(epoch)
            b_sz = len(next(iter(self.cv_data))[0])
            if self.cv_data_init is not None:
                #self.cv_data_init.sampler.set_epoch(epoch)
                b_sz += len(next(iter(self.cv_data_init))[0])
            print(f"[GPU{self.gpu_id}] Epoch CV {epoch} | Batchsize: {b_sz} | Steps: {len(self.cv_data)}")
            for i_cv, data in enumerate(self.cv_data):
                if self.cv_data_init is not None:
                    data_init = next(iter(self.cv_data_init))
                    gVTp    = torch.cat((data[0], data_init[0]), axis=0)
                    uvp     = torch.cat((data[1], data_init[1]), axis=0)
                else:
                    gVTp     = data[0]
                    uvp      = data[1]
                gVTp   = gVTp.to(self.gpu_id)
                uvp    = uvp.to(self.gpu_id)
                losses_cv = self._run_batch(gVTp, uvp, is_train=False)
                self.losses_cv = [losses_cv[j] + self.losses_cv[j] for j in range(6)]
        self.losses_cv = [float(self.losses_cv[j]/(i_cv+1)) for j in range(6)]

    def _save_checkpoint(self, epoch):
        ckp = self.model_uvp.state_dict()
        
        if self.debug:
            torch.save(ckp, self.nn_dir + "fluidnet_uvp.pt")
        else:
            torch.save(ckp, self.nn_dir + str(epoch) + "_fluidnet_uvp.pt")
            
            print("-------------------------------------------")
            print(epoch, self.losses, self.losses_cv, self.get_lr(self.optimizer))
            print("-------------------------------------------")

        
        with open(self.nn_dir + "fluidnet_uvpT.txt", "a") as writer:
            writer.write(str(epoch) + "," + str(self.losses[1:]) 
                         + "," + str(self.losses_cv[1:]) + "," + str(self.get_lr(self.optimizer)) + "\n")
        self.scheduler.step()


    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            t0 = time.time()
            self.losses = [0.0]*6
            self.losses_cv = [0.0]*6
            self._run_epoch(epoch)
            t1 = time.time()
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                print(t1-t0)


def load_train_objs(nn_dir, data_dir, levels, c_i, c_h, c_o, act_fn, r_p, loss_type,
                    use_symm, repeats, kernel, milestones, device, sims_vec, times_vec,
                    sims_vec_init, times_vec_init, use_skip=False, p_pred=False,
                    spectral_conv=False, dilation=1, a_bound=10, restart=False, 
                    advect=False, network="fluidnet", scale=True, noise=0.0, debug=False):


    if network == "fluidnet":
        model_uvp = FluidNet(levels, c_i, c_h, c_o, device, act_fn, r_p, loss_type, 
                             use_symm=use_symm, dilation=dilation, a_bound=a_bound,
                             repeats=repeats, use_skip=use_skip, f=kernel, p_pred=p_pred,
                             spectral_conv=spectral_conv).double()
    elif network == "convae":
        model_uvp = ConvAE(levels, c_i, c_h, c_o, device, act_fn, r_p, loss_type, 
                             use_symm=use_symm, dilation=dilation, a_bound=a_bound,
                             repeats=repeats, use_skip=use_skip, f=kernel, p_pred=p_pred,
                             spectral_conv=spectral_conv).double()

    torch.compile(model_uvp)
    print(count_parameters(model_uvp))
    print(model_uvp)
    
    if advect:
        model_AD = ADNet(device=device, CN_max=0.9).double()
        model_AD.eval()
    else:
        model_AD = None

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
    dataset_init = {}   
    for an in ["train", "cv"]:
        if network == "fluidnet" or network == "convae":
            dataset_pytorch = ADDataset
        elif network == "transolver" or network == "transolver_structured":
            dataset_pytorch = UnstructuredDataset

        dataset[an] = dataset_pytorch(data_dir, an, scale, is_init=False, p_pred=p_pred, noise=noise, debug=debug)
        
        dataset_init[an] = dataset_pytorch(data_dir, an, scale, is_init=True, p_pred=p_pred, noise=noise, debug=debug)

    optimizer = torch.optim.Adam([
                    {"params": model_uvp.parameters(),
                     "lr": start_lr,
                     #"weight_decay": 1e-5
                    }
    ])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    
    return dataset, dataset_init, model_uvp, model_AD, optimizer, scheduler


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )


def main(save_every: int, 
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
         device,
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
         scale=True):
    
    dataset, dataset_init, model_uvp, model_AD, optimizer, scheduler = load_train_objs(nn_dir, data_dir, levels, c_i, c_h, c_o, act_fn, 
                                                                                        r_p, loss_type, 
                                                                                        use_symm, repeats, kernel, milestones, device, sims_vec,
                                                                                        times_vec, sims_vec_init, times_vec_init,
                                                                                        use_skip=use_skip, p_pred=p_pred,
                                                                                        spectral_conv=spectral_conv, dilation=dilation, a_bound=a_bound, 
                                                                                        restart=restart,
                                                                                        advect=advect, network=network, 
                                                                                        scale=scale, debug=debug)
    train_data = prepare_dataloader(dataset["train"], batch_size-1)
    cv_data    = prepare_dataloader(dataset["cv"], batch_size-1)
    train_data_init = prepare_dataloader(dataset_init["train"], 1)
    cv_data_init    = prepare_dataloader(dataset_init["cv"], 1)
        
    trainer    = Trainer(model_uvp, model_AD, train_data, cv_data, 
                         train_data_init, cv_data_init, 
                         optimizer, scheduler, device, save_every,
                        nn_dir, p_pred, debug)
    trainer.train(total_epochs)

if __name__ == "__main__":

    data_dir = "/plp_scr1/agar_sh/data/TPH/"
    
    import argparse
    parser = argparse.ArgumentParser(description='Train convnet')
    parser.add_argument("-a", "--act_fn", type=str, help ="activation function", default="gelu")
    parser.add_argument("-l", "--levels", type=int, help ="levels", default=6)
    parser.add_argument("-f", "--c_h", type=int, help ="filters")
    parser.add_argument("-p", "--r_p", type=str, help="padding type", default="replicate")
    parser.add_argument("-gpu", "--gpu_nums", type=str)
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

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_nums
    device = 0

    nn_dir = "/plp_user/agar_sh/PBML/pytorch/TPH/CONVNN/trained_networks/"
    f_nn   =    network + "_levels_" + str(levels) + "_" + act_fn + \
                "_" + str(c_h) + "_" + r_p + "_" + loss_type + "_dil_" + str(dilation) +  \
                "_" + str(use_symm) + "_ab" + str(a_bound) + "_b" + str(batch_size) + \
                "_r" + str(repeats) + "_k" + str(kernel) + \
                "_ad" + str(advect) + "_p_pred" + str(p_pred) + "_deb" + str(debug) 
    
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

    nn_dir = nn_dir + f_nn + "/"
    save_every = 1

    sims_vec = {}
    times_vec = {}
    sims_vec_init = {}
    times_vec_init = {}

    for an in ["train", "cv"]:
        sims_vec[an], times_vec[an] = get_indices(data_dir, an, is_init=False, debug=debug)
        sims_vec_init[an], times_vec_init[an] = get_indices(data_dir, an, is_init=True, debug=debug)
    
    main(save_every, 
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
         device,
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
         scale)