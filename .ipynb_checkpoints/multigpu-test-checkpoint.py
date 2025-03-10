import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from datasetio import *
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import time


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12354"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

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
        self.model_uvp = DDP(model_uvp, device_ids=[gpu_id])
        if model_AD is not None:
            self.model_AD = model_AD.to(gpu_id)
            self.model_AD = DDP(model_AD, device_ids=[gpu_id])
        else:
            self.model_AD = None
        self.p_pred = p_pred
        self.dx_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(2).to(gpu_id)
        self.dy_center_kernel = torch.Tensor([-0.5,0,0.5]).double().unsqueeze(0).unsqueeze(1).unsqueeze(3).to(gpu_id)
        self.losses = [0.0]*6
        self.losses_cv = [0.0]*6
        self.nn_dir = nn_dir
        self.debug = debug

    def get_lr(self, optimizer):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
    def _run_epoch(self, epoch):

        self.train_data.sampler.set_epoch(epoch)
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for i, data in enumerate(self.train_data):
            gVTp     = data[0]
            #if self.gpu_id==1:
            #    print(f"[GPU{self.gpu_id}] data {gVTp} ")
            

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)


def load_train_objs(rank, nn_dir, data_dir, levels, c_i, c_h, c_o, act_fn, r_p, loss_type,
                    use_symm, repeats, kernel, milestones, use_skip=False, p_pred=False,
                    spectral_conv=False, dilation=1, a_bound=10, restart=False, 
                    advect=False, network="fluidnet", scale=True, noise=0.0, debug=False):


    model_uvp = FluidNet(levels, c_i, c_h, c_o, rank, act_fn, r_p, loss_type, 
                         use_symm=use_symm, dilation=dilation, a_bound=a_bound,
                         repeats=repeats, use_skip=use_skip, f=kernel, p_pred=p_pred,
                         spectral_conv=spectral_conv).double()

    print(count_parameters(model_uvp))
    print(model_uvp)
    
    if advect:
        model_AD = ADNet(device=device, CN_max=0.9).double()
        model_AD.eval()
    else:
        model_AD = None

    epoch       = 0
    start_lr    = 1e-3
    best_vloss  = 1e+16

    optimizer = torch.optim.Adam([
                    {"params": model_uvp.parameters(),
                     "lr": start_lr,
                     #"weight_decay": 1e-5
                    }
    ])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    
    return model_uvp, model_AD, optimizer, scheduler


def prepare_dataloader(dataset, rank, world_size, batch_size, pin_memory=True, num_workers=0):
    #sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=False,
        num_workers=num_workers, 
        drop_last=False, 
        #sampler=sampler
    )


def main(rank: int, world_size: int, 
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
         dataset,
         dataset_init,
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
    
    ddp_setup(rank, world_size)
    model_uvp, model_AD, optimizer, scheduler = load_train_objs(rank, nn_dir, data_dir, levels, c_i, c_h, c_o, act_fn, r_p, loss_type, 
                                                                                        use_symm, repeats, kernel, milestones, use_skip=use_skip, p_pred=p_pred,
                                                                                        spectral_conv=spectral_conv, dilation=dilation, a_bound=a_bound, 
                                                                                        restart=restart,
                                                                                        advect=advect, network=network, 
                                                                                        scale=scale, debug=debug)

    train_data = prepare_dataloader(dataset["train"], rank, world_size, batch_size-1, pin_memory=True, num_workers=0)
    cv_data    = prepare_dataloader(dataset["cv"], rank, world_size, batch_size-1, pin_memory=True, num_workers=0)
    if dataset_init["train"] is not None:
        train_data_init = prepare_dataloader(dataset_init["train"], rank, world_size, 1, pin_memory=True, num_workers=0)
        cv_data_init    =prepare_dataloader(dataset_init["cv"], rank, world_size, 1, pin_memory=True, num_workers=0)
    else:
        train_data_init = None
        cv_data_init = None
        
    trainer    = Trainer(model_uvp, model_AD, train_data, cv_data, 
                         train_data_init, cv_data_init, 
                         optimizer, scheduler, rank, save_every,
                        nn_dir, p_pred, debug)
    trainer.train(total_epochs)
    destroy_process_group()


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

    world_size = torch.cuda.device_count()
    nn_dir = "/plp_user/agar_sh/PBML/pytorch/TPH/CONVNN/trained_networks/"
    f_nn   =    network + "_levels_" + str(levels) + "_" + act_fn + \
                "_" + str(c_h) + "_" + r_p + "_" + loss_type + "_dil_" + str(dilation) +  \
                "_" + str(use_symm) + "_ab" + str(a_bound) + "_b" + str(batch_size) + \
                "_r" + str(repeats) + "_k" + str(kernel) + \
                "_ad" + str(advect) + "_p_pred" + str(p_pred) + "_deb" + str(debug) 
    if world_size>1:
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

    dataset = {}    
    dataset_init = {}   
    dataset["train"] = ADDataset(data_dir, "train", scale, is_init=False, p_pred=p_pred, noise=noise, debug=debug)
    dataset["cv"] = ADDataset(data_dir, "cv", scale, is_init=False, p_pred=p_pred, noise=noise, debug=debug)

    chunks = int(torch.floor(dataset["train"].__len__()/world_size))
    print(dataset["train"].__len__(), chunks, dataset["train"][chunks:chunks*2])

    nn_dir = nn_dir + f_nn + "/"
    save_every = 1
    mp.spawn(main, args=(world_size, 
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
                         dataset,
                         dataset_init,
                         use_skip, 
                         p_pred,                                                                         
                         spectral_conv, 
                         dilation, 
                         a_bound, 
                         restart, 
                         advect, 
                         network,
                         debug,
                         scale), 
        nprocs=world_size)