import os
import sys
import time
import math
from collections import OrderedDict

import numpy

import torch

from mpi4py import MPI
comm = MPI.COMM_WORLD


import logging
logger = logging.getLogger()

try:
    import horovod.torch as hvd
except:
    pass

# from larcv.distributed_queue_interface import queue_interface


from .trainercore import trainercore

from src.config import ComputeMode, ModeKind, DistributedMode



class distributed_trainer(trainercore):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args):
        # Rely on the base class for most standard parameters, only
        # search for parameters relevant for distributed computing here

        trainercore.__init__(self, args)

        if self.args.run.distributed_mode == DistributedMode.horovod:
            import horovod.torch as hvd
            hvd.init()
            # if self.args.run.compute_mode == ComputeMode.GPU:
                # os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
            self._rank            = hvd.rank()
            self._local_rank      = hvd.local_rank()
            self._size            = hvd.size()
        else:
            import socket
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            rank = MPI.COMM_WORLD.Get_rank()


            # Pytorch will look for these:
            local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
            size = MPI.COMM_WORLD.Get_size()
            rank = MPI.COMM_WORLD.Get_rank()

            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(size)
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

            self._rank = rank
            self._size = size
            self._local_rank = local_rank

            # It will want the master address too, which we'll broadcast:
            if rank == 0:
                master_addr = socket.gethostname()
            else:
                master_addr = None

            master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(2345)

            # What backend?  nccl on GPU, gloo on CPU
            if self.args.run.compute_mode == ComputeMode.GPU: backend = 'nccl'
            elif self.args.run.compute_mode == ComputeMode.CPU: backend = 'gloo'

            torch.distributed.init_process_group(
                backend=backend, init_method='env://')



        # Put the IO rank as the last rank in the COMM, since rank 0 does tf saves
        root_rank = self._size - 1

        # if self.args.compute_mode == "GPU":
        #     os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

        # if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        #     target_gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        #     os.environ['CUDA_VISIBLE_DEVICES'] = str(target_gpu)

        #     print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])


        # self._larcv_interface = queue_interface()#read_option='read_from_single_local_rank')
        # self._iteration       = 0
        # self._rank            = hvd.rank()
        # self._local_rank      = hvd.local_rank()
        self._global_step     = torch.as_tensor(-1)

        # print('This is rank', self._rank, ', local rank', self._local_rank)

        # if self._rank == 0:
        #     self.args.dump_config()



    # def __del__(self):
    #     if hvd.rank() == 0:
    #         trainercore.__del__(self)

    def save_model(self):

        if self._rank == 0:
            trainercore.save_model(self)


    def init_optimizer(self):


        # This takes the base optimizer (self._opt) and replaces
        # it with a distributed version

        trainercore.init_optimizer(self)

        #
        # if self.args.lr_schedule == '1cycle':
        #     self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self._opt, one_cycle_clr, last_epoch=-1)
        # elif self.args.lr_schedule == 'triangle_clr':
        #     self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self._opt, triangle_clr, last_epoch=-1)
        # elif self.args.lr_schedule == 'exp_range_clr':
        #     self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self._opt, exp_range_clr, last_epoch=-1)
        # elif self.args.lr_schedule == 'decay':
        #     self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self._opt, decay_after_epoch, last_epoch=-1)
        # elif self.args.lr_schedule == 'expincrease':
        #     self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self._opt, exp_increase_lr, last_epoch=-1)
        # else:
        #     self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self._opt, constant_lr, last_epoch=-1)

        self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._opt, constant_lr, last_epoch=-1)

        if self.args.run.distributed_mode == DistributedMode.horovod:
            self._opt = hvd.DistributedOptimizer(self._opt, named_parameters=self._net.named_parameters())





    def init_saver(self):
        if self._rank == 0:
            trainercore.init_saver(self)
        else:
            self._saver = None
            self._aux_saver = None


    def print_network_info(self):
        if self._rank == 0:
            trainer.print_network_info(self)
        return


    def restore_model(self):

        if self._rank == 0:
            state = trainercore.restore_model(self)

            if state is not None:
                self.load_state(state)
            else:
                self._global_step = torch.as_tensor(0)

        if self.args.run.distributed_mode == DistributedMode.horovod:

            # Broadcast the global step:
            self._global_step = hvd.broadcast_object(self._global_step, root_rank = 0)

            # Broadcast the state of the model:
            hvd.broadcast_parameters(self._net.state_dict(), root_rank = 0)

            # Broadcast the optimizer state:
            hvd.broadcast_optimizer_state(self._opt, root_rank = 0)

            # Horovod doesn't actually move the optimizer onto a GPU:
            if self.args.run.compute_mode == ComputeMode.GPU:
                for state in self._opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = state[k].to(self.default_device())



            # Broadcast the LR Schedule state:
            state_dict = hvd.broadcast_object(self._lr_scheduler.state_dict(), root_rank = 0)

        elif self.args.run.distributed_mode == DistributedMode.DDP:

            # print(f"{self._rank}: next(self._net.parameters()).device: { next(self._net.parameters()).device}")
            # print(f"{self._rank}: pre type(self._net): {type(self._net)}")
            self._net.to(self.default_device())

            # print(self._net.parameters)

            self._net = torch.nn.parallel.DistributedDataParallel(
                module = self._net,
                # find_unused_parameters = True,
                # static_graph = True,
            )

            # If using GPUs, move the model to GPU:
            if self.args.run.compute_mode == ComputeMode.GPU:
                for state in self._opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.default_device())

            # print(f"{self._rank}: post type(self._net): {type(self._net)}")
            # print(self._net.parameters)

            self._global_step = MPI.COMM_WORLD.bcast(self._global_step, root=0)
            if self.args.mode.name == ModeKind.train:
                state_dict = MPI.COMM_WORLD.bcast(self._lr_scheduler.state_dict(), root=0)

        # Load the state dict:
        if self.args.mode.name == ModeKind.train:
            self._lr_scheduler.load_state_dict(state_dict)

        return



    def summary(self, metrics, saver=""):
        if self._rank == 0:
            trainercore.summary(self, metrics, saver)
        return

    # def _compute_metrics(self, logits, minibatch_data, loss):
    def _compute_metrics(self, logits, vertex_data, loss, plane_to_losses):
        # This function calls the parent function which computes local metrics.
        # Then, it performs an all reduce on all metrics:
        # metrics = trainercore._compute_metrics(self, logits, minibatch_data, loss)
        metrics = trainercore._compute_metrics(self, logits, vertex_data, loss, plane_to_losses)
        if self.args.run.distributed_mode == DistributedMode.horovod:
            for key in metrics:
                metrics[key] = hvd.allreduce(metrics[key], name = key)
        elif self.args.run.distributed_mode == DistributedMode.DDP:
            for key in metrics:
                torch.distributed.all_reduce(metrics[key])
                metrics[key] /= self._size
        return metrics

    # def on_epoch_end(self):
    #     pass

    # def on_step_end(self):
    #     self._lr_scheduler.step()
    #     # pass


    def default_device_context(self):

        # Convert the input data to torch tensors
        if self.args.run.compute_mode == ComputeMode.GPU:
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                # Then, it's manually set, use it
                return torch.cuda.device(0)
            else:
                return torch.cuda.device(int(self._local_rank))
        else:
            return contextlib.nullcontext
            # device = torch.device('cpu')

    def default_device(self):

        if self.args.run.compute_mode == ComputeMode.GPU:
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                # Then, it's manually set, use it
                return torch.device("cuda:0")
            else:
                return torch.device(f"cuda:{self._local_rank}")
        else:
            device = torch.device('cpu')



    def to_torch(self, minibatch_data):

        # This function wraps the to-torch function but for a gpu forces

        device = self.default_device()

        minibatch_data = trainercore.to_torch(self, minibatch_data, device)

        return minibatch_data

    def log(self, metrics, saver=""):
        if self._rank == 0:
            trainercore.log(self, metrics, saver)




# max_steps = 5000
# base_lr = 0.003
peak_lr = 1.5
cycle_len = 0.8

def constant_lr(step):
    return 1.0

    # def decay_after_epoch(step):
    #     if step > self.args.iterations*cycle_len:
    #         return 0.1
    #     else:
    #         return 1.0

    # def lr_increase(step):

    #     # This function actually ignores the input and uses the global step variable
    #     # This allows it to get the learning rate correct after restore.

    #     # For this problem, the dataset size is 1e5.
    #     # So the epoch can be calculated easily:
    #     # epoch = (step * self.args.MINIBATCH_SIZE) / (1e5)

    #     base_lr   = self.args.learning_rate
    #     step_size = 5.0

    #     return 1.0 + step*step_size

    #     # # return 1.0 + max_lr

    #     # # Perform 500 warmup steps, gradually ramping the rate:
    #     # if epoch <= flat_warmup:
    #     #     return 1.0
    #     # elif epoch < flat_warmup + linear_warmup:
    #     #     return 1.0 + (target - 1) * (epoch - flat_warmup) / linear_warmup
    #     # elif epoch <= flat_warmup + linear_warmup + full:
    #     #     return target
    #     # else:
    #     #     return target * numpy.exp(-0.001*(epoch-(full+linear_warmup+flat_warmup)))


    # def one_cycle_clr(step):

    #     peak = peak_lr / self.args.learning_rate

    #     cycle_steps  = int(self.args.iterations*cycle_len)
    #     end_steps = self.args.iterations - cycle_steps
    #     # Which cycle are we in?

    #     cycle = int(step / cycle_steps)
    #     intra_step = 1.0 * (step % cycle_steps)

    #     base_multiplier = 1.0

    #     if cycle < 1:
    # #         base_multiplier *= 0.5

    #         if intra_step > cycle_steps*0.5:
    #             intra_step = cycle_steps - intra_step

    #         value = intra_step * (peak) /(0.5*cycle_steps)

    #     else:
    #         value = (intra_step / end_steps)*-1.0

    #     print ('using', base_multiplier + value)
    #     return base_multiplier + value

    # min_lr = {}
    # max_lr = {}
    # min_lr['2d'] = 0.0002
    # max_lr['2d'] = 0.0018
    # min_lr['3d'] = 0.0001
    # max_lr['3d'] = 0.0035

    # def triangle_clr(step):
    #     '''
    #     Implements the triangular cycle
    #     learning rate schedule
    #     '''
    #     step_size = 100
    #     cycle = math.floor(1 + step / (2 * step_size))
    #     func = 1 - abs(step / step_size - 2 * cycle + 1)
    #     diff = max_lr[self.args.image_type] - min_lr[self.args.image_type]

    #     return (min_lr[self.args.image_type] + diff * max(0, func)) / self.args.learning_rate

    # def exp_range_clr(step,
    #                   step_size = 100,
    #                   min_lr=min_lr[self.args.image_type],
    #                   max_lr=max_lr[self.args.image_type],
    #                   mode='exp_range',
    #                   gamma=0.999):
    #     '''
    #     Implements the cyclical lr with exp decrease
    #     learning rate schedule
    #     '''
    #     scale_func = 1
    #     if mode == 'exp_range':
    #         scale_func = gamma**step

    #     max_lr *= scale_func

    #     if max_lr <= min_lr:
    #         max_lr = min_lr

    #     step_size = 100
    #     cycle = math.floor(1 + step / (2 * step_size))
    #     func = 1 - abs(step / step_size - 2 * cycle + 1)
    #     diff = max_lr - min_lr

    #     return (min_lr + diff * max(0, func)) / self.args.learning_rate



    # def exp_increase_lr(step):
    #   '''
    #   This function increases the learning rate exponentialy
    #   from start_lr to end_lr. It can be used to study the loss
    #   vs. learning rate and fins a proper interaval in which
    #   to vary the learning rate.
    #   '''

    #   start_lr = self.args.learning_rate  # 1.e-7
    #   end_lr = self.args.learning_rate * 1.e8

    #   return math.exp(step * math.log(end_lr / start_lr) / self.args.iterations)
