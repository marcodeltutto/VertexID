import os
import tempfile
import sys
import time, datetime

import pathlib

import numpy

import torch
from torch.autograd import Variable
# import torchvision

from larcv import queueloader
from . import larcv_fetcher

import datetime

# This uses tensorboardX to save summaries and metrics to tensorboard compatible files.
# import tensorboardX
from torch.utils.tensorboard import SummaryWriter

import logging
logger = logging.getLogger()


from src.config import ModeKind, ComputeMode, ImageModeKind

class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args):
        self.args = args


        if self.is_training():
            self.mode   = 'train'
            access_mode = "random_blocks"
        elif self.args.mode.name == ModeKind.iotest:
            self.mode   = 'iotest'
            access_mode = "serial_access"
        elif self.args.mode.name == ModeKind.inference:
            self.mode   = "inference"
            access_mode = "serial_access"

        logger.info(f"Access mode: {access_mode}")

        self.larcv_fetcher = larcv_fetcher.larcv_fetcher(
            config            = self.args,
            mode              = self.mode,
            distributed       = self.args.run.distributed,
            access_mode       = access_mode,
            dimension         = self.args.data.input_dimension,
            data_format       = self.args.data.image_mode,
            downsample_images = self.args.data.downsample,
        )

        self.args.data.image_width = int(self.args.data.image_width / 2**self.args.data.downsample)
        self.args.data.image_height = int(self.args.data.image_height / 2**self.args.data.downsample)

        self._global_step     = -1
        self._rank            = 0


    def _initialize_io(self, color=0):



        f     = pathlib.Path(self.args.data.data_directory + self.args.data.file)
        aux_f = pathlib.Path(self.args.data.data_directory + self.args.data.aux_file)

        # Check that the training file exists:
        if not f.exists():
            raise Exception(f"Can not continue with file {f} - does not exist.")
        if not aux_f.exists():
            if self.is_training():
                logger.warning("WARNING: Aux file does not exist.  Setting to None for training")
                # self.args.data.aux_file = None
            else:
                # In inference mode, we are creating the aux file.  So we need to check
                # that the directory exists.  Otherwise, no writing.
                if not aux_f.parent.exists():
                    logger.warning("WARNING: Aux file's directory does not exist.")
                    # self.args.data.aux_file = None
                elif self.args.data.aux_file is None or str(self.args.data.aux_file).lower() == "none":
                    logger.warning("WARNING: no aux file set, so not writing inference results.")
                    # self.args.data.aux_file = None

        configured_keys = []
        configured_keys += ["primary",]

        self._train_data_size = self.larcv_fetcher.prepare_sample(
            name            = "primary",
            input_file      = str(f),
            batch_size      = self.args.run.minibatch_size,
            color           = color,
            print_config    = False # True if self._rank == 0 else False
        )
        self._val_data_size = 0
        if aux_f.exists():
            if self.is_training():
                # Fetching data for on the fly testing:
                self._val_data_size = self.larcv_fetcher.prepare_sample(
                    name            = "val",
                    input_file      = str(aux_f),
                    batch_size      = self.args.run.minibatch_size,
                    color           = color,
                    print_config    = False # True if self._rank == 0 else False
                )
                configured_keys += ["val",]

                # self._aux_data_size = self.larcv_fetcher.prepare_writer(
                #     input_file = f, output_file = str(aux_f))
        else:
            self._val_data_size = 0


        return configured_keys

    def is_training(self):

        return self.args.mode.name == ModeKind.train

    def init_network(self):

        # This sets up the necessary input shape:
        input_shape = self.larcv_fetcher.input_shape('primary')

        # Override the shape as we are going to use dense images
        input_shape = [input_shape[0], self.args.data.image_width, self.args.data.image_height]

        logger.info(f"Input shape: {input_shape}")

        # To initialize the network, we see what the name is
        # and act on that:
        # if self.args.network == "yolo":
        if self.args.data.image_mode == ImageModeKind.dense:
            from src.networks import yolo
            self._net = yolo.YOLO(input_shape, self.args.network)
        elif self.args.data.image_mode == ImageModeKind.sparse:
            from src.networks import sparse_yolo
            self._net = sparse_yolo.YOLO(input_shape, self.args.network)
        # else:
        #     raise Exception(f"Couldn't identify network {self.args.network.name}")

        if self.is_training():
            self._net.train(True)

        self._net.to(self.default_device())


    def initialize(self, io_only=False):

        self._initialize_io()

        if io_only:
            return

        with self.default_device_context():

            self.init_network()

            self.score_cut = torch.tensor(0.5, device=self.default_device())

            n_trainable_parameters = 0
            for var in self._net.named_parameters():
                n_trainable_parameters += numpy.prod(var[1].shape)
                # logger.info(f"  var: {var[0]} with shape {var[1].shape} and {numpy.prod(var[1].shape)} parameters.")
            logger.info("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))

            if self.is_training():
                self.init_optimizer()

            self.init_saver()

            state = self.restore_model()

            if not self.args.run.distributed:
                if state is not None:
                    self.load_state(state)
                else:
                    self._global_step = 0

            # if self.args.run.compute_mode == "CPU":
            #     pass
            # if self.args.run.compute_mode == ComputeMode.GPU:
            #     self._net.cuda()

            # if self.args.label_mode == 'all':
            # elif self.args.label_mode == 'split':
            #     self._log_keys = ['loss']
            #     for key in self.args.keyword_label:
            #         self._log_keys.append('acc/{}'.format(key))
            self._log_keys = ['loss', 'images_per_second']



            if self.args.mode.name == ModeKind.inference:
                self.inference_metrics = {}
                self.inference_metrics['n'] = 0

            self._lambda_noobj = 0.5
            self._lambda_coord = 5
            self._criterion_mse = torch.nn.MSELoss()
            self._criterion_ce = torch.nn.CrossEntropyLoss()

    def init_optimizer(self):

        from src.config import OptimizerKind

        # Create an optimizer:
        if self.args.mode.optimizer.name == OptimizerKind.SGD:
            self._opt = torch.optim.SGD(self._net.parameters(), lr=self.args.mode.optimizer.learning_rate,
                weight_decay=self.args.mode.optimizer.weight_decay)
        elif self.args.mode.optimizer.name == OptimizerKind.Adam:
            self._opt = torch.optim.Adam(self._net.parameters(), lr=self.args.mode.optimizer.learning_rate,
                weight_decay=self.args.mode.optimizer.weight_decay)


        def constant_lr(step):
            return 1.0

        self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._opt, constant_lr, last_epoch=-1)






    def init_saver(self):

        # Make sure something is initialized for inference mode
        self._saver = None

        save_dir = self.args.output_dir

        # This sets up the summary saver:
        if self.is_training():
            self._saver = SummaryWriter(save_dir)


        if self._val_data_size != 0 and self.is_training():
            self._aux_saver = SummaryWriter(save_dir + "/test/")
        elif self._val_data_size != 0 and not self.is_training():
            self._aux_saver = SummaryWriter(save_dir + "/val/")
        else:
            self._aux_saver = None


    def restore_model(self):
        ''' This function attempts to restore the model from file
        '''


        def check_inference_weights_path(file_path):

            # Look for the "checkpoint" file:
            checkpoint_file_path = file_path + "checkpoint"
            # If it exists, open it and read the latest checkpoint:
            if os.path.isfile(checkpoint_file_path):
                return checkpoint_file_path


        # First, check if the weights path is set:
        if self.args.mode.weights_location != "":
            checkpoint_file_path = check_inference_weights_path(self.args.mode.weights_location)
        else:
            _, checkpoint_file_path = self.get_model_filepath()

        if not os.path.isfile(checkpoint_file_path):
            logger.info("No previously saved model found.")
            return None
        # Parse the checkpoint file and use that to get the latest file path

        with open(checkpoint_file_path, 'r') as _ckp:
            for line in _ckp.readlines():
                if line.startswith("latest: "):
                    chkp_file = line.replace("latest: ", "").rstrip('\n')
                    chkp_file = os.path.dirname(checkpoint_file_path) + "/" + chkp_file
                    logger.info(f"Restoring weights from {chkp_file}")
                    break


        if self.args.run.compute_mode == ComputeMode.CPU:
            state = torch.load(chkp_file, map_location='cpu')
        else:
            state = torch.load(chkp_file)

        # I don't understand why torch is doing this.  But rank 0, in DDP,
        # Will add 'module.' to the beginning of every parameter...
        new_state_dict = {}
        for key in state['state_dict']:
            if key.startswith("module."):
                new_key = key.replace("module.", "")
                # print(f"replacing {key} with {new_key}")
            else:
                new_key = key
            new_state_dict[new_key] = state['state_dict'][key]

        state['state_dict'] = new_state_dict

        return state


    def load_state(self, state):

        self._net.load_state_dict(state['state_dict'])
        self._global_step = state['global_step']
        if self.is_training():
            self._opt.load_state_dict(state['optimizer'])
            self._lr_scheduler.load_state_dict(state['scheduler'])

            # If using GPUs, move the model to GPU:
            if self.args.run.compute_mode == ComputeMode.GPU:
                for state in self._opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        return True


    def save_model(self):
        '''Save the model to file

        '''

        current_file_path, checkpoint_file_path = self.get_model_filepath()

        # save the model state into the file path:
        state_dict = {
            'global_step' : self._global_step,
            'state_dict'  : self._net.state_dict(),
            'optimizer'   : self._opt.state_dict(),
            'scheduler'   : self._lr_scheduler.state_dict(),
        }

        # Make sure the path actually exists:
        if not os.path.isdir(os.path.dirname(current_file_path)):
            os.makedirs(os.path.dirname(current_file_path))

        torch.save(state_dict, current_file_path)

        # Parse the checkpoint file to see what the last checkpoints were:

        # Keep only the last 5 checkpoints
        n_keep = 100

        past_checkpoint_files = {}
        try:
            with open(checkpoint_file_path, 'r') as _chkpt:
                for line in _chkpt.readlines():
                    line = line.rstrip('\n')
                    vals = line.split(":")
                    if vals[0] != 'latest':
                        past_checkpoint_files.update({int(vals[0]) : vals[1].replace(' ', '')})
        except:
            pass

        # Remove the oldest checkpoints while the number is greater than n_keep
        while len(past_checkpoint_files) >= n_keep:
            min_index = min(past_checkpoint_files.keys())
            file_to_remove = os.path.dirname(checkpoint_file_path) + "/" + past_checkpoint_files[min_index]
            os.remove(file_to_remove)
            past_checkpoint_files.pop(min_index)

        # Update the checkpoint file
        with open(checkpoint_file_path, 'w') as _chkpt:
            _chkpt.write('latest: {}\n'.format(os.path.basename(current_file_path)))
            _chkpt.write('{}: {}\n'.format(self._global_step, os.path.basename(current_file_path)))
            for key in past_checkpoint_files:
                _chkpt.write('{}: {}\n'.format(key, past_checkpoint_files[key]))


    def get_model_filepath(self):
        '''Helper function to build the filepath of a model for saving and restoring:
        '''

        # Find the base path of the log directory
        file_path= self.args.output_dir  + "/checkpoints/"


        name = file_path + 'model-{}.ckpt'.format(self._global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path



    def _3d_to_2d(self, point, plane, pitch=0.4):
        '''
        Returns the y coordinate of a 3D point projected to 2D
        '''

        z_min = 0 # cm
        z_max = 500 # cm
        y_min = -100 # cm
        y_max = 100 # cm

        dim_x = 1536 / 2**self.args.data.downsample
        dim_y = 1024 / 2**self.args.data.downsample

        plane_to_theta = {0: 35.7, 1: -35.7, 2: 0}
        theta = plane_to_theta[plane]

        cos_theta = numpy.cos(numpy.deg2rad(theta))
        sin_theta = numpy.sin(numpy.deg2rad(theta))

        projected = cos_theta * point[2] + sin_theta * point[1]

        if theta > 0:
            minimum = cos_theta * z_min + sin_theta * y_min
            maximum = cos_theta * z_max + sin_theta * y_max
        else:
            minimum = cos_theta * z_min + sin_theta * y_max
            maximum = cos_theta * z_max + sin_theta * y_min

        padding = dim_x - (maximum - minimum) / pitch
        offset = abs(minimum / pitch)
        # print('min', minimum, 'max', maximum, 'offset', offset, 'padding', padding)

        return projected / pitch, padding, offset


    def _target_to_yolo(self, target, logits):

        if self.args.name == 'SBND':
            return self._target_to_yolo_sbnd(target, logits)
        else:
            return self._target_to_yolo_dune(target, logits)


    def _target_to_yolo_dune(self, target, logits):
        '''
        Takes the vertex data from larcv and transforms it
        to YOLO output.

        arguments:
        - target: the data from larcv
        - n_channels: the number of channels used during training
        - grid_size_w: the image width at the end of the network
        - grid_size_h: the image height at the end of the network

        returns:
        - the transformed target
        - a mask that can mask the entries where there are real objects
        '''
        # print('True vertex position:', target)

        with self.default_device_context():
            target_out = []
            mask = []

            pitch = self.args.data.pitch * 2**self.args.data.downsample
            padding_x = self.args.data.padding_x / 2**self.args.data.downsample
            padding_y = self.args.data.padding_y / 2**self.args.data.downsample

            batch_size = target.size(0)

            for plane in range(len(logits)):

                logits_p = logits[plane]

                n_channels = logits_p.size(3)
                grid_size_w = logits_p.size(1)
                grid_size_h = logits_p.size(2)

                # print('plane', plane, 'grid_size_w', grid_size_w, 'grid_size_h', grid_size_h)
                # print('logits shape', logits_p.shape)

                target_out_p = torch.zeros(batch_size, grid_size_w, grid_size_h, n_channels, device=self.default_device())
                mask_p = torch.zeros(batch_size, grid_size_w, grid_size_h, dtype=torch.bool, device=self.default_device())

                step_w = self.args.data.image_width / grid_size_w
                step_h = self.args.data.image_height / grid_size_h

                # print('vertex x', target[0, 2], 'y', target[0, 0])
                # print('vertex x', (target[0, 2]/pitch + padding_x/2), 'y', (target[0, 0]/pitch + padding_y/2))
                # print('step_w', step_w, 'step_h', step_h)


                for batch_id in range(batch_size):

                    projected_x, padding, offset = self._3d_to_2d(target[batch_id, :], plane, pitch)
                    projected_y = target[batch_id, 0] / pitch # common to all planes

                    t_x = (projected_x + offset + padding/2) / step_w
                    t_i = int(t_x)
                    t_y = (projected_y + padding_y/2) / step_h
                    t_j = int(t_y)

                    # t_x = (target[batch_id, 2]/pitch + padding_x/2) / step_w
                    # t_i = int(t_x)
                    # t_y = (target[batch_id, 0]/pitch + padding_y/2) / step_h
                    # t_j = int(t_y)

                    target_out_p[batch_id, t_i, t_j, 0] = t_x - t_i
                    target_out_p[batch_id, t_i, t_j, 1] = t_y - t_j
                    target_out_p[batch_id, t_i, t_j, 2] = 1.
                    target_out_p[batch_id, t_i, t_j, 3] = 1.

                    mask_p[batch_id, t_i, t_j] = 1

                    # print('Batch', batch_id, 't_i', t_i, 't_j', t_j)

                # print('target_out_p', target_out_p)
                # if self._global_step % 25 == 0:
                #     if not self.args.run.distributed or self._rank == 0:
                #         numpy.save(f'yolo_tgt_{plane}', target_out_p.cpu())

                target_out.append(target_out_p)
                mask.append(mask_p)

        return target_out, mask


    def _target_to_yolo_sbnd(self, target, logits):
        '''
        Takes the vertex data from larcv and transforms it
        to YOLO output.

        arguments:
        - target: the data from larcv
        - n_channels: the number of channels used during training
        - grid_size_w: the image width at the end of the network
        - grid_size_h: the image height at the end of the network

        returns:
        - the transformed target
        - a mask that can mask the entries where there are real objects
        '''
        # print('True vertex position:', target)

        with self.default_device_context():
            target_out = []
            mask = []

            batch_size = target.size(0)

            for plane in range(len(logits)):

                logits_p = logits[plane]

                n_channels = logits_p.size(3)
                grid_size_w = logits_p.size(1)
                grid_size_h = logits_p.size(2)

                target_out_p = torch.zeros(batch_size, grid_size_w, grid_size_h, n_channels, device=self.default_device())
                mask_p = torch.zeros(batch_size, grid_size_w, grid_size_h, dtype=torch.bool, device=self.default_device())

                step_w = self.args.data.image_width / grid_size_w
                step_h = self.args.data.image_height / grid_size_h

                for batch_id in range(batch_size):

                    projected_x = target[batch_id, plane, 0]
                    projected_y = target[batch_id, plane, 1]

                    t_x = projected_x / step_w
                    t_i = int(t_x)
                    t_y = projected_y / step_h
                    t_j = int(t_y)

                    if t_j > 39: t_j = 39
                    if t_j < 0: t_j = 0

                    target_out_p[batch_id, t_i, t_j, 0] = t_x - t_i
                    target_out_p[batch_id, t_i, t_j, 1] = t_y - t_j
                    target_out_p[batch_id, t_i, t_j, 2] = 1.
                    target_out_p[batch_id, t_i, t_j, 3] = 1.

                    mask_p[batch_id, t_i, t_j] = 1

                    # print('Batch', batch_id, 't_i', t_i, 't_j', t_j)

                # print('target_out_p', target_out_p)
                if self._global_step == 0:
                    if not self.args.run.distributed or self._rank == 0:
                        numpy.save(f'yolo_tgt_{plane}', target_out_p.cpu())

                target_out.append(target_out_p)
                mask.append(mask_p)

        return target_out, mask


    def _calculate_loss(self, target, mask, prediction, full=False):
        '''
        Calculates the loss.

        arguments:
        - target: the vertex data
        - mask: a mask indicating where the true vertices are
        - prediction: the prediction from the network
        - full: wheater to return all losses or only the total one

        returns:
        - a single scalar for the optimizer to use if full=False
        - all losses if full=True
        '''
        loss     = torch.tensor(0.0, requires_grad=True, device=self.default_device())
        # loss_x   = torch.tensor(0.0, requires_grad=True, device=self.default_device())
        # loss_y   = torch.tensor(0.0, requires_grad=True, device=self.default_device())
        # loss_obj = torch.tensor(0.0, requires_grad=True, device=self.default_device())
        # loss_cls = torch.tensor(0.0, requires_grad=True, device=self.default_device())

        plane_to_losses = {}

        for i, (t, m, p) in enumerate(zip(target, mask, prediction)):
            # if i != 2: continue
            l, x, y, o, c = self._calculate_loss_per_plane(t, m, p)
            loss = loss + l

            plane_to_losses[i] = [l, x, y, o, c]
            # loss_x = loss + x
            # loss_y = loss + y
            # loss_obj = loss + obj
            # loss_cls = loss + c

        if full:
            return loss, plane_to_losses

        return loss



    def _calculate_loss_per_plane(self, target, mask, prediction):
        '''
        Calculates the loss.

        arguments:
        - target: the vertex data
        - mask: a mask indicating where the true vertices are
        - prediction: the prediction from the network

        returns:
        - all losses
        '''

        prediction = prediction.view(prediction.size(0), prediction.size(1)*prediction.size(2), prediction.size(3))
        target     = target.view(target.size(0), target.size(1)*target.size(2), target.size(3))
        mask       = mask.view(target.size(0), -1)

        t_x   = target[:,:,0]
        t_y   = target[:,:,1]
        t_obj = target[:,:,2]
        t_cls = target[:,:,3:]

        p_x   = prediction[:,:,0]
        p_y   = prediction[:,:,1]
        p_obj = prediction[:,:,2]
        p_cls = prediction[:,:,3:]

        loss_obj = self._lambda_noobj * self._criterion_mse(p_obj[~mask], t_obj[~mask]) \
                 + self._criterion_mse(p_obj[mask], t_obj[mask])
        loss_x = self._lambda_coord * self._criterion_mse(p_x[mask], t_x[mask])
        loss_y = self._lambda_coord * self._criterion_mse(p_y[mask], t_y[mask])
        loss_cls = self._criterion_ce(p_cls[mask], torch.argmax(t_cls[mask], axis=1))

        loss = loss_obj + loss_x + loss_y # + loss_cls

        return loss, loss_x, loss_y, loss_obj, loss_cls



    def _calculate_accuracy(self, target, prediction, score_cut=0.5):
        '''
        Calculates the accuracy.

        arguments:
        - target: the minibatch_data from larcv
        - prediction: the prediction from the network
        - score_cut: what cut value to apply to the object confidence output

        returns:
        - iou: intersection over union (union: all cells where we have a real object or where
        we predict to be an object; intersection: cells where there is a real object, and we
        predict that there is an object
        - r^2 averaged over cells with real objects
        - acc_onevtx, the accuracy if there is only one vertex
        '''

        plane_to_accuracies = {}

        for i, (t, p) in enumerate(zip(target, prediction)):
            iou, r2, resolution, acc_onevtx = self._calculate_accuracy_per_plane(t, p)
            plane_to_accuracies[i] = [iou, r2, resolution, acc_onevtx]

        return plane_to_accuracies


    def _calculate_accuracy_per_plane(self, target, prediction):
        '''
        Calculates the accuracy on a single plane

        arguments:
        - target: the minibatch_data from larcv
        - prediction: the prediction from the network
        - score_cut: what cut value to apply to the object confidence output

        returns:
        - iou: intersection over union (union: all cells where we have a real object or where
        we predict to be an object; intersection: cells where there is a real object, and we
        predict that there is an object
        - r^2 averaged over cells with real objects
        - acc_onevtx, the accuracy if there is only one vertex
        '''

        with torch.no_grad():
            # Get the prediction and target for the object confidence
            p_obj = prediction[:,:,:,2]
            t_obj = target[:,:,:,2]

            batch_size = p_obj.size(0)
            #############
            # If there is only one vertex, calculate the accuracy
            # by taking the maximum
            _, idx_pred = p_obj.view(batch_size, -1).max(dim=1)
            _, idx_targ = t_obj.view(batch_size, -1).max(dim=1)

            correct_prediction = torch.eq(idx_pred, idx_targ)
            acc_onevtx = torch.mean(correct_prediction.float())
            #############
            #
            # # Construct bool tensor that shows where we have or expect to have an object


            mask_pred = torch.gt(p_obj, self.score_cut)
            mask_targ = t_obj > self.score_cut


            # Calculate iou
            union = (mask_targ | mask_pred).sum(dim=[1,2]).float() + 1e-7
            intersection = (mask_targ & mask_pred).sum(dim=[1,2]).float()

            iou = torch.mean(intersection / union)

            # Calculate r^2
            x_pred = prediction[mask_targ][:,0]
            y_pred = prediction[mask_targ][:,1]
            x_targ = target[mask_targ][:,0]
            y_targ = target[mask_targ][:,1]

            if self._global_step == 0:
                if not self.args.run.distributed or self._rank == 0:
                    numpy.save('xypred', numpy.array([x_pred.detach().cpu().float(), y_pred.detach().cpu().float()]))
                    numpy.save('xytarg', numpy.array([x_targ.detach().cpu().float(), y_targ.detach().cpu().float()]))

            grid_size_w = prediction.size(1)
            grid_size_h = prediction.size(2)
            # pitch = 0.4 # cm
            a = self.args.data.image_width / grid_size_w * self.args.data.pitch
            b = self.args.data.image_height / grid_size_h * self.args.data.pitch
            resolution = 0

            with torch.no_grad():
                r2 = self._criterion_mse(x_pred, x_targ) + self._criterion_mse(y_pred, y_targ)
                resolution += a * a * self._criterion_mse(x_pred, x_targ)
                resolution += b * b * self._criterion_mse(y_pred, y_targ)

            resolution = torch.sqrt(resolution)

            return iou, r2, resolution, acc_onevtx


    def _compute_metrics(self, logits, vertex_data, loss, plane_to_losses):
        '''
        Computes all metrics and returns them as a dict

        args:
        - logits: prediction (list, one entry for each plane)
        - vertex_data: the vertex data (list, one ontry for each plane)
        - loss: loss
        - plane_to_losses: dict: plane id -> list of losses (x, y, obj, cls)
        '''

        # Call all of the functions in the metrics dictionary:
        metrics = {}


        # Add the total loss
        metrics['loss'] = loss

        # Add the losses per plane p and per loss category
        for p in [0, 1, 2]:
            metrics[f'loss/loss_p{p}_x'] = plane_to_losses[p][0]
            metrics[f'loss/loss_p{p}_y'] = plane_to_losses[p][1]
            metrics[f'loss/loss_p{p}_obj'] = plane_to_losses[p][2]
            metrics[f'loss/loss_p{p}_cls'] = plane_to_losses[p][3]

        # Add the accuracies per plane
        plane_to_accuracies = self._calculate_accuracy(vertex_data, logits)
        for p in [0, 1, 2]:
            metrics[f'accuracy/accuracy_p{p}'] = plane_to_accuracies[p][0]
            metrics[f'iou/iou_p{p}'] = plane_to_accuracies[p][0]
            metrics[f'r2/r2_p{p}'] = plane_to_accuracies[p][1]
            metrics[f'r2/resolution_cm_p{p}'] = plane_to_accuracies[p][2]
            metrics[f'accuracy/acc_onevtx_p{p}'] = plane_to_accuracies[p][3]


        return metrics


    def default_device_context(self):

        if self.args.run.compute_mode == ComputeMode.GPU:
            return torch.cuda.device(0)
        else:
            return contextlib.nullcontext
            # device = torch.device('cpu')

    def default_device(self):

        if self.args.run.compute_mode == ComputeMode.GPU:
            return torch.device("cuda")
        else:
            device = torch.device('cpu')


    def log(self, metrics, saver=''):

        if self._global_step % self.args.mode.logging_iteration == 0:

            self._current_log_time = datetime.datetime.now()

            s = ""

            if 'it.' in metrics:
                # This prints out the iteration for ana steps
                s += "it.: {}, ".format(metrics['it.'])

            # Build up a string for logging:
            # if self._log_keys != []:
            #     s += ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in self._log_keys])
            # else:
            s += ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in metrics if key in self._log_keys])


            try:
                s += " ({:.2}s / {:.2} IOs / {:.2})".format(
                    (self._current_log_time - self._previous_log_time).total_seconds(),
                    metrics['io_fetch_time'],
                    metrics['step_time'])
            except:
                pass

            try:
                s += " (LR: {:.4})".format(
                    self._opt.state_dict()['param_groups'][0]['lr'])
            except:
                pass

            self._previous_log_time = self._current_log_time

            logger.info("{} Step {} metrics: {}".format(saver, self._global_step, s))



    def summary(self, metrics, saver=""):

        if self._saver is None:
            return

        if self._global_step % self.args.mode.summary_iteration == 0:
            for metric in metrics:
                name = metric
                if saver == "test":
                    self._aux_saver.add_scalar(metric, metrics[metric], self._global_step)
                else:
                    self._saver.add_scalar(metric, metrics[metric], self._global_step)


            # try to get the learning rate
            # print self._lr_scheduler.get_lr()
            if saver == "test":
                self._aux_saver.add_scalar("learning_rate", self._opt.state_dict()['param_groups'][0]['lr'], self._global_step)
            else:
                self._saver.add_scalar("learning_rate", self._opt.state_dict()['param_groups'][0]['lr'], self._global_step)
            pass


    def increment_global_step(self):

        previous_epoch = int((self._global_step * self.args.run.minibatch_size) / self._train_data_size)
        self._global_step += 1
        current_epoch = int((self._global_step * self.args.run.minibatch_size) / self._train_data_size)

        self.on_step_end()

        if previous_epoch != current_epoch:
            self.on_epoch_end()

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass

    def to_torch(self, minibatch_data, device=None):



        device_context = self.default_device_context()

        if device is None:
            device = self.default_device()
        with device_context:
            for key in minibatch_data:
                if key == 'entries' or key =='event_ids':
                    continue
                if key == 'image' and self.args.data.image_mode == ImageModeKind.sparse:
                    if self.args.data.input_dimension == 3:
                        minibatch_data['image'] = (
                                torch.tensor(minibatch_data['image'][0]).long(),
                                torch.tensor(minibatch_data['image'][1], device=device),
                                minibatch_data['image'][2],
                            )
                    else:
                        minibatch_data['image'] = (
                                torch.tensor(minibatch_data['image'][0]).long(),
                                torch.tensor(minibatch_data['image'][1], device=device),
                                minibatch_data['image'][2],
                            )
                # elif key == 'image' and self.args.image_mode == ImageModeKind.graph:
                #     minibatch_data[key] = minibatch_data[key].to(device)
                else:
                    minibatch_data[key] = torch.tensor(minibatch_data[key],device=device)

            return minibatch_data

    def train_step(self):


        # For a train step, we fetch data, run a forward and backward pass, and
        # if this is a logging step, we compute some logging metrics.
        torch.autograd.set_detect_anomaly(True)
        self._net.train()
        # print(self._net)

        global_start_time = datetime.datetime.now()

        # Reset the gradient values for this step:
        self._opt.zero_grad()

        # Fetch the next batch of data with larcv
        io_start_time = datetime.datetime.now()
        minibatch_data = self.larcv_fetcher.fetch_next_batch("primary", force_pop=True)
        io_end_time = datetime.datetime.now()

        if self._global_step == 0 and self._rank == 0:
            numpy.save("img",minibatch_data['image'])
            numpy.save("vtx",minibatch_data['vertex'])

        minibatch_data = self.to_torch(minibatch_data)

        with self.default_device_context():
            # if self._global_step == 0 and self._rank == 0:
            #     # Save one image, as an example
            #     self._saver.add_image('example_image', minibatch_data['image'][0])
            #     # Save the network, so we can see a network graph in tensorboard
            #     self._saver.add_graph(self._net, minibatch_data['image'])

            # Run a forward pass of the model on the input image:
            logits = self._net(minibatch_data['image'])

            vertex_data, vertex_mask = self._target_to_yolo(target=minibatch_data['vertex'],
                                                            logits=logits)
                                                            # n_channels=logits.size(3),
                                                            # grid_size_w=logits.size(1),
                                                            # grid_size_h=logits.size(2))
            # Compute the loss based on the logits
            loss, plane_to_losses = self._calculate_loss(vertex_data,
                                                         vertex_mask,
                                                         logits,
                                                         full=True)
            # Compute the gradients for the network parameters:
            loss.backward()
            # print('weights', self._net.initial_convolution.conv1.weight)
            # print('weights grad', self._net.initial_convolution.conv1.weight.grad)

            # Compute any necessary metrics:
            metrics = self._compute_metrics(logits, vertex_data, loss, plane_to_losses)



        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = self.args.run.minibatch_size / self._seconds_per_global_step
        except:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0

        metrics['io_fetch_time'] = (io_end_time - io_start_time).total_seconds()

        # print("Calculated metrics")


        step_start_time = datetime.datetime.now()
        # Apply the parameter update:
        self._opt.step()
        self._lr_scheduler.step()
        # print("Updated Weights")
        global_end_time = datetime.datetime.now()

        metrics['step_time'] = (global_end_time - step_start_time).total_seconds()


        self.log(metrics, saver="train")

        # print("Completed Log")

        self.summary(metrics, saver="train")

        # print("Summarized")

        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

        # Increment the global step value:
        self.increment_global_step()

        return metrics

    def val_step(self, n_iterations=1):

        # First, validation only occurs on training:
        if not self.is_training() : return

        # Second, validation can not occur without a validation dataloader.
        if self._val_data_size == 0: return

        # perform a validation step
        # Validation steps can optionally accumulate over several minibatches, to
        # fit onto a gpu or other accelerator

        # self._net.eval()

        with torch.no_grad():

            if self._global_step != 0 and self._global_step % self.args.run.aux_iteration == 0:


                # Fetch the next batch of data with larcv
                # (Make sure to pull from the validation set)
                minibatch_data = self.larcv_fetcher.fetch_next_batch("val",force_pop = True)

                # Convert the input data to torch tensors
                minibatch_data = self.to_torch(minibatch_data)

                # Run a forward pass of the model on the input image:
                logits = self._net(minibatch_data['image'])

                # Convert target to yolo format
                vertex_data, vertex_mask = self._target_to_yolo(target=minibatch_data['vertex'],
                                                                logits=logits)
                                                                # n_channels=logits.size(3),
                                                                # grid_size_w=logits.size(1),
                                                                # grid_size_h=logits.size(2))

                # Compute the loss
                loss, plane_to_losses = self._calculate_loss(vertex_data,
                                                             vertex_mask,
                                                             logits,
                                                             full=True)

                # Compute the metrics for this iteration:
                metrics = self._compute_metrics(logits, vertex_data, loss, plane_to_losses)

                self.log(metrics, saver="test")
                self.summary(metrics, saver="test")

                return metrics


    def ana_step(self):

        if self.is_training(): return

        # Set network to eval mode
        self._net.eval()

        # Fetch the next batch of data with larcv
        # Fetch the next batch of data with larcv
        if self._global_step == 0:
            force_pop = False
        else:
            force_pop = True
        minibatch_data = self.larcv_fetcher.fetch_next_batch("primary", force_pop=force_pop)

        # Convert the input data to torch tensors
        minibatch_data = self.to_torch(minibatch_data)

        with torch.no_grad():


            # Run a forward pass of the model on the input image:
            logits = self._net(minibatch_data['image'])

            # Convert target to yolo format
            vertex_data, vertex_mask = self._target_to_yolo(target=minibatch_data['vertex'],
                                                            logits=logits)
                                                            # n_channels=logits.size(3),
                                                            # grid_size_w=logits.size(1),
                                                            # grid_size_h=logits.size(2))

            # Compute the loss
            loss, plane_to_losses = self._calculate_loss(vertex_data,
                                                         vertex_mask,
                                                         logits,
                                                         full=True)

            # Compute the metrics for this iteration:
            metrics = self._compute_metrics(logits, vertex_data, loss, plane_to_losses)

            self.log(metrics)
            self.summary(metrics)
            self.accumulate_metrics(metrics)

        self._global_step += 1
        return metrics


    def accumulate_metrics(self, metrics):

        self.inference_metrics['n'] += 1
        for key in metrics:
            if key not in self.inference_metrics:
                self.inference_metrics[key] = metrics[key]
                # self.inference_metrics[f"{key}_sq"] = metrics[key]**2
            else:
                self.inference_metrics[key] += metrics[key]
                # self.inference_metrics[f"{key}_sq"] += metrics[key]**2

    def inference_report(self):
        if not hasattr(self, "inference_metrics"):
            return
        n = self.inference_metrics["n"]
        total_entries = n*self.args.run.minibatch_size
        logger.info(f"Inference report: {n} batches processed for {total_entries} entries.")
        for key in self.inference_metrics:
            if key == 'n' or '_sq' in key: continue
            value = self.inference_metrics[key] / n
            logger.info(f"  {key}: {value:.4f}")

    def stop(self):
        # Mostly, this is just turning off the io:
        # self._larcv_interface.stop()
        pass


    def checkpoint(self):

        if self.args.mode.checkpoint_iteration == -1:
            return

        if self._global_step % self.args.mode.checkpoint_iteration == 0 and self._global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model()


    def batch_process(self):

        start = time.time()
        post_one_time = None
        post_two_time = None

        times = []

        # This is the 'master' function, so it controls a lot

        # If we're not training, force the number of iterations to the epoch size or less
        if not self.is_training():
            if self.args.run.iterations > int(self._train_data_size/self.args.run.minibatch_size) + 1:
                self.args.run.iterations = int(self._train_data_size/self.args.run.minibatch_size) + 1
                logger.info('Number of iterations set to', self.args.run.iterations)

        # Run iterations
        for i in range(self.args.run.iterations):
            iteration_start = time.time()
            if self.is_training() and self._global_step >= self.args.run.iterations:
                logger.info('Finished training (iteration %d)' % self._global_step)
                self.checkpoint()
                break

            if self.is_training():
                self.val_step()
                self.train_step()
                self.checkpoint()
            else:
                self.ana_step()


            if post_one_time is None:
                post_one_time = time.time()
            elif post_two_time is None:
                post_two_time = time.time()
            times.append(time.time() - iteration_start)


        if self.is_training():
            if self._saver is not None:
                self._saver.close()
            if self._aux_saver is not None:
                self._aux_saver.close()

        end = time.time()

        total_images_per_batch = self.args.run.minibatch_size


        if self.args.mode.name == ModeKind.inference:
            self.inference_report()

        logger.info(f"Total time to batch_process: {end - start:.4f}")
        if post_one_time is not None:
            throughput = (self.args.run.iterations - 1) * total_images_per_batch
            throughput /= (end - post_one_time)
            logger.info("Total time to batch process except first iteration: "
                        f"{end - post_one_time:.4f}"
                        f", throughput: {throughput:.4f}")
        if post_two_time is not None:
            throughput = (self.args.run.iterations - 2) * total_images_per_batch
            throughput /= (end - post_two_time)
            logger.info("Total time to batch process except first two iterations: "
                        f"{end - post_two_time:.4f}"
                        f", throughput: {throughput:.4f}")
        if len(times) > 40:
            throughput = (40) * total_images_per_batch
            throughput /= (numpy.sum(times[-40:]))
            logger.info("Total time to batch process last 40 iterations: "
                        f"{numpy.sum(times[-40:]):.4f}"
                        f", throughput: {throughput:.4f}" )
