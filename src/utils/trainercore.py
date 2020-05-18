import os
import tempfile
from collections import OrderedDict
import datetime

import numpy
import torch
import tensorboardX

from larcv import queueloader

from . import data_transforms
from . import io_templates


class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args):
        self.args = args
        if not self.args.training:
            self._larcv_interface = queueloader.queue_interface(random_access_mode="serial_access")
        else:
            self._larcv_interface = queueloader.queue_interface(random_access_mode="random_blocks")

        self._iteration       = 0
        self._global_step     = -1

        self._cleanup         = []

    def __del__(self):
        for f in self._cleanup:
            os.unlink(f.name)


    def _initialize_io(self, color=0):

        # First, verify the files exist:
        if not os.path.exists(self.args.file):
            raise Exception(f"File {self.args.file} not found")

        # Use the templates to generate a configuration string, which we store into a temporary file
        if self.args.training:
            config = io_templates.train_io(input_file=self.args.file, image_dim=self.args.input_dimension,
                label_mode=self.args.label_mode, compression=self.args.downsample_images)
        else:
            config = io_templates.ana_io(input_file=self.args.file, image_dim=self.args.input_dimension,
                label_mode=self.args.label_mode, compression=self.args.downsample_images)


        # Generate a named temp file:
        main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        main_file.write(config.generate_config_str())
        print(config.generate_config_str())

        main_file.close()
        self._cleanup.append(main_file)

        # Prepare data managers:
        io_config = {
            'filler_name' : config._name,
            'filler_cfg'  : main_file.name,
            'verbosity'   : 5,
            'make_copy'   : True
        }

        # Build up the data_keys:
        data_keys = OrderedDict()
        data_keys['image'] = 'data'
        for proc in config._process_list._processes:
            if proc._name == 'data' or 'Downsample' in proc._name:
                continue
            else:
                data_keys[proc._name] = proc._name

        # Assign the keywords here:
        if self.args.label_mode == 'all':
            self.args.keyword_label = 'label'
        else:
            self.args.keyword_label = []
            for key in data_keys.keys():
                if key != 'image':
                    self.args.keyword_label.append(key)

        print(data_keys)

        if self.args.distributed:
            self._larcv_interface.prepare_manager(mode='primary',
                                                  io_config=io_config,
                                                  minibatch_size=self.args.minibatch_size,
                                                  data_keys=data_keys,
                                                  # files=self.args.file,
                                                  random_access_mode="random_blocks",
                                                  read_option="read_from_all_ranks_mpi")
        else:
            self._larcv_interface.prepare_manager(mode      = 'primary',
                                                  io_config = io_config,
                                                  minibatch_size = self.args.minibatch_size,
                                                  data_keys = data_keys )


        if not self.args.training:
            self._larcv_interface.set_next_index('primary', self.args.start_index)

        # All of the additional tools are in case there is a test set up:
        if self.args.aux_file is not None:


            if self.args.training:
                config = io_templates.test_io(input_file=self.args.aux_file, image_dim=self.args.input_dimension,
                    label_mode=self.args.label_mode)

                # Generate a named temp file:
                aux_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                aux_file.write(config.generate_config_str())
                # print([proc._name for proc in config._process_list._processes])

                aux_file.close()
                self._cleanup.append(aux_file)
                io_config = {
                    'filler_name' : config._name,
                    'filler_cfg'  : aux_file.name,
                    'verbosity'   : 5,
                    'make_copy'   : True
                }

                # Build up the data_keys:
                data_keys = OrderedDict()
                data_keys['image'] = 'aux_data'
                for proc in config._process_list._processes:
                    if proc._name == 'aux_data':
                        continue
                    else:
                        data_keys[proc._name] = proc._name


                if self.args.distributed:
                    self._larcv_interface.prepare_manager(mode='aux',
                                                          io_config=io_config,
                                                          minibatch_size=self.args.aux_minibatch_size,
                                                          data_keys=data_keys,
                                                          # files=self.args.aux_file,
                                                          random_access_mode="serial_access",
                                                          read_option="read_from_all_ranks_mpi")
                else:
                    self._larcv_interface.prepare_manager('aux', io_config, self.args.aux_minibatch_size, data_keys, files=self.args.aux_file)

        if 'output_file' in self.args and self.args.output_file is not None:
            if not self.args.training:
                config = io_templates.output_io(input_file=self.args.file, output_file=self.args.output_file)

                out_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                out_file.write(config.generate_config_str())
                print(config.generate_config_str())

                out_file.close()
                self._cleanup.append(out_file)

                self._larcv_interface.prepare_writer(io_config=out_file.name, input_files=self.args.file, output_file=self.args.output_file)



    def init_network(self):

        dims = self._larcv_interface.fetch_minibatch_dims('primary')

        # This sets up the necessary output shape:
        output_shape = dims[self.args.keyword_label]
        input_shape = dims['image']
        print('OVERRIDING input_shape, remember to fix this once you have a proper file!', input_shape)
        input_shape = [input_shape[0], self.args.image_width, self.args.image_height]


        # To initialize the network, we see what the name is
        # and act on that:
        if self.args.network == "yolo":
            from src.networks import yolo
            self._net = yolo.YOLO(input_shape, output_shape, self.args)
        else:
            raise Exception(f"Couldn't identify network {self.args.network}")


        if self.args.training:
            self._net.train(True)


    def initialize(self, io_only=False):


        self._initialize_io()


        if io_only:
            return


        self.init_network()

        n_trainable_parameters = 0
        for var in self._net.parameters():
            n_trainable_parameters += numpy.prod(var.shape)
        print("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))

        self.init_optimizer()

        self.init_saver()

        state = self.restore_model()

        if state is not None:
            self.load_state(state)
        else:
            self._global_step = 0


        if self.args.compute_mode == "CPU":
            pass
        if self.args.compute_mode == "GPU":
            self._net.cuda()

        if self.args.label_mode == 'all':
            self._log_keys = ['loss', 'accuracy']
        elif self.args.label_mode == 'split':
            self._log_keys = ['loss']
            for key in self.args.keyword_label:
                self._log_keys.append('acc/{}'.format(key))


    def get_device(self):
        # Convert the input data to torch tensors
        if self.args.compute_mode == "GPU":
            device = torch.device('cuda')
            # print(device)
        else:
            device = torch.device('cpu')


        return device

    def init_optimizer(self):

        # Create an optimizer:
        if self.args.optimizer == "SDG":
            self._opt = torch.optim.SGD(self._net.parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay)
        else:
            self._opt = torch.optim.Adam(self._net.parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay)





        device = self.get_device()

        # weights = torch.tensor([prediction.size(2)*prediction.size(3)-1., 1.], device=self.get_device())
        # weights = torch.sum(weights) / weights

        # bce_loss = torch.nn.BCELoss(weight=weights)
        self._criterion_mse = torch.nn.MSELoss()
        self._criterion_ce = torch.nn.CrossEntropyLoss()

        n_empty = 48 * 32 - 1.
        n_occupied = 1.
        self._weight_empty = 1 / n_empty
        self._weight_occupied = 1 - self._weight_empty

        # here we store the loss weights:
        # print('NEED TO CHANGE WEIGHTS AND LABELS!!!')
        # if self.args.label_mode == 'all':
        #     self._label_weights = torch.tensor([
        #         4930., 247., 2311., 225., 11833., 1592., 3887., 378., 4966., 1169., 1944., 335.,
        #         5430., 201., 1630., 67., 13426., 1314., 3111., 243., 5070., 788., 1464., 163.,
        #         5851.,3267.,1685.,183.,7211.,3283.,2744.,302.,5804.,1440.,1302., 204.
        #         ], device=device)
        #     weights = torch.sum(self._label_weights) / self._label_weights
        #     self._label_weights = weights / torch.sum(weights)

        #     self._criterion = torch.nn.CrossEntropyLoss(weight=self._label_weights)


        # elif self.args.label_mode == 'split':
        #     # These are the raw category occurences
        #     self._label_weights = {
        #         'label_cpi'  : torch.tensor([50932., 61269.], device=device),
        #         'label_prot' : torch.tensor([36583., 46790., 28828.], device=device),
        #         'label_npi'  : torch.tensor([70572., 41629.], device=device),
        #         'label_neut' : torch.tensor([39452., 39094., 33655.], device=device)
        #     }

        #     self._criterion = {}

        #     for key in self._label_weights:
        #         weights = torch.sum(self._label_weights[key]) / self._label_weights[key]
        #         self._label_weights[key] = weights / torch.sum(weights)

        #     for key in self._label_weights:
        #         self._criterion[key] = torch.nn.CrossEntropyLoss(weight=self._label_weights[key])


    def init_saver(self):

        # This sets up the summary saver:
        if self.args.training:
            self._saver = tensorboardX.SummaryWriter(self.args.log_directory)

        if self.args.aux_file is not None and self.args.training:
            self._aux_saver = tensorboardX.SummaryWriter(self.args.log_directory + "/test/")
        elif self.args.aux_file is not None and not self.args.training:
            self._aux_saver = tensorboardX.SummaryWriter(self.args.log_directory + "/val/")

        else:
            self._aux_saver = None
        # This code is supposed to add the graph definition.
        # It doesn't currently work
        # temp_dims = list(dims['image'])
        # temp_dims[0] = 1
        # dummy_input = torch.randn(size=tuple(temp_dims), requires_grad=True)
        # self._saver.add_graph(self._net, (dummy_input,))

        # Here, either restore the weights of the network or initialize it:


    def restore_model(self):
        ''' This function attempts to restore the model from file
        '''

        _, checkpoint_file_path = self.get_model_filepath()

        print(checkpoint_file_path)

        if not os.path.isfile(checkpoint_file_path):
            print("Returning none!")
            return None
        # Parse the checkpoint file and use that to get the latest file path

        with open(checkpoint_file_path, 'r') as _ckp:
            for line in _ckp.readlines():
                if line.startswith("latest: "):
                    chkp_file = line.replace("latest: ", "").rstrip('\n')
                    chkp_file = os.path.dirname(checkpoint_file_path) + "/" + chkp_file
                    print("Restoring weights from ", chkp_file)
                    break

        if self.args.compute_mode == "CPU":
            state = torch.load(chkp_file, map_location='cpu')
        else:
            state = torch.load(chkp_file)

        return state

    def load_state(self, state):


        self._net.load_state_dict(state['state_dict'])
        self._opt.load_state_dict(state['optimizer'])
        self._global_step = state['global_step']

        # If using GPUs, move the model to GPU:
        if self.args.compute_mode == "GPU":
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
        if self.args.checkpoint_directory == None:
            file_path= self.args.log_directory  + "/checkpoints/"
        else:
            file_path= self.args.checkpoint_directory  + "/checkpoints/"


        name = file_path + 'model-{}.ckpt'.format(self._global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path


    def _target_to_yolo(self, target, n_channels=5, grid_size_w=56, grid_size_h=40, device=None):
        '''
        Takes the vertex data from larcv and transform it
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
        pitch = 0.4 * 2**self.args.downsample_images
        padding_x = int(286 / 2**self.args.downsample_images)
        padding_y = int(124 / 2**self.args.downsample_images)

        batch_size = target.size(0)

        # Convert the input data to torch tensors
        if self.args.compute_mode == "GPU":
            if device is None:
                device = torch.device('cuda')
        else:
            if device is None:
                device = torch.device('cpu')

        target_out = torch.zeros(batch_size, grid_size_w, grid_size_h, n_channels, device=device)
        mask = torch.zeros(batch_size, grid_size_w, grid_size_h, dtype=torch.bool, device=device)

        step_w = (self.args.image_width / 2**self.args.downsample_images) / grid_size_w
        step_h = (self.args.image_height / 2**self.args.downsample_images) / grid_size_h

        # print('vertex x', target[0, 2], 'y', target[0, 0])
        # print('vertex x', (target[0, 2]/pitch + padding_x/2), 'y', (target[0, 0]/pitch + padding_y/2))
        # print('step_w', step_w, 'step_h', step_h)


        for batch_id in range(batch_size):
            t_x = (target[batch_id, 2]/pitch + padding_x/2) / step_w
            t_i = int(t_x)
            t_y = (target[batch_id, 0]/pitch + padding_y/2) / step_h
            t_j = int(t_y)

            target_out[batch_id, t_i, t_j, 0] = t_x - t_i
            target_out[batch_id, t_i, t_j, 1] = t_y - t_j
            target_out[batch_id, t_i, t_j, 2] = 1.
            target_out[batch_id, t_i, t_j, 3] = 1.

            mask[batch_id, t_i, t_j] = 1

            # print('Batch', batch_id, 't_i', t_i, 't_j', t_j)

        return target_out, mask

    def _calculate_loss(self, minibatch_data, prediction):
        ''' 
        Calculate the loss.

        arguments:
        - minibatch_data: the minibatch_data from larcv
        - prediction: the prediction from the network

        returns:
        - a single scalar for the optimizer to use
        '''

        target, mask = self._target_to_yolo(target=minibatch_data['vertex'],
                                            n_channels=prediction.size(3),
                                            grid_size_w=prediction.size(1),
                                            grid_size_h=prediction.size(2))


        original_shape = prediction.size()
        bs = original_shape[0]

        prediction = prediction.view(prediction.size(0), prediction.size(1)*prediction.size(2), prediction.size(3))
        target     = target.view(target.size(0), target.size(1)*target.size(2), target.size(3))
        mask       = mask.view(target.size(0), -1).byte()

        t_x   = target[:,:,0]
        t_y   = target[:,:,1]
        t_obj = target[:,:,2]
        t_cls = target[:,:,3:]

        p_x   = prediction[:,:,0]
        p_y   = prediction[:,:,1]
        p_obj = prediction[:,:,2]
        p_cls = prediction[:,:,3:]

        # i = 0
        # for t,p in zip(t_obj[0], p_obj[0]):
        #     print(i, t, p)
        #     i += 1

        loss_obj = self._weight_empty * self._criterion_mse(p_obj[~mask], t_obj[~mask]) \
                 + self._weight_occupied * self._criterion_mse(p_obj[mask], t_obj[mask])
        loss_x = self._criterion_mse(p_x[mask], t_x[mask])
        loss_y = self._criterion_mse(p_y[mask], t_y[mask])
        loss_cls = self._criterion_ce(p_cls[mask], torch.argmax(t_cls[mask], dim=1))

        loss = loss_x + loss_y + loss_obj + loss_cls

        return loss


    def _calculate_accuracy(self, minibatch_data, prediction, score_cut=0.5):
        ''' 
        Calculate the accuracy.

        arguments:
        - minibatch_data: the minibatch_data from larcv
        - prediction: the prediction from the network
        - score_cut: what cut value to apply to the object confidence output

        returns:
        - iou: intersection over union (union: all cells where we have a real object or where
        we predict to be an object; intersection: cells where there is a real object, and we 
        predict that there is an object
        - r^2 averaged over cells with real objects
        '''

        target, _ = self._target_to_yolo(target=minibatch_data['vertex'],
                                         n_channels=prediction.size(3),
                                         grid_size_w=prediction.size(1),
                                         grid_size_h=prediction.size(2))

        # Get the predcition and target for the object confidence
        p_obj = prediction[:,:,:,2]
        t_obj = target[:,:,:,2]

        # Construct bool tensor that shows where we have or expect to have an object
        mask_pred = p_obj > score_cut
        mask_targ = t_obj > score_cut

        # Calculate iou
        union = torch.sum(mask_targ | mask_pred)
        intersection = torch.sum(p_obj[mask_targ] > score_cut)
        iou = intersection.float() / union.float()

        # Calculate r^2
        x_pred = prediction[mask_targ][:,0]
        y_pred = prediction[mask_targ][:,1]
        x_targ = target[mask_targ][:,0]
        y_targ = target[mask_targ][:,1]

        with torch.no_grad():
            r2 = self._criterion_mse(x_pred, x_targ) + self._criterion_mse(y_pred, y_targ)

        return iou.item(), r2.item()


    def _compute_metrics(self, logits, minibatch_data, loss):

        # Call all of the functions in the metrics dictionary:
        metrics = {}

        metrics['loss'] = loss.data

        iou, r2 = self._calculate_accuracy(minibatch_data, logits)
        metrics['accuracy'] = iou
        metrics['iou'] = iou
        metrics['r2'] = r2

        return metrics

    def log(self, metrics, saver=''):


        if self._global_step % self.args.logging_iteration == 0:

            self._current_log_time = datetime.datetime.now()

            s = ""

            if 'it.' in metrics:
                # This prints out the iteration for ana steps
                s += "it.: {}, ".format(metrics['it.'])

            # Build up a string for logging:
            if self._log_keys != []:
                s += ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in self._log_keys])
            else:
                s += ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in metrics])


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

            print("{} Step {} metrics: {}".format(saver, self._global_step, s))



    def summary(self, metrics,saver=""):

        if self._saver is None:
            return

        if self._global_step % self.args.summary_iteration == 0:
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


    def fetch_next_batch(self, mode='primary', metadata=False):

        # For the serial mode, call next here:
        self._larcv_interface.prepare_next(mode)

        minibatch_data = self._larcv_interface.fetch_minibatch_data(mode, pop=True, fetch_meta_data=metadata)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(mode)


        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

        # Strip off the primary/aux label in the keys:
        if mode != 'primary':
            # Can't do this in a loop due to limitations of python's dictionaries.
            minibatch_data["label_cpi"]  = minibatch_data.pop("aux_label_cpi")
            minibatch_data["label_npi"]  = minibatch_data.pop("aux_label_npi")
            minibatch_data["label_prot"] = minibatch_data.pop("aux_label_prot")
            minibatch_data["label_neut"] = minibatch_data.pop("aux_label_neut")


        # Here, do some massaging to convert the input data to another format, if necessary:
        if self.args.image_mode == 'dense':
            # Need to convert sparse larcv into a dense numpy array:
            if self.args.input_dimension == 3:
                minibatch_data['image'] = data_transforms.larcvsparse_to_dense_3d(minibatch_data['image'])
            else:
                x_dim = int(1536/2**self.args.downsample_images)
                y_dim = int(1024/2**self.args.downsample_images)
                minibatch_data['image'] = data_transforms.larcvsparse_to_dense_2d(minibatch_data['image'],
                                                                                  dense_shape=[x_dim, y_dim])
        elif self.args.image_mode == 'sparse':
            # Have to convert the input image from dense to sparse format:
            if self.args.input_dimension == 3:
                minibatch_data['image'] = data_transforms.larcvsparse_to_scnsparse_3d(minibatch_data['image'])
            else:
                minibatch_data['image'] = data_transforms.larcvsparse_to_scnsparse_2d(minibatch_data['image'])
        elif self.args.image_mode == 'graph':
            minibatch_data['image'] = data_transforms.larcvsparse_to_torchgeometric(minibatch_data['image'])
            pass
        else:
            raise Exception("Image Mode not recognized")

        return minibatch_data

    def increment_global_step(self):

        previous_epoch = int((self._global_step * self.args.minibatch_size) / self._epoch_size)
        self._global_step += 1
        current_epoch = int((self._global_step * self.args.minibatch_size) / self._epoch_size)

        self.on_step_end()

        if previous_epoch != current_epoch:
            self.on_epoch_end()

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass

    def to_torch(self, minibatch_data, device=None):

        # Convert the input data to torch tensors
        if self.args.compute_mode == "GPU":
            if device is None:
                device = torch.device('cuda')
            # print(device)

        else:
            if device is None:
                device = torch.device('cpu')


        for key in minibatch_data:
            if key == 'entries' or key =='event_ids':
                continue
            if key == 'image' and self.args.image_mode == "sparse":
                if self.args.input_dimension == 3:
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
            elif key == 'image' and self.args.image_mode == 'graph':
                minibatch_data[key] = minibatch_data[key].to(device)
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
        minibatch_data = self.fetch_next_batch()
        io_end_time = datetime.datetime.now()

        # numpy.save('tmp', minibatch_data['image'])

        minibatch_data = self.to_torch(minibatch_data)

        # Run a forward pass of the model on the input image:
        logits = self._net(minibatch_data['image'])
        # print("Completed forward pass")

        # Compute the loss based on the logits
        loss = self._calculate_loss(minibatch_data, logits)
        # print('loss', loss.item())

        # Compute the gradients for the network parameters:
        loss.backward()
        # print("Completed backward pass")

        # print('weights', self._net.initial_convolution.conv1.weight)
        # print('weights grad', self._net.initial_convolution.conv1.weight.grad)

        # Compute any necessary metrics:
        metrics = self._compute_metrics(logits, minibatch_data, loss)



        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = self.args.minibatch_size / self._seconds_per_global_step
        except:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0

        metrics['io_fetch_time'] = (io_end_time - io_start_time).total_seconds()

        # print("Calculated metrics")


        step_start_time = datetime.datetime.now()
        # Apply the parameter update:
        self._opt.step()
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
        if not self.args.training: return

        # Second, validation can not occur without a validation dataloader.
        if self.args.aux_file is None: return

        # perform a validation step
        # Validation steps can optionally accumulate over several minibatches, to
        # fit onto a gpu or other accelerator

        # self._net.eval()

        with torch.no_grad():

            if self._global_step != 0 and self._global_step % self.args.aux_iteration == 0:


                # Fetch the next batch of data with larcv
                # (Make sure to pull from the validation set)
                minibatch_data = self.fetch_next_batch('aux')

                # Convert the input data to torch tensors
                minibatch_data = self.to_torch(minibatch_data)

                # Run a forward pass of the model on the input image:
                logits = self._net(minibatch_data['image'])

                # # Here, we have to map the logit keys to aux keys
                # for key in logits.keys():
                #     new_key = 'aux_' + key
                #     logits[new_key] = logits.pop(key)



                # Compute the loss
                loss = self._calculate_loss(minibatch_data, logits)

                # Compute the metrics for this iteration:
                metrics = self._compute_metrics(logits, minibatch_data, loss)


                self.log(metrics, saver="test")
                self.summary(metrics, saver="test")

                return metrics

    def ana_step(self, iteration=None):

        if self.args.training: return

        # Set network to eval mode
        self._net.eval()

        # Fetch the next batch of data with larcv
        minibatch_data = self.fetch_next_batch(metadata=True)

        # Convert the input data to torch tensors
        minibatch_data = self.to_torch(minibatch_data)

        # Run a forward pass of the model on the input image:
        with torch.no_grad():
            logits = self._net(minibatch_data['image'])


        if self.args.label_mode == 'all':
            softmax = torch.nn.Softmax(dim=-1)(logits)
        else:
            softmax = { key : torch.nn.Softmax(dim=-1)(logits[key]) for key in logits }

        # Call the larcv interface to write data:
        if self.args.output_file is not None:
            if self.args.label_mode == 'all':
                writable_logits = numpy.asarray(softmax.cpu())
                self._larcv_interface.write_output(data=writable_logits[0], datatype='meta', producer='all',
                    entries=minibatch_data['entries'], event_ids=minibatch_data['event_ids'])
            else:
                for entry in range(self.args.minibatch_size):
                    if iteration > 1 and minibatch_data['entries'][entry] == 0:
                        print ('Reached max number of entries.')
                        break
                    this_entry = [minibatch_data['entries'][entry]]
                    this_event_id = [minibatch_data['event_ids'][entry]]
                    for key in softmax:
                        writable_logits = numpy.asarray(softmax[key].cpu())[entry]
                        self._larcv_interface.write_output(data=writable_logits, datatype='tensor1d', producer=key,
                            entries=this_entry, event_ids=this_event_id)

        # If the input data has labels available, compute the metrics:
        if (self.args.label_mode == 'all' and 'label' in minibatch_data) or \
           (self.args.label_mode == 'split' and 'label_neut' in minibatch_data):
            # Compute the loss
            loss = self._calculate_loss(minibatch_data, logits)

            # Compute the metrics for this iteration:
            metrics = self._compute_metrics(logits, minibatch_data, loss)

            if iteration is not None:
                metrics.update({'it.' : iteration})


            self.log(metrics, saver="ana")
            # self.summary(metrics, saver="test")

            return metrics


    def stop(self):
        # Mostly, this is just turning off the io:
        self._larcv_interface.stop()

    def checkpoint(self):

        if self.args.checkpoint_iteration == -1:
            return

        if self._global_step % self.args.checkpoint_iteration == 0 and self._global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model()


    def batch_process(self):

        # At the begining of batch process, figure out the epoch size:
        self._epoch_size = self._larcv_interface.size('primary')

        # This is the 'master' function, so it controls a lot

        # If we're not training, force the number of iterations to the epoch size or less
        if not self.args.training:
            if self.args.iterations > int(self._epoch_size/self.args.minibatch_size) + 1:
                self.args.iterations = int(self._epoch_size/self.args.minibatch_size) + 1
                print('Number of iterations set to', self.args.iterations)


        # Run iterations
        for i in range(self.args.iterations):
            if self.args.training and self._iteration >= self.args.iterations:
                print('Finished training (iteration %d)' % self._iteration)
                self.checkpoint()
                break

            if self.args.training:
                self.val_step()
                self.train_step()
                self.checkpoint()
            else:
                self.ana_step(i)


        if self.args.training:
            if self._saver is not None:
                self._saver.close()
            if self._aux_saver is not None:
                self._aux_saver.close()
