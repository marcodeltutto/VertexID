import os
import time
import glob

from . import data_transforms
import tempfile

import numpy
import h5py

import logging
logger = logging.getLogger()

from larcv.config_builder import ConfigBuilder
from src.config import ImageModeKind

class larcv_fetcher(object):

    def __init__(self, config, mode, distributed, access_mode, dimension, data_format, downsample_images, seed=None):

        if mode not in ['train', 'inference', 'iotest']:
            raise Exception("Larcv Fetcher can't handle mode ", mode)

        if access_mode != "serial_access" and mode == "inference":
            logger.warn("Using random blocks in inference - possible bug!")

        if distributed:
            from larcv import distributed_queue_interface
            self._larcv_interface = distributed_queue_interface.queue_interface(
                random_access_mode=access_mode, seed=seed)
        else:
            from larcv import queueloader
            self._larcv_interface = queueloader.queue_interface(
                random_access_mode=access_mode, seed=seed)

        self.config            = config
        self.mode              = mode
        self.image_mode        = data_format
        self.input_dimension   = dimension
        self.distributed       = distributed
        self.downsample_images = downsample_images


        self.writer     = None


    def __del__(self):
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.finalize()



    def prepare_sample(self, name, input_file, batch_size, color=None, start_index = 0, print_config=True):

        # If quotations are in the file name, remove them
        if "\"" in input_file:
            input_file = input_file.replace("\"", "", 2)

        files = glob.glob(input_file)

        if len(files) == 0:
            raise Exception(f"Cannot find files with pattern {input_file}.")

        logger.info(f'larcv_fetcher: {name}: using {len(files)} file(s).')

        # First, verify the files exist:
        for f in files:
            if not os.path.exists(f):
                raise Exception(f"File {f} not found")


        cb = ConfigBuilder()
        cb.set_parameter(files, "InputFiles")
        cb.set_parameter(3, "ProcessDriver", "IOManager", "Verbosity")
        cb.set_parameter(3, "ProcessDriver", "Verbosity")
        cb.set_parameter(3, "Verbosity")

        # Build up the data_keys:
        data_keys = {}
        data_keys['image'] = name+'data'

        # Need to embed the data in SBND:
        if self.config.name == "SBND" and self.input_dimension == 2:
            cb.add_preprocess(
                datatype        = "sparse2d",
                producer        = "sbndwire",
                process         = "Embed",
                OutputProducer  = "sbndwire",
                TargetSize      = [2048,1280]
            )

        # Downsampling
        if self.downsample_images != 0:
            cb.add_preprocess(
                datatype = "sparse2d",
                Product = "sparse2d",
                producer = "sbndwire" if self.config.name == 'SBND' else "dunevoxels",
                process  = "Downsample",
                OutputProducer = "sbndwire" if self.config.name == 'SBND' else "dunevoxels",
                Downsample = 2**self.downsample_images,
                PoolType = 1 # average,
                # PoolType = 2 # max
            )


        # Need to load up on data fillers.
        if self.input_dimension == 2:
            cb.add_batch_filler(
                datatype   = "sparse2d",
                producer   = "sbndwire" if self.config.name == 'SBND' else "dunevoxels",
                name       = name+"data",
                MaxVoxels = 20000,
                Augment    = False,
                Channels   = [0, 1, 2],
                )

        else:
            cb.add_batch_filler(
                datatype  = "sparse3d",
                producer  = "dunevoxels",
                name      = name+"data",
                MaxVoxels = 30000,
                Augment   = False,
                )

        # Add something to convert the neutrino particles into bboxes:
        if self.config.name == 'DUNE':
            cb.add_preprocess(
                datatype = "particle",
                producer = "neutrino",
                process  = "BBoxFromParticle",
                OutputProducer = "neutrino"
            )
            cb.add_batch_filler(
                datatype  = "bbox3d",
                producer  = "neutrino",
                name      = name+"bbox",
                MaxBoxes  = 2,
                Channels  = [0,]
                )
        else:
            cb.add_batch_filler(
                datatype  = "bbox2d",
                producer  = "bbox_neutrino",
                name      = name+"bbox",
                MaxBoxes  = 1,
                Channels  = [0,1,2]
                )


        # Add the label configs:
        # for label_name, l in zip(['neut', 'prot', 'cpi', 'npi'], [3, 3, 2, 2]):
        #     cb.add_batch_filler(
        #         datatype     = "PID",
        #         producer     = f"{label_name}ID",
        #         name         = name+f'label_{label_name}',
        #         PdgClassList = [i for i in range(l)]
        #     )
        #     data_keys[f'label_{label_name}'] = name+f'label_{label_name}'


        if print_config:
            logger.info(cb.print_config())

        # Prepare data managers:
        io_config = {
            'filler_name' : name,
            'filler_cfg'  : cb.get_config(),
            'verbosity'   : 5,
            'make_copy'   : False
        }

        # Assign the keywords here:
        self.keyword_label = []
        for key in data_keys.keys():
            if key != 'image':
                self.keyword_label.append(key)

        data_keys.update({"vertex" : name+"bbox"})



        self._larcv_interface.prepare_manager(name, io_config, batch_size, data_keys, color=color)

        # TODO: READ THIS FROM IMAGE META:
        self.vertex_origin    = numpy.asarray([0.,-100.,0.])
        self.image_dimensions = numpy.asarray([360., 200., 500.])



        if self.mode == "inference" and not self.distributed:
            self._larcv_interface.set_next_index(name, start_index)

        while self._larcv_interface.is_reading(name):
            time.sleep(0.1)

        # # Here, we pause in distributed mode to make sure all loaders are ready:
        # if self.distributed:
        #     from mpi4py import MPI
        #     MPI.COMM_WORLD.Barrier()

        return self._larcv_interface.size(name)



    def fetch_minibatch_dims(self, name):
        return self._larcv_interface.fetch_minibatch_dims(name)

    def input_shape(self, name):

        dims = self.fetch_minibatch_dims(name)

        return dims['image']

    def output_shape(self, name):

        dims = self.fetch_minibatch_dims(name)

        # This sets up the necessary output shape:
        output_shape = { key : dims[key] for key in self.keyword_label}

        return output_shape

    def fetch_next_batch(self, name, force_pop=False):

        metadata=True

        pop = True
        if not force_pop:
            pop = False


        while self._larcv_interface.is_reading(name):
            # print("Sleeping in larcv_fetcher")
            time.sleep(0.01)

        minibatch_data = self._larcv_interface.fetch_minibatch_data(name,
            pop=pop,fetch_meta_data=metadata)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(name)


        # This brings up the next data to current data
        if pop:
            # print(f"Preparing next {name}")
            self._larcv_interface.prepare_next(name)
            # time.sleep(0.1)


        # If the returned data is None, return none and don't load more:
        if minibatch_data is None:
            return minibatch_data

        # Reshape as needed from larcv:
        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])


        # Parse out the vertex info:
        if self.config.name == 'DUNE':
            minibatch_data['vertex'] = minibatch_data['vertex'][:,:,0,0:3]
            minibatch_data['vertex'] = minibatch_data['vertex'].reshape((-1, 3))

        else:
            minibatch_data['vertex'] = minibatch_data['vertex'][:,:,0,0:2]

            # These should be retrieved by the image meta TODO
            width = 614.40
            height = 399.51
            min_x = [-9.6, -9.6, -57.59]
            min_y = [1.87, 1.87, 1.87]

            for p in [0, 1, 2]:
                minibatch_data['vertex'][:,p,0] = self.config.data.image_width * (minibatch_data['vertex'][:,p,0] - min_x[p]) / width
                minibatch_data['vertex'][:,p,1] = self.config.data.image_height * (minibatch_data['vertex'][:,p,1] - min_y[p]) / height

        # print('minibatch_data[vertex]:', minibatch_data['vertex'])

        # Also, we map the vertex from 0 to 1 across the image.  The image size is
        # [360, 200, 500] and the origin is at [0, -100, 0]
        # minibatch_data['vertex'] += self.vertex_origin
        # minibatch_data['vertex'] /= self.image_dimensions


        # Here, do some massaging to convert the input data to another format, if necessary:
        if self.image_mode == ImageModeKind.dense:
            # Need to convert sparse larcv into a dense numpy array:
            if self.input_dimension == 3:
                minibatch_data['image'] = data_transforms.larcvsparse_to_dense_3d(minibatch_data['image'])
            else:
                x_dim = int(self.config.data.image_width / 2**self.downsample_images)
                y_dim = int(self.config.data.image_height / 2**self.downsample_images)
                minibatch_data['image'] = data_transforms.larcvsparse_to_dense_2d(minibatch_data['image'], dense_shape=[x_dim, y_dim])
        elif self.image_mode == ImageModeKind.sparse:
            # Have to convert the input image from dense to sparse format:
            if self.input_dimension == 3:
                minibatch_data['image'] = data_transforms.larcvsparse_to_scnsparse_3d(minibatch_data['image'])
            else:
                minibatch_data['image'] = data_transforms.larcvsparse_to_scnsparse_2d(minibatch_data['image'])
        # elif self.image_mode == ImageModeKind.graph:
        #
        #     if self.input_dimension == 3:
        #         minibatch_data['image'] = data_transforms.larcvsparse_to_pointcloud_3d(minibatch_data['image'])
        #     else:
        #         minibatch_data['image'] = data_transforms.larcvsparse_to_pointcloud_2d(minibatch_data['image'])

        else:
            raise Exception("Image Mode not recognized")

        return minibatch_data


    def prepare_writer(self, input_file, output_file):

        from larcv import larcv_writer
        config = io_templates.output_io(input_file  = input_file)

        # Generate a named temp file:
        main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        main_file.write(config.generate_config_str())

        main_file.close()

        self.writer = larcv_writer.larcv_writer(main_file.name, output_file)

    def write(self, data, producer, entry, event_id):
        self.writer.write(data, datatype='sparse2d', producer=producer, entry=entry, event_id=event_id)
