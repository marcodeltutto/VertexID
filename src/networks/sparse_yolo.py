import torch
import torch.nn as nn
import numpy as np

import sparseconvnet as scn

class SparseBlock(nn.Module):

    def __init__(self, infilters, outfilters, kernel, batch_norm, activation, nplanes=1):

        nn.Module.__init__(self)

        self.batch_norm = batch_norm
        self.leaky_relu = activation == "leaky"

        self.conv1 = scn.SubmanifoldConvolution(dimension=3,
            nIn         = infilters,
            nOut        = outfilters,
            filter_size = [nplanes,kernel,kernel],
            bias        = False)

        if self.batch_norm:
            if self.leaky_relu: self.bn1 = scn.BatchNormLeakyReLU(outfilters)
            else:               self.bn1 = scn.BatchNormReLU(outfilters)
        else:
            if self.leaky_relu: self.relu = scn.LeakyReLU()
            else:               self.relu = scn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        else:
            out = self.relu(out)

        return out



class SparseResidualBlock(nn.Module):


    def __init__(self, infilters, outfilters1, outfilters2, kernel, batch_norm, activation, nplanes=1):
    # def __init__(self, infilters, outfilters, batch_norm, leaky_relu, nplanes=1):
        nn.Module.__init__(self)

        self.batch_norm = batch_norm
        self.leaky_relu = activation == "leaky"

        self.conv1 = scn.SubmanifoldConvolution(dimension=3,
            nIn         = infilters,
            nOut        = outfilters1,
            filter_size = [nplanes,1,1],
            bias=False)


        if self.batch_norm:
            if self.leaky_relu: self.bn1 = scn.BatchNormLeakyReLU(outfilters1)
            else:                self.bn1 = scn.BatchNormReLU(outfilters1)

        self.conv2 = scn.SubmanifoldConvolution(dimension=3,
            nIn         = outfilters1,
            nOut        = outfilters2,
            filter_size = [nplanes,kernel,kernel],
            bias        = False)

        if self.batch_norm:
            self.bn2 = scn.BatchNormalization(outfilters2)

        self.residual = scn.Identity()

        if self.leaky_relu: self.relu = scn.LeakyReLU()
        else:               self.relu = scn.ReLU()

        self.add = scn.AddTable()

    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)

        if self.batch_norm:
            out = self.bn1(out)
        else:
            out = self.relu(out)

        out = self.conv2(out)

        if self.batch_norm:
            out = self.bn2(out)

        # The addition of sparse tensors is not straightforward, since

        out = self.add([out, residual])

        out = self.relu(out)

        return out





class SparseConvolutionDownsample(nn.Module):

    def __init__(self, infilters, outfilters, kernel, stride, batch_norm, activation, nplanes=1):
        nn.Module.__init__(self)

        self.batch_norm = batch_norm
        self.leaky_relu = activation == "leaky"

        self.conv = scn.Convolution(dimension=3,
            nIn             = infilters,
            nOut            = outfilters,
            filter_size     = [nplanes,kernel,kernel],
            filter_stride   = [1,stride,stride],
            bias            = False
        )

        if self.batch_norm:
            self.bn   = scn.BatchNormalization(outfilters)

        if self.leaky_relu: self.relu = scn.LeakyReLU()
        else:               self.relu = scn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm:
            out = self.bn(out)

        out = self.relu(out)
        return out


class SparseBlockSeries(torch.nn.Module):

    def __init__(self, infilters, outfilters1, outfilters2, n_blocks, kernel, batch_norm, activation, nplanes=1, residual=True):

        torch.nn.Module.__init__(self)




        if residual:
            self.blocks = [ SparseResidualBlock(infilters, outfilters1, outfilters2, kernel, batch_norm, activation, nplanes=nplanes) for i in range(n_blocks) ]
        else:
            self.blocks = [ SparseBlock(infilters, outfilters1 , kernel, batch_norm, activation, nplanes=nplanes) for i in range(n_blocks)]

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x



class YOLOBlock(nn.Module):
    '''
    A YOLO block
    '''

    def __init__(self, inp_dim_w, inp_dim_h, num_classes):

        nn.Module.__init__(self)

        self._inp_dim_w = inp_dim_w
        self._inp_dim_h = inp_dim_h
        self._num_classes = num_classes
        # self._cuda = cuda

    def predict_transform(self, prediction):
        '''
        This function takes a detection feature map
        and turns it into a 2-D tensor,
        where each row of the tensor corresponds
        to attributes of a bounding box.
        '''

        # Permute to make the tensor as (batch_id:size_x:size_y:features)
        prediction = prediction.permute(0,2,3,1)

        # Apply sigmoid to x, y, obj features
        x  = torch.sigmoid(prediction[:,:,:,0:2])
        prediction = torch.cat([x, prediction[:,:,:,2:]], dim=-1)

        return prediction


    def forward(self, x):

        x = self.predict_transform(x)

        return x





def filter_increase(n_filters):
    return n_filters * 2




class YOLO(nn.Module):

    def __init__(self, input_shape, args):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the args module

        # self.nplanes    = args.nplanes
        # self.label_mode = args.label_mode

        self.input_shape = input_shape
        # self.anchors = args.yolo_anchors
        self.num_classes = args.yolo_num_classes

        self.input_tensor = scn.InputLayer(dimension=3, spatial_size=[3,*input_shape[1:]])

        self._x_yolo = None

        # if args.compute_mode == "CPU": self._cuda = False
        # else: self._cuda = True

        prev_filters = 1 #3
        n_filters = 32

        #
        # First convolutional block
        #
        self.initial_convolution = SparseBlock(
            infilters  = prev_filters,
            outfilters = n_filters,
            kernel     = args.kernel_size,
            batch_norm = args.batch_norm,
            activation = 'leaky')
        prev_filters = n_filters
        n_filters = filter_increase(n_filters)


        # This is taken from the original Yolo design,
        # it means that the first residial block is done
        # only one time, the second 2 times, the third 8 times,
        # and so on...
        self.blocks_multiplicity = [1, 2, 8, 8, 4]

        # This is the number of downsampling/res blocks
        # that we have in the network, 5 in the case of
        # original yolo
        self.n_core_blocks = 5

        #
        # Downsampling and residual blocks
        #
        self.downsample = torch.nn.ModuleList()
        self.residual   = torch.nn.ModuleList()

        for i in range(len(self.blocks_multiplicity)):

            # Downlsampling block
            self.downsample.append(SparseConvolutionDownsample(
                                        infilters=prev_filters,
                                        outfilters=n_filters,
                                        kernel=2,
                                        stride=2,
                                        batch_norm=args.batch_norm,
                                        activation='leaky'))

            # Residual block series
            self.residual.append(
                SparseBlockSeries(
                    infilters   = n_filters,
                    outfilters1 = prev_filters,
                    outfilters2 = n_filters,
                    n_blocks    = self.blocks_multiplicity[i],
                    kernel      = args.kernel_size,
                    batch_norm  = args.batch_norm,
                    activation  = "leaky",
                    residual    = True))



            prev_filters = n_filters
            n_filters = filter_increase(n_filters)

        # Now there is another series of convolutional blocks
        # Parameters below correspond to the darknet configuration
        # Basically there are 7 convolutional blocks, with the
        # following caracteristics:
        # batch_normalize = [True] * 7
        # pad = [1] * 7
        # stride = [1] * 7
        # filter_sizes = [512, 1024, 512, 1024, 512, 1024, 5]
        # kernel_size = [1, 3, 1, 3, 1, 3, 1]
        # activation = ['leaky', 'leaky', 'leaky', 'leaky', 'leaky', 'leaky', 'linear']

        # Just one last bottleneck layer
        batch_normalize = [True] * 1
        pad = [1] * 1
        stride = [1] * 1
        filter_sizes = [5]
        kernel_size = [3]
        activation = ['leaky']

        self.convolution_blocks_1 = []

        for i in range(0, len(filter_sizes)):

            self.convolution_blocks_1.append(SparseBlock(infilters=prev_filters,
                                                   outfilters=filter_sizes[i],
                                                   kernel=kernel_size[i],
                                                   batch_norm=batch_normalize[i],
                                                   activation=activation[i]))

            prev_filters = filter_sizes[i]

            self.add_module("convolution_block_1_{}".format(i), self.convolution_blocks_1[-1])

        self.sparse_to_dense = scn.SparseToDense(dimension=3, nPlanes=filter_sizes[i])

        self.yololayer_1 = YOLOBlock(inp_dim_w=self.input_shape[1],
                                     inp_dim_h=self.input_shape[2],
                                     num_classes=self.num_classes,
                                     # cuda=self._cuda
                                     )
        # self.add_module("yololayer_1", self.yololayer_1)



    def forward(self, x):
        # print(x)
        batch_size = x[-1]

        # Reshape this tensor into the right shape to apply this multiplane network.
        self.nplanes = 3
        # print(x[0].shape)
        x = self.input_tensor(x)
        # print('initial', x.spatial_size)
        x = self.initial_convolution(x)
        # print('after initial_convolution', x.spatial_size, x.features.shape)

        for i in range(len(self.blocks_multiplicity)):
            x = self.downsample[i](x)
            # print(i, 'after downsample', x.spatial_size, x.features.shape)
            x = self.residual[i](x)
            # print(i, 'after residual', x.spatial_size, x.features.shape)

        for i in range(0, len(self.convolution_blocks_1)):
            x = self.convolution_blocks_1[i](x)
            # print(i, 'after convolution_blocks_1', x.spatial_size, x.features.shape)
        x = self.sparse_to_dense(x)
        # print(x.shape)
        x = torch.chunk(x, chunks=self.nplanes, dim=2)
        x = tuple(torch.squeeze(_x, dim=2) for _x in x)
        # This is shaped
        # print(x[0].shape)
        x = tuple(self.yololayer_1(_x) for _x in x)
        # print('after yolo_1', x[0].shape)
        return x
