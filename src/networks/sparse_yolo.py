import torch
import torch.nn as nn
import numpy as np

import sparseconvnet as scn

class SparseBlock(nn.Module):

    def __init__(self, infilters, outfilters, kernel, stride, padding, batch_norm, activation, nplanes=1):

        nn.Module.__init__(self)

        self.batch_norm = batch_norm
        self.leaky_relu = activation == "leaky_relu"

        self.conv1 = scn.SubmanifoldConvolution(dimension=3,
            nIn=infilters,
            nOut=outfilters,
            filter_size=[nplanes,kernel,kernel],
            bias=False)

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


    def __init__(self, infilters, outfilters, kernel, stride, padding, batch_norm, activation, nplanes=1):
    # def __init__(self, infilters, outfilters, batch_norm, leaky_relu, nplanes=1):
        nn.Module.__init__(self)

        self.batch_norm = batch_norm
        self.leaky_relu = activation == "leaky_relu"

        self.conv1 = scn.SubmanifoldConvolution(dimension=3,
            nIn         = infilters,
            nOut        = outfilters,
            filter_size = [nplanes,kernel,kernel],
            bias=False)


        if self.batch_norm:
            if self.leaky_relu: self.bn1 = scn.BatchNormLeakyReLU(outfilters)
            else:                self.bn1 = scn.BatchNormReLU(outfilters)

        self.conv2 = scn.SubmanifoldConvolution(dimension=3,
            nIn         = outfilters,
            nOut        = outfilters,
            filter_size = [nplanes,kernel,kernel],
            bias        = False)

        if self.batch_norm:
            self.bn2 = scn.BatchNormalization(outfilters)

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

    def __init__(self, infilters, outfilters, kernel, stride, padding, batch_norm, activation, nplanes=1):
        nn.Module.__init__(self)

        self.batch_norm = batch_norm
        self.leaky_relu = activation == "leaky_relu"

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

    def __init__(self, infilters, n_blocks, kernel, stride, padding, batch_norm, activation, nplanes=1, residual=True):

        torch.nn.Module.__init__(self)




        if residual:
            self.blocks = [ SparseResidualBlock(infilters, infilters, kernel, stride, padding, batch_norm, activation, nplanes=nplanes) for i in range(n_blocks) ]
        else:
            self.blocks = [ SparseBlock(infilters, infilters, kernel, stride, padding, batch_norm, activation, nplanes=nplanes) for i in range(n_blocks)]

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

    # def __init__(self, inp_dim_w, inp_dim_h, anchors, num_classes, cuda):
    def __init__(self, inp_dim_w, inp_dim_h, anchors, num_classes):

        nn.Module.__init__(self)

        self._inp_dim_w = inp_dim_w
        self._inp_dim_h = inp_dim_h
        self._anchors = anchors
        self._num_classes = num_classes
        # self._cuda = cuda

    def predict_transform(self, prediction):
        '''
        This function takes a detection feature map
        and turns it into a 2-D tensor,
        where each row of the tensor corresponds
        to attributes of a bounding box.
        '''

        prediction = prediction.permute(0,2,3,1)
        prediction[:,:,:,0:2] = torch.sigmoid(prediction[:,:,:,0:2])

        return prediction



        # # The output of YOLO is a convolutional feature map that contains
        # # the bounding box attributes along the depth of the feature map.

        # batch_size = prediction.size(0)
        # stride =  self._inp_dim_w // prediction.size(2)
        # grid_size_w = prediction.size(2) # self._inp_dim_w // stride
        # grid_size_h = prediction.size(3) # self._inp_dim_h // stride
        # bbox_attrs = 3 + self._num_classes # 5 contains pos_x, pos_y, obj_score
        # num_anchors = len(self._anchors)
        # # print('>>> prediction', prediction.shape)

        # # Now, prediction is a tensor with dimensions
        # # n x n x f, with f number of filters and n
        # # is the dimension of the downsample image

        # # The first step is to use 'view' to make this tensor
        # # of dimension (n*n) x f, so that the image dimensions
        # # are aligned in a single vector
        # # prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size_w*grid_size_h)
        # prediction = prediction.view(batch_size, bbox_attrs, grid_size_w*grid_size_h)
        # # print('>>> prediction', prediction.shape)

        # # Then we spaw (transpose) this 2 dimensions
        # # so to have a tensor f x (n*n)
        # prediction = prediction.transpose(1,2).contiguous()
        # # print('>>> prediction', prediction.shape)

        # # Now we are going to reshape this in a way that each row
        # # of this tensor corresponds to attributes of a bounding box
        # # prediction = prediction.view(batch_size, grid_size_w*grid_size_h*num_anchors, bbox_attrs)
        # prediction = prediction.view(batch_size, grid_size_w*grid_size_h, bbox_attrs)
        # # print('>>> prediction', prediction.shape)

        # # The dimensions of the anchors is wrt to the height and width
        # # attributes of the net block. These attributes describe
        # # the dimensions of the input image, which is larger
        # # (by a factor of 'stride') than the detection map.
        # # We need to divide the anchors by the stride of the detection feature map.
        # # anchors = [(a[0]/stride, a[1]/stride) for a in self._anchors]
        # # print('>>> anchors', anchors)

        # # Now, one row of the tensor contains:
        # # - center_x
        # # - center_y
        # # - obj_score

        # # Need to apply sigmoid to center_x, center_y and obj_score
        # prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
        # prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
        # prediction[:,:,2] = torch.sigmoid(prediction[:,:,2])

        # # Need to add the center offsets
        # # grid_x = np.arange(grid_size_w)
        # # grid_y = np.arange(grid_size_h)
        # # a, b = np.meshgrid(grid_x, grid_y)
        # # x_offset = torch.FloatTensor(a).view(-1,1)
        # # y_offset = torch.FloatTensor(b).view(-1,1)

        # # if self._cuda:
        # #     x_offset = x_offset.cuda()
        # #     y_offset = y_offset.cuda()

        # # x_y_offset = torch.cat((x_offset, y_offset), 1).view(-1,2).unsqueeze(0)
        # # prediction[:,:,:2] += x_y_offset

        # return prediction

    def forward(self, x):

        x = self.predict_transform(x)

        return x





def filter_increase(n_filters):
    return n_filters * 2




class YOLO(nn.Module):

    def __init__(self, input_shape, args=None):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the args module

        # self.nplanes    = args.nplanes
        # self.label_mode = args.label_mode

        self.input_shape = input_shape
        # self.anchors = args.yolo_anchors
        self.anchors = [(116, 90), (156, 198), (373, 326)]
        self.num_classes = args.network.yolo_num_classes


        self.input_tensor = scn.InputLayer(dimension=3, spatial_size=[1,1536, 1024])
        spatial_size = [1536, 1024]

        self._x_yolo = None

        # if args.compute_mode == "CPU": self._cuda = False
        # else: self._cuda = True

        prev_filters = 1 #3
        n_filters = 32

        #
        # First convolutional block
        #
        self.initial_convolution = SparseBlock(infilters=prev_filters,
                                         outfilters=n_filters,
                                         kernel=3,
                                         stride=1,
                                         padding=1,
                                         batch_norm=True,
                                         activation='leaky')
        prev_filters = n_filters
        n_filters = filter_increase(n_filters)


        # This is taken from the original Yolo design,
        # it means that the first residial block is done
        # only one time, the second 2 times, the third 8 times,
        # and so on...
        blocks_multiplicity = [1, 2, 8, 8, 4]

        # This is the number of downsampling/res blocks
        # that we have in the network, 5 in the case of
        # original yolo
        self.n_core_blocks = 5

        self.dowsample = []
        self.residual = []

        #
        # Downsampling and residual blocks
        #
        for i in range(0, self.n_core_blocks):

            # Downlsampling block
            self.dowsample.append(SparseConvolutionDownsample(
                                        infilters=prev_filters,
                                        outfilters=n_filters,
                                        kernel=3,
                                        stride=2,
                                        padding=1,
                                        batch_norm=True,
                                        activation='leaky'))

            self.add_module("downsample_{}".format(i), self.dowsample[-1])
            # Residual block series
            self.residual.append(
                SparseBlockSeries(
                    infilters = n_filters,
                    n_blocks  = blocks_multiplicity[i],
                    kernel    = 3,
                    stride    = 2,
                    padding   = 1,
                    batch_norm = True,
                    activation = "leaky",
                    residual  = True))


            self.add_module("resblock_{}".format(i), self.residual[-1])

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
                                                   stride=stride[i],
                                                   padding=pad[i],
                                                   batch_norm=batch_normalize[i],
                                                   activation=activation[i]))

            prev_filters = filter_sizes[i]

            self.add_module("convolution_block_1_{}".format(i), self.convolution_blocks_1[-1])


        self.yololayer_1 = YOLOBlock(inp_dim_w=self.input_shape[1],
                                     inp_dim_h=self.input_shape[2],
                                     anchors=self.anchors,
                                     num_classes=self.num_classes,
                                     # cuda=self._cuda
                                     )
        # self.add_module("yololayer_1", self.yololayer_1)

        self.sparse_to_dense = scn.SparseToDense(dimension=3, nPlanes=self.num_classes)


    def forward(self, x):
        print(x)
        batch_size = x[-1]

        # Reshape this tensor into the right shape to apply this multiplane network.
        self.nplanes = 3
        # x = torch.chunk(x, chunks=self.nplanes, dim=1)
        print(x[0].shape)
        x = self.input_tensor(x)

        # print('initial', x[0].size())
        x = self.initial_convolution(x)
        # print('after initial_convolution', x[0].size())

        for i in range(0, self.n_core_blocks):
            x = self.dowsample[i](x)
            # print(i, 'after dowsample', x[0].size())
            x = self.residual[i](x)
            # print(i, 'after residual', x[0].size())

        for i in range(0, len(self.convolution_blocks_1)):
            x = self.convolution_blocks_1[i](x)
            # print(i, 'after convolution_blocks_1', x[0].size())
        x = self.sparse_to_dense(x)
        print(len(x))
        print(x[0].shape)
        x = self.yololayer_1(x)
        # print('after yolo_1', x[0].size())
        return x
