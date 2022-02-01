import torch
import torch.nn as nn
import numpy as np




class Block(nn.Module):
    '''
    A convolutional block
    '''

    def __init__(self, infilters, outfilters, kernel, stride, padding, batch_norm, activation):

        nn.Module.__init__(self)

        self.batch_norm = batch_norm

        if not batch_norm: bias = True
        else: bias = False

        if padding:
            pad = (kernel - 1) // 2
        else:
            pad = 0

        self.conv1 = torch.nn.Conv2d(
            in_channels  = infilters,
            out_channels = outfilters,
            kernel_size  = kernel,
            stride       = stride,
            padding      = padding,
            bias         = bias)

        if batch_norm:
            self.bn1  = torch.nn.BatchNorm2d(outfilters)

        if activation == 'leaky':
            self.activ = torch.nn.LeakyReLU()
        else:
            self.activ = torch.nn.ReLU()

    def forward(self, x):

        # print('Block x is ', x)
        out = self.conv1(x)

        if self.batch_norm:
            out = self.bn1(out)
        out = self.activ(out)

        return out


class ResidualBlock(nn.Module):
    '''
    A residual block, with two convolutions
    '''

    def __init__(self, infilters, outfilters1, outfilters2, padding=1, batch_norm=False, activation='leaky'):
        nn.Module.__init__(self)

        self.batch_norm = batch_norm


        conv1_kernel = 1
        if padding:
            pad = (conv1_kernel - 1) // 2
        else:
            pad = 0

        self.conv1 = torch.nn.Conv2d(
            in_channels  = infilters,
            out_channels = outfilters1,
            kernel_size  = conv1_kernel,
            stride       = 1,
            padding      = pad,
            bias         = False)

        if batch_norm:
            self.bn1  = torch.nn.BatchNorm2d(outfilters1)


        conv2_kernel = 3
        if padding:
            pad = (conv2_kernel - 1) // 2
        else:
            pad = 0

        self.conv2 = torch.nn.Conv2d(
            in_channels  = outfilters1,
            out_channels = outfilters2,
            kernel_size  = conv2_kernel,
            stride       = 1,
            padding      = pad,
            bias         = False)

        if batch_norm:
            self.bn2  = torch.nn.BatchNorm2d(outfilters2)

        if activation == 'leaky':
            self.activ = torch.nn.LeakyReLU()
        else:
            self.activ = torch.nn.ReLU()

    def forward(self, x):

        residual = x

        out = self.conv1(x)

        if self.batch_norm:
            out = self.bn1(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)

        # out = self.activ(out + residual)
        out = self.activ(out) + self.activ(residual)
        return out


class ResidualBlockSeries(torch.nn.Module):
    '''
    A series of residual blocks
    '''

    def __init__(self, n_blocks, infilters, outfilters1, outfilters2, padding=1, batch_norm=False, activation='leaky'):
        torch.nn.Module.__init__(self)

        self.blocks = torch.nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(
                ResidualBlock(
                    infilters,
                    outfilters1,
                    outfilters2,
                    padding,
                    batch_norm,
                    activation
                )
            )


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x


class RouteBlock(nn.Module):
    '''
    A route block, currenty not used
    '''

    def __init__(self, start, end=None):

        nn.Module.__init__(self)

        self._start = start
        self._end = end

    def forward(self, x, outputs):

        if end is None:
            out = outputs[i + start]
        else:
            map1 = outputs[i + start]
            map2 = outputs[i + end]
            out = torch.cat((map1, map2), 1)

        return out



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

    def __init__(self, input_shape, anchors, args):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the args module

        # self.nplanes    = args.nplanes
        # self.label_mode = args.label_mode

        print(args)

        self.input_shape = input_shape
        # self.anchors = args.yolo_anchors
        self.anchors = anchors
        self.num_classes = args.yolo_num_classes


        prev_filters = 1 #3
        n_filters = args.n_initial_filters

        #
        # First convolutional block
        #
        self.initial_convolution = Block(infilters=prev_filters,
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
        self.blocks_multiplicity = [1, 2, 8, 8, 4]

        # This is the number of downsampling/res blocks
        # that we have in the network, 5 in the case of
        # original yolo
        # self.n_core_blocks = args.n_core_blocks

        self.dowsample = []
        self.residual = []

        #
        # Downsampling and residual blocks
        #
        self.downsample = torch.nn.ModuleList()
        self.residual   = torch.nn.ModuleList()

        # for i in range(0, self.n_core_blocks): # seems like this should loop over self.blocks_multiplicity
        for i in range(len(self.blocks_multiplicity)):

            # Downlsampling block
            self.downsample.append(
                Block(
                    infilters  = prev_filters,
                    outfilters = n_filters,
                    kernel     = args.kernel_size,
                    stride     = 2,
                    padding    = 1,
                    batch_norm = args.batch_norm,
                    activation = 'leaky'
                )
            )

            # Residual block series
            self.residual.append(
                ResidualBlockSeries(
                    # n           = i, # This argument didn't do anything ...
                    n_blocks    = self.blocks_multiplicity[i],
                    infilters   = n_filters,
                    outfilters1 = prev_filters,
                    outfilters2 = n_filters))

            # self.add_module("resblock_{}".format(i), self.residual[-1])

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
        batch_normalize = [args.batch_norm] * 1
        pad = [1] * 1
        stride = [1] * 1
        filter_sizes = [5]
        kernel_size = [3]
        activation = ['leaky']

        self.convolution_blocks_1 = torch.nn.ModuleList()

        for i in range(0, len(filter_sizes)):

            self.convolution_blocks_1.append(Block(infilters=prev_filters,
                                                   outfilters=filter_sizes[i],
                                                   kernel=kernel_size[i],
                                                   stride=stride[i],
                                                   padding=pad[i],
                                                   batch_norm=batch_normalize[i],
                                                   activation=activation[i]))

            prev_filters = filter_sizes[i]

            # self.add_module("convolution_block_1_{}".format(i), self.convolution_blocks_1[-1])


        self.yololayer_1 = YOLOBlock(inp_dim_w=self.input_shape[1],
                                     inp_dim_h=self.input_shape[2],
                                     anchors=self.anchors,
                                     num_classes=self.num_classes,
                                     # cuda=self._cuda
                                     )
        # self.add_module("yololayer_1", self.yololayer_1)

    def forward(self, x):

        batch_size = x.shape[0]

        # Reshape this tensor into the right shape to apply this multiplane network.
        self.nplanes = 3
        x = torch.chunk(x, chunks=self.nplanes, dim=1)

        # print('initial', x[0].size())
        x = tuple(self.initial_convolution(_x) for _x in x)
        # print('after initial_convolution', x[0].size())

        for i in range(len(self.blocks_multiplicity)):
            x = tuple(self.downsample[i](_x) for _x in x)
            # print(i, 'after dowsample', x[0].size())
            x = tuple(self.residual[i](_x) for _x in x)
            # print(i, 'after residual', x[0].size())

        for i in range(0, len(self.convolution_blocks_1)):
            x = tuple(self.convolution_blocks_1[i](_x) for _x in x)
            # print(i, 'after convolution_blocks_1', x[0].size())


        x = tuple(self.yololayer_1(_x) for _x in x)
        # print('after yolo_1', x[0].size())

        return x
