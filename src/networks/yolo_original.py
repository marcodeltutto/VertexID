import torch
import torch.nn as nn
import numpy as np

from . network_config import network_config, str2bool

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = False):
    '''
    This function takes a detection feature map 
    and turns it into a 2-D tensor, 
    where each row of the tensor corresponds 
    to attributes of a bounding box.
    '''

    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction


class YOLOFlags(network_config):
    '''
    This class contains all the
    flags needed for the network
    '''

    def __init__(self):
        network_config.__init__(self)
        self._name = "yolo"
        self._help = "yolov3 network"

    def build_parser(self, network_parser):
        # this_parser = network_parser
        this_parser = network_parser.add_parser(self._name, help=self._help)

        this_parser.add_argument("--yolo-config",
            type    = str,
            default = 'cfg/yolov3.cfg',
            help    = "The yolo v3 configuration file.")
        this_parser.add_argument("--yolo-weights",
            type    = str,
            default = 'yolov3.weights',
            help    = "The yolo v3 configuration file.")

       


class EmptyLayer(nn.Module):
    '''
    This is just an empty layer,
    and is used to makes a 'shortcut'
    (residual layer). It is also used
    to construct route layers
    '''
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    '''
    This is the detection layer,
    it stores the anchors used for the 
    YOLO layer
    '''
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    '''
    This function reads blocks that have been parsed from
    the YOLO configuration file, and creates apposite 
    pytorch modules
    '''
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3 # Initial number of filters
    output_filters = []

    #
    # Loop over the blocks, and construct the modules
    #
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        
        #
        # Convolutional layer
        #
        if (x["type"] == "convolutional"):
            
            activation  = x["activation"]
            filters     = int(x["filters"])
            padding     = int(x["pad"])
            kernel_size = int(x["size"])
            stride      = int(x["stride"])

            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
        
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
        
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
        
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
        
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
        
        #
        # Upsampling layer
        #
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
        
        #     
        # Route layer
        #
        elif (x["type"] == "route"):

            # layers=-1, -4 means that two layers 
            # will be concatenated, with relative indexes -1 and -4
            # The concatenation is done in the forward pass

            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        
        #
        # shortcut corresponds to skip connection
        #
        elif x["type"] == "shortcut":
            # The res block is implemented
            # in the forward pass
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
        
        #
        # Yolo detection layer
        #
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
        
            # Save a DetectionLayer which remembers 
            # about the anchors
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
                              
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net_info, module_list)



class YOLO(nn.Module):
    '''
    Implentation of the YOLO
    CNN in pytorch
    '''
    def __init__(self, output_shape, args):
        super(YOLO, self).__init__()
        
        # Parse the configuration file
        print('args.yolo_config', args.yolo_config)
        self._blocks = self.parse_cfg(args.yolo_config)

        if args.compute_mode == "CPU": self._cuda = False
        else: self._cuda = True

        self.args = args
        
        # Construct all the modules
        self._net_info, self._module_list = create_modules(self._blocks)


    def parse_cfg(self, cfgfile):
        """
        Takes a configuration file
        
        Returns a list of blocks. Each blocks describes a block in the neural
        network to be built. Block is represented as a dictionary in the list
        
        """
        
        file = open(cfgfile, 'r')
        lines = file.read().split('\n')                        # store the lines in a list
        lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
        lines = [x for x in lines if x[0] != '#']              # get rid of comments
        lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
        
        block = {}
        blocks = []
        
        for line in lines:
            if line[0] == "[":               # This marks the start of a new block
                if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                    blocks.append(block)     # add it the blocks list
                    block = {}               # re-init the block
                block["type"] = line[1:-1].rstrip()     
            else:
                key,value = line.split("=") 
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)
        
        return blocks
        

    def forward(self, x):

        # Remove the first one as it contains
        # network metadata
        modules = self._blocks[1:]

        # We are going to store all the 
        # outputs here, as some of them
        # will be needed for res block
        # or for route layers
        outputs = {}

        # print (x.shape)
        
        write = 0
        for i, module in enumerate(modules):        
            module_type = (module["type"])

            # print (i, module_type, 'start', x.shape)

            #
            # Convolutional layer
            #
            if module_type == "convolutional" or module_type == "upsample":
                x = self._module_list[i](x)
            
            #
            # Route layer
            #
            elif module_type == "route":
                # Here we concatenate the outputs
                # of layers with indeces in module["layers"]
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                
            #
            # Shortcut layer
            #
            elif  module_type == "shortcut":
                # This is a residual block
                # Here we add the outputs of 
                # the previous layers and of 
                # 3 layers before.
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
    

            #
            # YOLO layer
            #
            elif module_type == 'yolo':        
                anchors     = self._module_list[i][0].anchors
                inp_dim     = int (self._net_info["height"])
                num_classes = int (module["classes"])
        
                # Transform the data
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, self._cuda)

                if not write:
                    detections = x
                    write = 1
                else:       
                    detections = torch.cat((detections, x), 1)
        
            # print (i, module_type, 'end', x.shape)
            outputs[i] = x
        
        return detections


    def load_weights(self, weightfile='yolov3.weights'):
        '''
        Opens the pretrained weights
        and loads them
        '''
        try:
            weightfile = self.args.yolo_weights
        except:
            print('Did you specify a weight file in the configuration?')
            exit()

        fp = open(weightfile, "rb")
    
        # The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
