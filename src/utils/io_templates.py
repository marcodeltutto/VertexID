from . import larcv_io



# Here, we set up a bunch of template IO formats in the form of callable functions:

def train_io(input_file, image_dim, label_mode, prepend_names="", compression=0):
    if image_dim == 2:
        max_voxels = 20000
        data_proc = gen_sparse2d_data_filler(name=prepend_names + "data", producer="\"dunevoxels\"", max_voxels=max_voxels)
    else:
        max_voxels = 16000
        data_proc = gen_sparse3d_data_filler(name=prepend_names + "data", producer="\"dunevoxels\"", max_voxels=max_voxels)

    vertex_proc = gen_vertex_filler(name=prepend_names, producer="\"neutrino\"")

    label_proc = gen_label_filler(label_mode, prepend_names)


    config = larcv_io.ThreadIOConfig(name="TrainIO")

    if compression != 0:
        print('compression is', compression)
        image_compression = gen_compression(
            name="Downsample_image", input_label="\"dunevoxels\"",
            compression_level = compression, pooling_type = "average")
        config.add_process(image_compression)

    config.add_process(data_proc)
    config.add_process(vertex_proc)
    for l in label_proc:
        config.add_process(l)

    config.set_param("InputFiles", input_file)

    return config


def test_io(input_file, image_dim, label_mode, prepend_names="aux_"):
    if image_dim == 2:
        max_voxels = 20000
        data_proc = gen_sparse2d_data_filler(name=prepend_names + "data", producer="\"dunevoxels\"", max_voxels=max_voxels)
    else:
        max_voxels = 16000
        data_proc = gen_sparse3d_data_filler(name=prepend_names + "data", producer="\"dunevoxels\"", max_voxels=max_voxels)

    label_proc = gen_label_filler(label_mode, prepend_names)


    config = larcv_io.ThreadIOConfig(name="TestIO")

    config.add_process(data_proc)
    for l in label_proc:
        config.add_process(l)

    config.set_param("InputFiles", input_file)

    return config


def ana_io(input_file, image_dim, label_mode, prepend_names=""):
    if image_dim == 2:
        max_voxels = 20000
        data_proc = gen_sparse2d_data_filler(name=prepend_names + "data", producer="\"dunevoxels\"", max_voxels=max_voxels)
    else:
        max_voxels = 16000
        data_proc = gen_sparse3d_data_filler(name=prepend_names + "data", producer="\"dunevoxels\"", max_voxels=max_voxels)


    label_proc = gen_label_filler(label_mode, prepend_names)


    config = larcv_io.ThreadIOConfig(name="AnaIO")
    # Force ana files to go in order:

    config._params['RandomAccess'] = "0"
    config.add_process(data_proc)
    for l in label_proc:
        config.add_process(l)

    config.set_param("InputFiles", input_file)

    return config

def output_io(input_file, output_file):




    config = larcv_io.IOManagerConfig(name="IOManager")
    # Force ana files to go in order:

    config._params['RandomAccess'] = "0"

    config.set_param("InputFiles", input_file)
    config.set_param("OutputFile", output_file)

    # These lines slim down the output file.
    # Without them, 25 output events is 2.8M and takes 38s
    # With the, 25 output events is 119K and takes 36s
    # config.set_param("ReadOnlyType", "[\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"cluster2d\",\"cluster2d\",\"cluster3d\",\"cluster3d\"]")  
    # config.set_param("ReadOnlyName", "[\"neutrino\",\"cpiID\",\"neutID\",\"npiID\",\"protID\",\"all\",\"duneseg\",\"duneseg\",\"segment\",\"duneseg\",\"segment\"]")  

    config.set_param("ReadOnlyType", "[\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"cluster2d\"]")  
    config.set_param("ReadOnlyName", "[\"neutrino\",\"cpiID\",\"neutID\",\"npiID\",\"protID\",\"all\",\"segment\",\"duneseg\"]")  
    
    # config.set_param("ReadOnlyType", "[\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\"]")  
    # config.set_param("ReadOnlyName", "[\"neutrino\",\"cpiID\",\"neutID\",\"npiID\",\"protID\",\"all\",\"segment\"]")  

    return config


def gen_sparse2d_data_filler(name, producer, max_voxels):

    proc = larcv_io.ProcessConfig(proc_name=name, proc_type="BatchFillerSparseTensor2D")

    proc.set_param("Verbosity",         "3")
    proc.set_param("TensorProducer",  producer)
    proc.set_param("IncludeValues",     "true")
    proc.set_param("MaxVoxels",         max_voxels)
    proc.set_param("Channels",          "[0,1,2]")
    proc.set_param("UnfilledVoxelValue","-999")
    proc.set_param("Augment",           "false")

    return proc


def gen_sparse3d_data_filler(name, producer, max_voxels):

    proc = larcv_io.ProcessConfig(proc_name=name, proc_type="BatchFillerSparseTensor3D")

    proc.set_param("Verbosity",         "3")
    proc.set_param("TensorProducer",  producer)
    proc.set_param("IncludeValues",     "true")
    proc.set_param("MaxVoxels",         max_voxels)
    proc.set_param("UnfilledVoxelValue","-999")
    proc.set_param("Augment",           "true")

    return proc

def gen_compression(name, input_label, compression_level, pooling_type):
    proc = larcv_io.ProcessConfig(proc_name=name, proc_type="Downsample")

    proc.set_param("Producer",       f"{input_label}")
    proc.set_param("Product",         "\"sparse2d\"")
    proc.set_param("OutputProducer", f"{input_label}")
    proc.set_param("Downsample",       2**compression_level)
    if pooling_type == "max":
        proc.set_param("PoolType", 2)
    elif pooling_type == "average":
        proc.set_param("PoolType", 1)

    return proc

def gen_vertex_filler(name, producer="\"neutrino\""):

    proc = larcv_io.ProcessConfig(proc_name=name + "vertex", proc_type="BatchFillerVertex")

    proc.set_param("Verbosity",         "2")
    proc.set_param("ParticleProducer",  producer)
    proc.set_param("PdgClassList",      "[{}]".format(",".join([str(i) for i in range(36)])))

    return proc


def gen_label_filler(label_mode, prepend_names):

    if label_mode == 'all':

        proc = larcv_io.ProcessConfig(proc_name=prepend_names + "label", proc_type="BatchFillerPIDLabel")

        proc.set_param("Verbosity",         "3")
        proc.set_param("ParticleProducer",  "all")
        proc.set_param("PdgClassList",      "[{}]".format(",".join([str(i) for i in range(36)])))

        return [proc]

    else:
        procs = []
        for name, l in zip(['neut', 'prot', 'cpi', 'npi'], [3, 3, 2, 2]): 
            proc  = larcv_io.ProcessConfig(proc_name=prepend_names + "label_" + name, proc_type="BatchFillerPIDLabel")

            proc.set_param("Verbosity",         "3")
            proc.set_param("ParticleProducer",  name+"ID")
            proc.set_param("PdgClassList",      "[{}]".format(",".join([str(i) for i in range(l)])))

            procs.append(proc)
        return procs






