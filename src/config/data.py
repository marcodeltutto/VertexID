from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# @dataclass
class ImageModeKind(Enum):
    dense  = 0
    sparse = 1

@dataclass
class Data:
    downsample:           int = 1
    data_directory:       str = "/lus/grand/projects/neutrino_osc_ADSP/datasets/SBND/VertexID/"
    file:                 str = "sbnd_vertexID_train.h5"
    aux_file:             str = "sbnd_vertexID_val.h5"
    image_mode: ImageModeKind = ImageModeKind.dense
    input_dimension:      int = 2
    image_width:          int = 2048
    image_height:         int = 1280
    pitch:                float = 0.3
    padding_x:            int = 0
    padding_y:            int = 0

@dataclass
class SBND(Data):
    data_directory:       str = "/lus/grand/projects/neutrino_osc_ADSP/datasets/SBND/VertexID/"
    file:                 str = "sbnd_vertexID_train.h5"
    aux_file:             str = "sbnd_vertexID_val.h5"

@dataclass
class DUNE(Data):
  data_directory: str = "/lus/grand/projects/neutrino_osc_ADSP/datasets/DUNE/pixsim_full/"
  file:           str = "merged_sample_chunk8_99.h5"
  aux_file:       str = "merged_sample_chunk8_98.h5"
  image_width:    int = 1536
  image_height:   int = 1024
  pitch:        float = 0.4
  padding_x:      int = 286
  padding_y:      int = 124

cs = ConfigStore.instance()
cs.store(group="data", name="SBND", node=SBND)
cs.store(group="data", name="DUNE", node=DUNE)
