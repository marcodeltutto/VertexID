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
    data_directory:       str = "/grand/neutrino_osc_ADSP/users/deltutto/vertex_id/devel_vtxid/"
    file:                 str = "sbnd-numuCC-CosmicTaggerGenNeutrino-20220203T174758.art_larcv_proc.h5"
    aux_file:             str = "sbnd-numuCC-CosmicTaggerGenNeutrino-20220203T174758.art_larcv_proc.h5"
    image_mode: ImageModeKind = ImageModeKind.dense
    input_dimension:      int = 2
    image_width:          int = 2048
    image_height:         int = 1280
    pitch:                float = 0.3
    padding_x:            int = 0
    padding_y:            int = 0

cs = ConfigStore.instance()
cs.store(group="data", name="data", node=Data)
