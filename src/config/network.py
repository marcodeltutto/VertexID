from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING





@dataclass
class Yolo:
    name:                str  = "yolo"
    bias:                bool = True
    batch_norm:          bool = False
    n_initial_filters:    int = 32
    filter_size:          int = 5
    kernel_size:          int = 3
    # n_core_blocks:        int = 5
    residual:            bool = True
    yolo_num_classes:     int = 1
    # yolo_anchors:        int = [(116, 90), (156, 198), (373, 326)]



cs = ConfigStore.instance()
# cs.store(name="network", node=Network)
cs.store(group="network", name="yolo", node=Yolo)
