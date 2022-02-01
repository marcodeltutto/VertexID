from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING





@dataclass
class Yolo:
    name:                str  = "yolo"
    bias:                bool = True
    batch_norm:          bool = True
    n_initial_filters:    int = 16
    blocks_per_layer:     int = 2
    blocks_deepest_layer: int = 5
    blocks_final:         int = 5
    network_depth:        int = 6
    residual:            bool = True
    yolo_num_classes:     int = 1
    # yolo_anchors:        int = [(116, 90), (156, 198), (373, 326)]



cs = ConfigStore.instance()
# cs.store(name="network", node=Network)
cs.store(group="network", name="yolo", node=Yolo)
