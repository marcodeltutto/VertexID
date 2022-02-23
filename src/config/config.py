from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import List, Any
from omegaconf import MISSING

from .network   import Yolo
from .mode      import Mode
from .data      import Data

class ComputeMode(Enum):
    CPU   = 0
    GPU   = 1

class Precision(Enum):
    float32  = 0
    mixed    = 1
    bfloat16 = 2
    float16  = 3

class DistributedMode(Enum):
    horovod = 0
    DDP     = 1


    # yolo_anchors:        int = [(116, 90), (156, 198), (373, 326)]




@dataclass
class Run:
    distributed:          bool        = True
    distributed_mode: DistributedMode = DistributedMode.horovod
    compute_mode:         ComputeMode = ComputeMode.GPU
    iterations:           int         = 500
    aux_iteration:        int         = 10
    minibatch_size:       int         = 2
    id:                   str         = MISSING
    precision:            Precision   = Precision.float32

cs = ConfigStore.instance()

cs.store(group="run", name="base_run", node=Run)

cs.store(
    name="disable_hydra_logging",
    group="hydra/job_logging",
    node={"version": 1, "disable_existing_loggers": False, "root": {"handlers": []}},
)


defaults = [
    {"run"       : "base_run"},
    {"mode"      : "train"},
    {"data"      : "SBND"},
    {"network"   : "yolo"}
]

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    #         "_self_",
    #         {"run" : Run()}
    #     ]
    # )

    run:        Run       = MISSING
    mode:       Mode      = MISSING
    data:       Any       = MISSING
    network:    Yolo      = MISSING
    output_dir: str       = "output/${framework.name}/${network.name}/${run.id}/"
    name:       str       = "none"

cs.store(name="base_config", node=Config)
