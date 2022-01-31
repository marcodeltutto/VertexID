from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

class OptimizerKind(Enum):
    Adam = 0
    SGD  = 1

class LRSchedule(Enum):
    flat          = 0
    one_cycle     = 1
    traingle_clr  = 2
    exp_range_clr = 3
    decay         = 4
    expincrease   = 5

@dataclass
class Optimizer:
    learning_rate:       float = 0.0003
    name:        OptimizerKind = OptimizerKind.Adam
    gradient_accumulation: int = 1
    lambda_noobj:        float = 0.5
    lambda_coord:        float = 0.5
    lr_schedule:    LRSchedule = LRSchedule.flat
    weight_decay:        float = 0.0


class ModeKind(Enum):
    train     = 0
    iotest    = 1
    inference = 2

@dataclass
class Mode:
    name:          ModeKind = ModeKind.train
    no_summary_images: bool = False
    weights_location:  str  = ""

@dataclass
class Train(Mode):
    checkpoint_iteration: int = 500
    summary_iteration:    int = 1
    logging_iteration:    int = 1
    optimizer:      Optimizer = Optimizer()

@dataclass
class Inference(Mode):
    name:         ModeKind  = ModeKind.inference
    start_index:       int  = 0
    summary_iteration: int  = 1
    logging_iteration: int  = 1

@dataclass
class IOTest(Mode):
    name:   ModeKind = ModeKind.iotest
    start_index: int = 0



cs = ConfigStore.instance()
cs.store(name="optimizer", node=Optimizer)


cs = ConfigStore.instance()
cs.store(group="mode", name="train",     node=Train)
cs.store(group="mode", name="inference", node=Inference)
cs.store(group="mode", name="iotest",    node=IOTest)
