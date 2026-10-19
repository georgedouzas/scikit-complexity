"""Run systems frame by frame with clocks, live parameter changes, observers, trajectories, and a model catalog."""

from ._catalog import MODELS, ModelEntry, catalog, create, describe, register_model, run
from ._clocks import AsFastAsPossible, Clock, RealTime
from ._frames import Frame
from ._observers import Observer, Recorder
from ._simulations import Simulation
from ._trajectories import Trajectory

__all__: list[str] = [
    'MODELS',
    'AsFastAsPossible',
    'Clock',
    'Frame',
    'ModelEntry',
    'Observer',
    'RealTime',
    'Recorder',
    'Simulation',
    'Trajectory',
    'catalog',
    'create',
    'describe',
    'register_model',
    'run',
]
