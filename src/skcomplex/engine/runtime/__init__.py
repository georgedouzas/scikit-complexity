"""Run systems frame by frame with clocks, live parameter changes, observers, trajectories, and a model catalog."""

from ._catalog import MODELS, ModelEntry, catalog, create, describe, register_model, run, summarize
from ._clocks import AsFastAsPossibleClock, BaseClock, RealTimeClock
from ._conservation import ConservationObserver, InvariantObserver
from ._frames import Frame
from ._observers import BaseObserver, RecorderObserver
from ._simulations import Simulation
from ._trajectories import Trajectory

__all__: list[str] = [
    'MODELS',
    'AsFastAsPossibleClock',
    'BaseClock',
    'BaseObserver',
    'ConservationObserver',
    'Frame',
    'InvariantObserver',
    'ModelEntry',
    'RealTimeClock',
    'RecorderObserver',
    'Simulation',
    'Trajectory',
    'catalog',
    'create',
    'describe',
    'register_model',
    'run',
    'summarize',
]
