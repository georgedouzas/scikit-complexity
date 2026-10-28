"""Run systems frame by frame with clocks, live parameter changes, observers, trajectories, and a model catalog."""

from ._as_fast_as_possible import AsFastAsPossibleClock
from ._base import BaseClock, BaseObserver
from ._catalog import MODELS, ModelEntry, catalog, create, describe, register_model, run, summarize
from ._conservation import ConservationObserver
from ._frames import Frame
from ._invariants import InvariantObserver
from ._real_time import RealTimeClock
from ._recorder import RecorderObserver
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
