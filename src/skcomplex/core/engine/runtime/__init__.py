"""Run systems frame by frame with clocks, observers, and trajectories.

A run hands every frame to its observers, such as a recorder or a check that money is conserved, waits between frames as
its clock says, and returns a trajectory of the values over time, such as the size of a population at times 0, 1, and 2.
"""

from ._as_fast_as_possible import AsFastAsPossibleClock
from ._base import BaseClock, BaseObserver
from ._conservation import ConservationObserver
from ._invariants import InvariantObserver
from ._real_time import RealTimeClock
from ._recorder import RecorderObserver
from ._simulate import compile_problem, run_problem
from ._trajectories import Trajectory

__all__: list[str] = [
    'AsFastAsPossibleClock',
    'BaseClock',
    'BaseObserver',
    'ConservationObserver',
    'InvariantObserver',
    'RealTimeClock',
    'RecorderObserver',
    'Trajectory',
    'compile_problem',
    'run_problem',
]
