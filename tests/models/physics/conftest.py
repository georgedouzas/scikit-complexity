"""Share the run helpers and the series checks of the physics model tests."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.engine.solvers import BaseSolver, ScipyODESolver
from skcomplex.models import describe
from skcomplex.specs import ObservableSpec

ACCURATE = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12)


def _run(
    system: System,
    until: float,
    dt: float,
    solver: BaseSolver = ACCURATE,
    *,
    observables: Sequence[ObservableSpec] = (),
    schedule: Mapping[str, Callable[[float], float]] | None = None,
) -> Trajectory:
    """Return the trajectory of a system run to a time."""
    simulation = Simulation(system, dt=dt, solver=solver, observables=observables, schedule=schedule)
    return simulation.check().prepare().run(until=until).trajectory_


def _series(trajectory: Trajectory, *names: str) -> list[npt.NDArray[np.float64]]:
    """Return the series of several variables as arrays."""
    return [np.asarray(trajectory.values(name), dtype=float) for name in names]


def _upward_crossings(times: npt.NDArray[np.float64], values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the interpolated times where a series crosses zero upwards."""
    index = np.flatnonzero((values[:-1] < 0) & (values[1:] >= 0))
    fraction = -values[index] / (values[index + 1] - values[index])
    return times[index] + fraction * (times[index + 1] - times[index])


def _relative_spread(values: npt.NDArray[np.float64]) -> float:
    """Return the range of a conserved quantity relative to its size."""
    return float(np.ptp(values) / max(np.abs(values).max(), 1e-12))


def _run_observed(name: str, system: System, until: float, dt: float) -> Trajectory:
    """Return the trajectory of a model's run with its observables."""
    observables = describe(name).observables_of(system)
    return (
        Simulation(system, dt=dt, random_state=0, observables=observables)
        .check()
        .prepare()
        .run(until=until)
        .trajectory_
    )
