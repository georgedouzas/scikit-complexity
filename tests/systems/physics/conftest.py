"""Share the run helpers and the series checks of the physics system tests."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from skcomplex.core.components.system import BaseSystem
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.engine.solvers import BaseSolver, ScipyODESolver
from skcomplex.core.specs import ObservableSpec

ACCURATE = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12)


def _run_default(
    system: BaseSystem,
    until: float,
    dt: float,
    solver: BaseSolver | None = None,
    observables: Sequence[ObservableSpec] = (),
) -> Trajectory:
    """Return the trajectory of the parts of a system under the default solver of their laws, unless one is given."""
    composite = system.prepare().system_
    return simulate(composite, until=until, dt=dt, solver=solver, observables=observables, random_state=0)


def _run(
    system: BaseSystem,
    until: float,
    dt: float,
    solver: BaseSolver = ACCURATE,
    *,
    observables: Sequence[ObservableSpec] = (),
    schedule: Mapping[str, Callable[[float], float]] | None = None,
) -> Trajectory:
    """Return the trajectory of a system run to a time, or of its parts under a schedule of their parameters."""
    if schedule is None:
        return simulate(system, until=until, dt=dt, solver=solver, observables=observables)
    composite = system.prepare().system_
    return simulate(composite, until=until, dt=dt, solver=solver, observables=observables, schedule=schedule)


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


def _run_observed(system: BaseSystem, until: float, dt: float) -> Trajectory:
    """Return the trajectory of a system's run with its tuned observables."""
    return simulate(system, until=until, dt=dt, random_state=0)
