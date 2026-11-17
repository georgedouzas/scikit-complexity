"""Share the run helpers and the series checks of the physics system tests."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from skcomplex.core.components import Observable
from skcomplex.core.components.systems import BaseSystem
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.engine.solvers import BaseSolver, ScipyODESolver
from skcomplex.core.systems import BaseSystem as PreparedSystem

ACCURATE = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12)


def _run_default(
    system: BaseSystem | PreparedSystem,
    until: float,
    dt: float,
    solver: BaseSolver | None = None,
    observables: Sequence[Observable] = (),
) -> Trajectory:
    """Return the trajectory of the components of a system under the default solver of their laws, or a given one."""
    if isinstance(system, PreparedSystem):
        components = system.prepare().system_.set_params(solver=solver or 'auto').prepare()
        return components.simulate(until=until, dt=dt, observables=observables, random_state=0)
    composite = system.prepare().system_
    return simulate(composite, until=until, dt=dt, solver=solver, observables=observables, random_state=0)


def _run(
    system: BaseSystem | PreparedSystem,
    until: float,
    dt: float,
    solver: BaseSolver = ACCURATE,
    *,
    observables: Sequence[Observable] = (),
    schedule: Mapping[str, Callable[[float], float]] | None = None,
) -> Trajectory:
    """Return the trajectory of a system run to a time, or of its components under a schedule of their parameters."""
    if isinstance(system, PreparedSystem):
        prepared = system.prepare()
        prepared.system_.set_params(solver=solver).prepare(prepared.observables)
        return prepared.simulate(until=until, dt=dt, observables=observables, schedule=schedule)
    if schedule is None:
        return simulate(system, until=until, dt=dt, solver=solver, observables=observables)
    composite = system.prepare().system_
    return simulate(composite, until=until, dt=dt, solver=solver, observables=observables, schedule=schedule)


def _series(trajectory: Trajectory, *names: str) -> list[npt.NDArray[np.float64]]:
    """Return the series of several variables as arrays."""
    return [np.asarray(trajectory.get_values(name), dtype=float) for name in names]


def _upward_crossings(times: npt.NDArray[np.float64], values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the interpolated times where a series crosses zero upwards."""
    index = np.flatnonzero((values[:-1] < 0) & (values[1:] >= 0))
    fraction = -values[index] / (values[index + 1] - values[index])
    return times[index] + fraction * (times[index + 1] - times[index])


def _relative_spread(values: npt.NDArray[np.float64]) -> float:
    """Return the range of a conserved quantity relative to its size."""
    return float(np.ptp(values) / max(np.abs(values).max(), 1e-12))


def _run_observed(system: BaseSystem | PreparedSystem, until: float, dt: float) -> Trajectory:
    """Return the trajectory of a system's run with its tuned observables."""
    if isinstance(system, PreparedSystem):
        return system.prepare().simulate(until=until, dt=dt, random_state=0)
    return simulate(system, until=until, dt=dt, random_state=0)
