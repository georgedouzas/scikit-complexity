"""Advance fields whose fluxes are linear drifts by shifting each slice along its characteristics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
import scipy.fft
from scipy.ndimage import map_coordinates

from ...core.laws import FluxLaw, RateLaw
from ...core.spaces import ContinuousTime, LatticeSpace, PhaseSpace
from ...errors import SolverError
from ...specs import SupportSpec
from ...types import Array, FormKind, OrderKind
from ..problems import ProblemSpec
from ..problems._finite_volume_kernel import FiniteVolumeKernel
from ..problems._steps import step_count
from ._solver import Solver

SPLINE_ORDER = 3
SPLINE_MODES = {'zero-flux': 'nearest', 'fixed': 'constant'}


class SemiLagrangianSolver(Solver):
    """Directional splitting of Cheng and Knorr, which shifts every slice of a field along each axis in turn.

    Every flux of the field must be a linear drift `a u`, with `a` free of the state and constant along its own axis. A
    shift along a periodic axis multiplies the Fourier modes of the slices through `scipy.fft`, which conserves the
    total, and a shift along an interval evaluates a cubic spline through `scipy.ndimage.map_coordinates`. The solver is
    stable at any Courant number. Rate laws advance by one midpoint step between two half transport steps.

    Args:
        dt:
            The largest internal time step.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.elements import DensityElement, FieldElement
        >>> from skcomplex.core.interactions import AdvectionInteraction
        >>> from skcomplex.core.laws.utils import coordinate
        >>> from skcomplex.core.spaces import LatticeSpace, PhaseSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.engine.solvers import SemiLagrangianSolver
        >>> from skcomplex.specs import AxisSpec
        >>> space = PhaseSpace(LatticeSpace((16,)), [AxisSpec('v', 8, -1.0, 1.0)]).prepare()
        >>> start = np.outer(2 + np.sin(2 * np.pi * np.arange(16) / 16), np.ones(8))
        >>> gas = FieldElement(
        ...     'gas', element=DensityElement, space=space, values={'f': start}, discretization='finite-volume'
        ... )
        >>> stream = AdvectionInteraction('s', target='gas', state='f', drift=lambda s: [coordinate('v'), 0])
        >>> system = System(elements=[gas], interactions=[stream])
        >>> solver = SemiLagrangianSolver(dt=1.0)
        >>> final = Simulation(system, solver=solver).prepare().run(until=2.0).trajectory_.values('gas.f')[-1]
        >>> abs(float(final.sum() / start.sum()) - 1.0) < 1e-12
        True
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={PhaseSpace, LatticeSpace},
            laws={FluxLaw, RateLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.ODE

    @classmethod
    def _is_implicit(cls) -> bool:
        """Return whether explicit stability limits do not apply, as for shifts along characteristics."""
        return True

    def __init__(
        self,
        dt: float = 0.01,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.dt = dt
        self.random_state = random_state
        self.until = until

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the solver, raising a named error for a flux that is not a linear drift.

        Args:
            context:
                The compiled problem.

        Raises:
            SolverError:
                The problem has no flux, or a flux is not a linear drift.
        """
        super()._prepare(context)
        problem = self.problem_
        if not problem.volumes:
            error_msg = 'SemiLagrangianSolver shifts along the drifts of flux laws, but the problem has none.'
            raise SolverError(error_msg)
        for volume in problem.volumes:
            if not volume.drifts:
                error_msg = (
                    f'SemiLagrangianSolver shifts along linear drifts, but the flux of {volume.owner} is not a linear '
                    'drift, with a coefficient free of the state and constant along its own axis. Use SSPRKSolver.'
                )
                raise SolverError(error_msg)

    def _shifted_periodic(self, values: Array, displacement: Array, axis: int, spacing: float) -> Array:
        """Return slices of an axis shifted by a displacement per slice through the Fourier shift theorem."""
        length = values.shape[axis]
        waves = 2 * np.pi * np.fft.rfftfreq(length, d=spacing)
        shape = [1] * values.ndim
        shape[axis] = -1
        slices = np.take(displacement, [0], axis=axis)
        modes = np.exp(-1j * waves.reshape(shape) * slices)
        if length % 2 == 0:
            modes = np.where(waves.reshape(shape) == waves[-1], np.cos(waves[-1] * slices), modes)
        return np.asarray(scipy.fft.irfft(scipy.fft.rfft(values, axis=axis) * modes, n=length, axis=axis))

    def _shifted_spline(self, values: Array, displacement: Array, axis: int, *, spacing: float, kind: str) -> Array:
        """Return slices of an axis shifted by a displacement per slice through a cubic spline, slice by slice."""
        moved = np.moveaxis(values, axis, -1)
        steps = np.moveaxis(displacement, axis, -1)[..., 0] / spacing
        sites = np.arange(moved.shape[-1], dtype=float)
        result = np.empty(moved.shape)
        for index in np.ndindex(moved.shape[:-1]):
            result[index] = map_coordinates(
                moved[index],
                [sites - steps[index]],
                order=SPLINE_ORDER,
                mode=SPLINE_MODES[kind],
            )
        return np.moveaxis(result, -1, axis)

    def _shifted_state(self, volume: FiniteVolumeKernel, x: Array, p: Array, *, t: float, h: float, axis: int) -> Array:
        """Return the state vector after the states of one field shift along one axis for a time."""
        result = x.copy()
        states = volume._states(x)
        drifts = volume.drift_of(axis, states, x, p, t)
        spacing = volume.spacings[axis]
        for start, values, drift, (given, _) in zip(volume.starts, states, drifts, volume.boundaries, strict=True):
            kind = volume._kind(axis, given)
            displacement = np.broadcast_to(drift, values.shape) * h
            if not np.any(displacement):
                continue
            if kind == 'periodic':
                shifted = self._shifted_periodic(values, displacement, axis, spacing)
            else:
                shifted = self._shifted_spline(values, displacement, axis, spacing=spacing, kind=kind)
            result[start : start + volume.size] = shifted.ravel()
        return result

    def _sweep(self, x: Array, p: Array, t: float, h: float) -> Array:
        """Return the state after a symmetric sweep over all the axes of all the fields for a time."""
        state = x
        for volume in self.problem_.volumes:
            axes = range(len(volume.dimensions))
            *outer, inner = axes
            for axis in outer:
                state = self._shifted_state(volume, state, p, t=t, h=h / 2, axis=axis)
            state = self._shifted_state(volume, state, p, t=t, h=h, axis=inner)
            for axis in reversed(outer):
                state = self._shifted_state(volume, state, p, t=t + h / 2, h=h / 2, axis=axis)
        return state

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Advance from time `t` to time `until` in equal steps no longer than `dt`.

        Args:
            t:
                The current time.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The time to advance to.

        Returns:
            state:
                The state vector at time `until`.
        """
        steps = step_count(t, until, self.dt)
        h = (until - t) / steps
        sources = self.problem_.sources
        state = x
        for index in range(steps):
            time = t + index * h
            if sources is None:
                state = self._sweep(state, p, time, h)
                continue
            state = self._sweep(state, p, time, h / 2)
            middle = state + h / 2 * np.asarray(sources(time + h / 2, state, p), dtype=float)
            state = state + h * np.asarray(sources(time + h / 2, middle, p), dtype=float)
            state = self._sweep(state, p, time + h / 2, h / 2)
        return state
