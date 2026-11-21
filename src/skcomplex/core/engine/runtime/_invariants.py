"""Check that observables stay constant during a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

import numpy as np

from ...errors import ConservationError, ValidationError
from ..problems import BaseProblemSpec, FrameSpec
from ._base import BaseObserver


class InvariantObserver(BaseObserver):
    """An observer that checks that each named observable stays near its first value.

    At every frame each observable must stay within its tolerance of its value at the first frame, such as the energy of
    a star within `1e-6` of its starting energy, and the observer raises an error that names the observable and the
    frame otherwise. A run adds one when `simulate` is given `invariants`. It checks the energy of a pendulum, the total
    of a closed population, and the probability of a quantum state.

    Read more in the user guide, under Observe.

    Args:
        invariants:
            The tolerance of every invariant, by the path of its observable.

    Attributes:
        initial_:
            The value of every invariant at the first frame.

    See Also:
        ConservationObserver: Checks that conserved quantities hold.

    Examples:
        >>> from skcomplex.core.engine.runtime import InvariantObserver
        >>> from skcomplex.systems.physics.nonlinear_oscillators import HenonHeilesSystem
        >>> observer = InvariantObserver({'star.energy': 1e-6})
        >>> _ = HenonHeilesSystem().prepare().simulate(until=1, observers=[observer])
        >>> round(observer.initial_['star.energy'], 4)
        0.0659
    """

    def __init__(self, invariants: Mapping[str, float]) -> None:
        self.invariants = invariants

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Forget the values of the invariants of an earlier run.

        It first raises an error for an invariant without observable or with a negative tolerance. Every invariant must
        name an observable with a nonnegative tolerance.

        Args:
            context:
                The compiled problem the run advances.
        """
        if not isinstance(self.invariants, Mapping):
            error_msg = f'invariants = {self.invariants!r} must map the path of every observable to its tolerance.'
            raise ValidationError(error_msg)
        for path, tolerance in self.invariants.items():
            if not isinstance(path, str) or isinstance(tolerance, bool) or not isinstance(tolerance, int | float):
                error_msg = f'The invariant {path!r} needs a string path and a numeric tolerance, not {tolerance!r}.'
                raise ValidationError(error_msg)
            if not tolerance >= 0:
                error_msg = f'The tolerance {tolerance!r} of the invariant {path} must be nonnegative.'
                raise ValidationError(error_msg)
        del context
        self.initial_: dict[str, float] = {}

    def _run(self, frame: FrameSpec | None = None) -> None:
        """Check the invariants of a frame against the first frame.

        Args:
            frame:
                The recorded frame, or `None` at the last run, which checks nothing.

        Raises:
            ValidationError:
                An invariant names no observable of the run.

            ConservationError:
                An invariant changed by more than its tolerance.
        """
        if frame is None:
            return
        for path, tolerance in self.invariants.items():
            if path not in frame.observables:
                error_msg = (
                    f'The invariant {path} names no observable of the run. Pass an observable labelled {path!r} '
                    'in `observables`.'
                )
                raise ValidationError(error_msg)
            value = float(np.sum(np.asarray(frame.observables[path], dtype=float)))
            initial = self.initial_.setdefault(path, value)
            if abs(value - initial) > tolerance:
                error_msg = (
                    f'The invariant {path} changed from {initial:.6g} to {value:.6g} at frame {frame.step}, time '
                    f'{frame.time:.6g}, beyond its tolerance {tolerance:g}.'
                )
                raise ConservationError(error_msg)
