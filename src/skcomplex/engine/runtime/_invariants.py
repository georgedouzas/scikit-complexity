"""Check that observables stay constant during a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

import numpy as np

from ...errors import ConservationError, ValidationError
from ..problems import BaseProblemSpec, FrameSpec
from ._base import BaseObserver


class InvariantObserver(BaseObserver):
    """Observer that checks at every frame that every named observable stays within its tolerance of its first value.

    A simulation attaches one when it is given invariants.

    Args:
        invariants:
            The tolerance of every invariant, by the path of its observable.

    Attributes:
        initial_:
            The value of every invariant at the first frame.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.specs import ObservableSpec
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import InvariantObserver, Simulation
        >>> from skcomplex.models.biology.ecology import PopulationElement
        >>> system = System(
        ...     elements=[GroupElement('herd', element=PopulationElement, size=2, values={'n': [2.0, 6.0]})]
        ... )
        >>> total = ObservableSpec('total', group='herd', reduce='sum', variable='n')
        >>> monitor = InvariantObserver({'total': 1e-9})
        >>> _ = Simulation(system, dt=1, observables=[total], observers=[monitor]).check().prepare().run(3)
        >>> monitor.initial_
        {'total': 8.0}
    """

    def __init__(self, invariants: Mapping[str, float]) -> None:
        self.invariants = invariants

    def _check(self) -> None:
        """Raise an error unless every invariant names an observable with a nonnegative tolerance."""
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

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Forget the values of the invariants of an earlier run.

        Args:
            context:
                The compiled problem the run advances.
        """
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
