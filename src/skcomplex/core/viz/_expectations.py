"""Draw the local expectation values of quantum elements and groups as images and series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ..compile import CompiledAgents, CompiledElement, Problem, QuantumProblem
from ..components import view_of
from ..runtime import Frame
from ..space import LatticeSpace
from ._visuals import SPACE_TIME, ImageVisual, _log_scalars, register_visual


def _shape_of(element: CompiledElement) -> tuple[int, ...] | None:
    """Return the lattice shape of the local values of a quantum group or particle, or `None` for scalar values."""
    space = getattr(element.element, 'space', None)
    if element.members is None and view_of(element.element).display is not None:
        return None
    return space.dimensions if isinstance(space, LatticeSpace) else (int(element.members or 0) or -1,)


def _profile_name(element: CompiledElement) -> str:
    """Return the name of the local values of a group, its displayed operator, or `probability` for a particle."""
    if element.members is None:
        return 'probability'
    return str(view_of(element.element).display)


class ExpectationVisual(ImageVisual):
    """Visual that logs the local expectations of quantum elements and groups.

    A group logs the expectation of the displayed local operator of every member, such as `z` of spins, as an image of
    its lattice under `<path>/<name>`, with a space-time image under `<path>/space_time/<name>` in one dimension. A
    particle on sites logs the probability of every site the same way under `probability`. Any other element logs the
    expectation of each Hermitian local operator as a series under `<path>/<name>`.

    Args:
        states:
            Unused, kept from `ImageVisual`.

        low:
            The value of the darkest color, or `None` for the smallest value of every frame.

        high:
            The value of the brightest color, or `None` for the largest value of every frame.

        history:
            The largest number of rows of the space-time image, or `None` for as many as sites, between 64 and 256.

    Examples:
        >>> from skcomplex.core.viz import ExpectationVisual
        >>> ExpectationVisual(low=-0.5, high=0.5)
        ExpectationVisual(states=None, low=-0.5, high=0.5, history=None)
    """

    def __repr__(self) -> str:
        """Return the options."""
        return super().__repr__().replace('ImageVisual', 'ExpectationVisual', 1)

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log the local expectations of a quantum element or group at one frame.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path.

            frame:
                The frame.
        """
        problem = frame.problem
        if not isinstance(problem, QuantumProblem):
            return
        values = problem.local_values(element.path, frame.state)
        shape = _shape_of(element)
        for name, value in values.items():
            path = f'{element.path}/{name}'
            if shape is None:
                _log_scalars(recording, path, value)
                continue
            array: npt.NDArray[np.float64] = np.asarray(value, dtype=float)
            if len(shape) == 1:
                self._log_space_time(recording, f'{element.path}/{SPACE_TIME}/{name}', array)
            self._log_image(recording, path, array.reshape(shape if shape[0] > 0 else (-1,)), cyclic=False)

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
        """Return the space-time and image views of a quantum group or particle, or a series view of an element.

        Args:
            element:
                The compiled element, with its path.

            problem:
                The compiled problem the element belongs to.

        Returns:
            views:
                The space-time view first in one dimension, then the image view, or one time series view.
        """
        if not isinstance(element, CompiledElement):
            return []
        shape = _shape_of(element)
        if shape is None:
            return [rrb.TimeSeriesView(origin=element.path, name=element.path)]
        name = _profile_name(element)
        views: list[rrb.View] = []
        if len(shape) == 1 and self._history_of(shape[0]):
            views.append(rrb.Spatial2DView(origin=f'{element.path}/{SPACE_TIME}/{name}', name=f'{name} history'))
        views.append(rrb.Spatial2DView(origin=f'{element.path}/{name}', name=name))
        return views


register_visual('expectations', ExpectationVisual)
