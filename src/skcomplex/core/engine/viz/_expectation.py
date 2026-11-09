"""Draw the local expectation values of operator elements and groups.

They are drawn as images and series.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb
import scipy.sparse as sp
from sage.all__sagemath_symbolics import SR

from ...components.spaces import LatticeSpace
from ...components.specs import CompiledAgentsSpec, CompiledElementSpec, OperatorViewSpec
from ...errors import ValidationError
from ...types import Array
from ..compile.passes._operator_compiler import OperatorCompiler
from ..problems import BaseProblemSpec, FrameSpec
from ..problems._operator_kernel import OperatorKernel
from ..problems._operator_problem_spec import OperatorProblemSpec
from ._base import BaseVisual
from ._images import _SPACE_TIME, ImageVisual


class ExpectationVisual(ImageVisual):
    """Visual that logs the local expectations of operator elements and groups.

    A group logs the expectation of the displayed local operator of every member, such as `z`, as an image of
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
        >>> from skcomplex.core.engine.viz import ExpectationVisual
        >>> ExpectationVisual(low=-0.5, high=0.5)
        ExpectationVisual(high=0.5, low=-0.5)
    """

    def _log(self, recording: rr.RecordingStream, element: CompiledElementSpec, frame: FrameSpec) -> None:
        """Log the local expectations of an operator element or group at one frame.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path.

            frame:
                The frame.
        """
        problem = frame.problem
        if not isinstance(problem, OperatorProblemSpec):
            return
        values = self._local_values(problem, element.path, frame.state)
        shape = self._shape_of(element)
        for name, value in values.items():
            path = f'{element.path}/{name}'
            if shape is None:
                BaseVisual._log_scalars(recording, path, value)
                continue
            array: npt.NDArray[np.float64] = np.asarray(value, dtype=float)
            if len(shape) == 1:
                self._log_space_time(recording, f'{element.path}/{_SPACE_TIME}/{name}', array)
            self._log_image(recording, path, array.reshape(shape if shape[0] > 0 else (-1,)), cyclic=False)

    def _views(self, element: CompiledElementSpec | CompiledAgentsSpec, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the views of an operator element, group, or particle.

        A group or particle gets space-time and image views, and an element a series view.

        Args:
            element:
                The compiled element, with its path.

            problem:
                The compiled problem the element belongs to.

        Returns:
            views:
                The space-time view first in one dimension, then the image view, or one time series view.
        """
        if not isinstance(element, CompiledElementSpec):
            return []
        shape = self._shape_of(element)
        if shape is None:
            return [rrb.TimeSeriesView(origin=element.path, name=element.path)]
        name = self._profile_name(element)
        views: list[rrb.View] = []
        if len(shape) == 1 and self._history_of(shape[0]):
            views.append(rrb.Spatial2DView(origin=f'{element.path}/{_SPACE_TIME}/{name}', name=f'{name} history'))
        views.append(rrb.Spatial2DView(origin=f'{element.path}/{name}', name=name))
        return views

    @staticmethod
    def _profile_name(element: CompiledElementSpec) -> str:
        """Return the name of the local values of a group.

        It is its displayed operator, or `probability` for a particle.
        """
        if element.members is None:
            return 'probability'
        return str(ExpectationVisual._view_of(element).display)

    @staticmethod
    def _view_of(element: CompiledElementSpec) -> OperatorViewSpec:
        """Return the operator view of a compiled element, of its first member for a group.

        Args:
            element:
                The compiled operator element or group.

        Returns:
            view:
                The operator view of the element or of its first member.

        Raises:
            ValidationError:
                The element is neither an element nor a group of elements.
        """
        member = OperatorCompiler._member_of(element.element)
        if member is None:
            error_msg = f'{element.path} is neither an operator element nor a group of them.'
            raise ValidationError(error_msg)
        return OperatorCompiler._view_of(member)

    @staticmethod
    def _shape_of(element: CompiledElementSpec) -> tuple[int, ...] | None:
        """Return the lattice shape of the local values of an operator group or particle.

        It returns `None` for scalars.
        """
        space = getattr(element.element, 'space', None)
        if element.members is None and ExpectationVisual._view_of(element).display is not None:
            return None
        return space.shape_ if isinstance(space, LatticeSpace) else (int(element.members or 0) or -1,)

    @staticmethod
    def _probabilities(problem: OperatorProblemSpec, x: Array) -> npt.NDArray[np.float64]:
        """Return the probability of every basis state of the whole space."""
        state = problem._state_of(x)
        return np.abs(state) ** 2 if problem.pure else np.real(np.diagonal(state))

    @staticmethod
    def _local_matrices(problem: OperatorProblemSpec, path: str, name: str) -> list[sp.csr_array]:
        """Return the matrices on the whole space of one local operator.

        There is one per member of an element or group.
        """
        cached = problem.cache.get(f'{path}:{name}')
        if cached is None:
            lowering = problem.lowering
            if lowering is None:
                error_msg = f'The operator problem holds no lowering of the operators of {path}.'
                raise ValidationError(error_msg)
            subsystems = [
                item for key, item in lowering.subsystems.items() if key == path or key.startswith(f'{path}[')
            ]
            cached = [lowering._matrix(((item.key, item.operators[name]),), path) for item in subsystems]
            problem.cache[f'{path}:{name}'] = cached
        return cached

    @staticmethod
    def _local_values(problem: OperatorProblemSpec, path: str, x: Array) -> dict[str, npt.NDArray[np.float64] | float]:
        """Return the local expectations of an operator element or group at a state.

        The visual draws them. A group gives the expectation of the displayed local operator of every member. A single
        element without a displayed operator, such as a particle on sites, gives the probability of every basis state,
        and one with it gives the expectation of each of its Hermitian local operators.

        Args:
            problem:
                The operator problem.

            path:
                The path of the element or group.

            x:
                The state vector.

        Returns:
            values:
                A mapping from the name of a local operator, or `'probability'`, to an array or a number.
        """
        element = next(item for item in problem.elements if item.path == path)
        display = ExpectationVisual._view_of(element).display
        if element.members is None and display is None:
            index = problem.subsystems.index(path)
            shaped = ExpectationVisual._probabilities(problem, x).reshape(problem.dimensions)
            others = tuple(axis for axis in range(len(problem.dimensions)) if axis != index)
            return {'probability': shaped.sum(axis=others)}
        if element.members is not None and display is not None:
            matrices = ExpectationVisual._local_matrices(problem, path, display)
            terms = [OperatorKernel(matrix, SR(1)).prepare() for matrix in matrices]
            values = [problem._expect([term], 0.0, x, np.zeros(0)) for term in terms]
            return {display: np.real(np.asarray(values))}
        names = [
            name for name, local in ExpectationVisual._view_of(element).get_local_operators().items() if local.hermitian
        ]
        return {
            name: float(
                np.real(problem._expect([OperatorKernel(matrix, SR(1)).prepare()], 0.0, x, np.zeros(0))),
            )
            for name in names
            for matrix in ExpectationVisual._local_matrices(problem, path, name)
        }
