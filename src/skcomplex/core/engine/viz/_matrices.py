"""Draw matrix-valued states and parameters in Rerun as heat maps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ...components.specs import CompiledAgentsSpec, CompiledElementSpec
from ...errors import ValidationError
from ..problems import BaseProblemSpec, FrameSpec
from ._base import BaseVisual

MATRIX_RANK = 2


class MatrixVisual(BaseVisual):
    """Visual that logs a matrix-valued state or parameter as an image colored by value, one image per frame.

    The visual draws the variable `variable` of the element at `path`, or the state or parameter named
    `<path>.<variable>` of a problem without such an element, such as a trained matrix of a system that trains, under
    the entity path `<path>/<variable>`. The colors span the fixed range `color_range` when it is given and the range
    of the frame otherwise.

    Args:
        variable:
            The name of the matrix-valued state or parameter.

        color_range:
            The values of the darkest and the brightest color, or `None` for the range of every frame.

        path:
            The entity path of the element, or the path the name of the variable starts with.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Attributes:
        name_:
            The dotted name of the variable in the problem.

        shape_:
            The shape of the matrix.

    Raises:
        ValidationError:
            The variable is not a name, or the problem has no matrix of that name.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.core.engine.viz import MatrixVisual
        >>> from skcomplex.systems.computer_science.neural_networks import ElmanNetworkSystem
        >>> problem = PassCompiler().prepare(ElmanNetworkSystem(n_bits=30).prepare().system_).run().problem_
        >>> MatrixVisual('A', path='net/recurrence').prepare(problem).name_
        'net/recurrence.A'
    """

    name_: str
    shape_: tuple[int, ...]

    def __init__(
        self,
        variable: str = 'W',
        *,
        color_range: tuple[float, float] | None = None,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        super().__init__(path=path, recording=recording)
        self.variable = variable
        self.color_range = color_range

    def _check(self) -> None:
        """Raise an error unless the path and the variable are strings."""
        super()._check()
        if not isinstance(self.variable, str):
            error_msg = f'variable = {self.variable!r} must be the name of a matrix-valued state or parameter.'
            raise ValidationError(error_msg)

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Find the matrix of the element at the path, or the matrix named by the path and the variable, and the view.

        Args:
            context:
                The compiled problem.

        Raises:
            ValidationError:
                No problem is passed, or it has no matrix of that name.
        """
        if context is None:
            error_msg = 'A visual prepares on a compiled problem. Pass the problem to `prepare`.'
            raise ValidationError(error_msg)
        element = next((item for item in context.elements if item.path == self.path), None)
        if element is not None:
            super()._prepare(context)
            names = {**element.states, **element.parameters}
            self.name_ = names.get(self.variable, f'{self.path}.{self.variable}')
        else:
            self.name_ = f'{self.path}.{self.variable}'
            self.recording_ = self.recording if self.recording is not None else rr.get_data_recording()
            self.views_ = [rrb.Spatial2DView(origin=f'{self.path}/{self.variable}', name=self.name_)]
        layouts = [layout for layout in (context.states, context.parameters) if layout.has(self.name_)]
        if not layouts or len(layouts[0].entry(self.name_).shape) != MATRIX_RANK:
            error_msg = f'The problem has no matrix named {self.name_!r}. Name a state or parameter of two dimensions.'
            raise ValidationError(error_msg)
        self.shape_ = layouts[0].entry(self.name_).shape

    def _views(self, element: CompiledElementSpec | CompiledAgentsSpec, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the 2D spatial view of the matrix."""
        del problem
        return [rrb.Spatial2DView(origin=f'{element.path}/{self.variable}', name=f'{element.path}.{self.variable}')]

    def _log_matrix(self, recording: rr.RecordingStream, frame: FrameSpec) -> None:
        """Log the matrix at a frame as an image colored by value."""
        values = np.reshape(np.asarray(frame.value(self.name_), dtype=float), self.shape_)
        low, high = self.color_range if self.color_range is not None else (float(values.min()), float(values.max()))
        colors = BaseVisual._palette_colors(values, low, high)
        height, width = values.shape
        image = rr.Image(bytes=colors.tobytes(), width=width, height=height, color_model='RGB', datatype='U8')
        recording.log(f'{self.path}/{self.variable}', image)

    def _run(self, inputs: FrameSpec | None) -> None:
        """Log the matrix at a frame.

        Args:
            inputs:
                The frame.

        Raises:
            ValidationError:
                No frame is passed, or no Rerun recording is active.
        """
        if inputs is None:
            error_msg = 'A visual runs on a frame. Pass the frame to `run`.'
            raise ValidationError(error_msg)
        if self.recording_ is None:
            error_msg = 'No Rerun recording is active. Pass `recording` or start one with `rerun.init`.'
            raise ValidationError(error_msg)
        self._log_matrix(self.recording_, inputs)
