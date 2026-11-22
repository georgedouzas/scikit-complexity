"""Draw the phases of a group of oscillators against their position."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ...components import EntrySpec
from ..problems import BaseProblemSpec, CompiledElementSpec, FrameSpec
from ..problems._compiled_agents_spec import _CompiledAgentsSpec
from ._base import UPWARD, BaseVisual

PHASE_MARGIN = 0.1
HEIGHT_SHARE = 0.6
POSITION_MARGIN = 0.05


class PhaseRingVisual(BaseVisual):
    """A visual that draws the phases of a group of oscillators against their position.

    It draws every oscillator as a point at its position and its phase wrapped into `(-pi, pi]`, coloured by a palette
    that wraps around, such as `PhaseRingVisual('theta')`. Oscillators in step show as a flat run of one colour, and
    oscillators out of step as scattered points. The other values are drawn as lines over time. It shows fireflies
    flashing together, the generators of a power grid, and the neurons of a ring firing in waves.

    Read more in the user guide, under Observe.

    Args:
        phase:
            The name of the state that holds the phase of every member.

        position:
            The name of a scalar state or of the first coordinate of a vector state that holds the position of every
            member, or `None` for the member index.

        path:
            The entity path of the element or agent group to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    See Also:
        PointsVisual: Draws a group at its positions in space.

    Examples:
        >>> from skcomplex.core.engine.viz import PhaseRingVisual
        >>> PhaseRingVisual('theta').phase
        'theta'
    """

    def __init__(
        self,
        phase: str,
        position: str | None = None,
        *,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        self.phase = phase
        self.position = position
        self.path = path
        self.recording = recording

    def _log(self, recording: rr.RecordingStream, element: CompiledElementSpec, frame: FrameSpec) -> None:
        """Log the wrapped phases of the members against their positions.

        The other states are logged as time series.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.
        """
        phases = EntrySpec._wrap_angle(np.ravel(frame.get_value(element.states[self.phase])))
        if self.position is None:
            positions = np.arange(phases.size, dtype=float)
        else:
            coordinates = np.asarray(frame.get_value(element.states[self.position]), dtype=float)
            positions = coordinates.reshape(phases.size, -1)[:, 0]
        extent = float(np.ptp(positions)) if positions.size else 1.0
        points = np.column_stack([positions, phases * self._get_phase_scale(extent)]) * UPWARD
        radius = BaseVisual._get_radius_of(max(extent, 1.0), phases.size)
        recording.log(
            f'{element.path}/phases',
            rr.Points2D(points, radii=radius, colors=BaseVisual._get_phase_colors(phases)),
        )
        for state, name in element.states.items():
            if state not in {self.phase, self.position}:
                BaseVisual._log_scalars(
                    recording,
                    BaseVisual._get_state_path(element.path, state),
                    frame.get_value(name),
                )

    def _get_views(
        self,
        element: CompiledElementSpec | _CompiledAgentsSpec,
        problem: BaseProblemSpec,
    ) -> list[rrb.View]:
        """Return a 2D view of the phases against the positions.

        It is bounded by one turn of phase.

        Args:
            element:
                The compiled element, with its path.

            problem:
                The compiled problem the element belongs to.

        Returns:
            views:
                A 2D spatial view of `<path>/phases`.
        """
        if not isinstance(element, CompiledElementSpec):
            return []
        count = max(element.members or 1, 1)
        if self.position is None:
            low, high = 0.0, float(count - 1)
        else:
            entry = problem.states.read(problem.initial_state, element.states[self.position])
            positions = np.asarray(entry, dtype=float).reshape(count, -1)[:, 0]
            low, high = float(positions.min()), float(positions.max())
        margin = max(high - low, 1.0) * POSITION_MARGIN
        height = (np.pi + PHASE_MARGIN) * self._get_phase_scale(high - low)
        bounds = rrb.VisualBounds2D(
            x_range=(low - margin, high + margin),
            y_range=(-height, height),
        )
        path = f'{element.path}/phases'
        return [rrb.Spatial2DView(origin=path, name=element.path, visual_bounds=bounds)]

    @staticmethod
    def _get_phase_scale(extent: float) -> float:
        """Return the height of one radian.

        One turn of phase then spans a share of the extent of the positions.
        """
        return HEIGHT_SHARE * max(extent, 1.0) / (2 * np.pi)
