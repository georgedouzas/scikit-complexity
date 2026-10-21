"""Draw the phases of a group of oscillators against their position along a ring or a chain."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ..base import wrap_angle
from ..compile import CompiledAgents, CompiledElement, Problem
from ..runtime import Frame
from ._visuals import UPWARD, Visual, _log_scalars, _phase_colors, _radius_of, register_visual

PHASE_MARGIN = 0.1
HEIGHT_SHARE = 0.6
POSITION_MARGIN = 0.05


def _phase_scale(extent: float) -> float:
    """Return the height of one radian, so one turn of phase spans a share of the extent of the positions."""
    return HEIGHT_SHARE * max(extent, 1.0) / (2 * np.pi)


class PhaseRingVisual(Visual):
    """Visual that logs the wrapped phases of a group of oscillators against their position as colored points.

    Every member is a point at its position along the group and its phase wrapped into `(-pi, pi]`, colored by the
    cyclic palette of phases, so a coherent stretch of oscillators shows as a flat run of one color and an incoherent
    stretch as scattered points. The points are logged under `<path>/phases` with the vertical axis up, one turn of
    phase spanning six tenths of the extent of the positions, and the other states as time series under
    `<path>/<state>`.

    Args:
        phase:
            The name of the state that holds the phase of every member.

        position:
            The name of a scalar state or of the first coordinate of a vector state that holds the position of every
            member, or `None` for the member index.

    Examples:
        >>> from skcomplex.core.viz import PhaseRingVisual
        >>> PhaseRingVisual('theta').phase
        'theta'
    """

    def __init__(self, phase: str, position: str | None = None) -> None:
        self.phase = phase
        self.position = position

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log the wrapped phases of the members against their positions, and the other states as time series.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.
        """
        phases = wrap_angle(np.ravel(frame.value(element.states[self.phase])))
        if self.position is None:
            positions = np.arange(phases.size, dtype=float)
        else:
            coordinates = np.asarray(frame.value(element.states[self.position]), dtype=float)
            positions = coordinates.reshape(phases.size, -1)[:, 0]
        extent = float(np.ptp(positions)) if positions.size else 1.0
        points = np.column_stack([positions, phases * _phase_scale(extent)]) * UPWARD
        radius = _radius_of(max(extent, 1.0), phases.size)
        recording.log(f'{element.path}/phases', rr.Points2D(points, radii=radius, colors=_phase_colors(phases)))
        for state, name in element.states.items():
            if state not in {self.phase, self.position}:
                _log_scalars(recording, f'{element.path}/{state}', frame.value(name))

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
        """Return a 2D view of the phases against the positions, bounded by one turn of phase.

        Args:
            element:
                The compiled element, with its path.

            problem:
                The compiled problem the element belongs to.

        Returns:
            views:
                A 2D spatial view of `<path>/phases`.
        """
        if not isinstance(element, CompiledElement):
            return []
        count = max(element.members or 1, 1)
        if self.position is None:
            low, high = 0.0, float(count - 1)
        else:
            entry = problem.states.read(problem.initial_state, element.states[self.position])
            positions = np.asarray(entry, dtype=float).reshape(count, -1)[:, 0]
            low, high = float(positions.min()), float(positions.max())
        margin = max(high - low, 1.0) * POSITION_MARGIN
        height = (np.pi + PHASE_MARGIN) * _phase_scale(high - low)
        bounds = rrb.VisualBounds2D(
            x_range=(low - margin, high + margin),
            y_range=(-height, height),
        )
        path = f'{element.path}/phases'
        return [rrb.Spatial2DView(origin=path, name=element.path, visual_bounds=bounds)]


register_visual('phase_ring', PhaseRingVisual)
