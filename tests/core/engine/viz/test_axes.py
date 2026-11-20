"""Test the axes visual."""

from typing import Annotated

import numpy as np
import pytest
import rerun as rr

from skcomplex.core.components.elements import Element, Tensor, VisualSpec
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import BaseSpace, LatticeSpace
from skcomplex.core.engine.viz import AxisVisual, RerunObserver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System


class _NematicElement(Element):
    """Element with a density, a vector, and a symmetric traceless tensor."""

    def __init__(
        self,
        label,
        Q: Annotated[object, Tensor('traceless')] = 0.0,
        rho: float = 1.0,
        p: object = 0.0,
        *,
        space: BaseSpace | None = None,
    ):
        super().__init__(label, space=space)
        self.Q = Q
        self.rho = rho
        self.p = p


def _film():
    """Return a field on a lattice of 8 by 8 sites with a uniform tensor that decays and a constant density."""
    Q, p, rho = np.tile([0.5, 0.0], (64, 1)), np.tile([1.0, 0.0], (64, 1)), np.arange(64.0)
    field = _NematicElement('film', Q=Q, rho=rho, p=p, space=LatticeSpace((8, 8)))
    decay = Interaction(
        'decay',
        roles={'target': 'film'},
        laws=lambda s: [
            RateLaw(s.target.Q, -s.target.Q),
            RateLaw(s.target.rho, 0.0 * s.target.rho),
            RateLaw(s.target.p, 0.0 * s.target.p),
        ],
    )
    return System(elements=[field], interactions=[decay])


class _Recording:
    """Recording that keeps what a visual logs."""

    def __init__(self):
        self.logged = []

    def log(self, path, item):
        """Keep a path and what it logs."""
        self.logged.append((path, item))


def _logged(state, scalar, every):
    """Return what the axes visual of a state logs at the first and second frames of a short run."""
    recording = _Recording()
    system = _film().prepare()
    problem = system._problem
    visual = AxisVisual(state, scalar=scalar, every=every, path='film', recording=recording).prepare(problem)
    frames = []

    class FrameObserver(RerunObserver):
        def _run(self, inputs):
            if inputs is not None:
                frames.append(inputs)

    system.simulate(until=0.2, dt=0.1, observers=[FrameObserver()])
    for frame in frames:
        visual.run(frame)
    return recording.logged, len(frames), visual


def test_run_logs_one_segment_per_sampled_site_over_the_image_at_every_frame():
    """A tensor state logs one segment along its principal axis per sampled site over the image at every frame."""
    logged, frames, visual = _logged('Q', 'rho', 2)
    segments = [item for path, item in logged if path == 'film/Q/axes']
    images = [item for path, item in logged if path == 'film/Q/image']
    assert len(segments) == frames
    assert len(images) == frames
    assert frames >= 2
    assert len(segments[0].strips.as_arrow_array()) == len(visual.sites_) == 16


def test_run_writes_a_recording_in_the_viewer_stream(tmp_path):
    """The nematic with a visual of axes writes a recording of its frames."""
    visuals = {'film': VisualSpec('axes', {'state': 'Q', 'scalar': 'rho', 'every': 2})}
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd', visuals=visuals)
    _film().prepare().simulate(until=0.2, dt=0.1, observers=[observer])
    assert isinstance(observer.visuals_[0], AxisVisual)
    assert observer.logged_frames_ >= 2
    assert (tmp_path / 'run.rrd').stat().st_size > 0


def test_run_logs_arrows_of_a_vector_state():
    """A vector state logs one arrow per sampled site."""
    logged, _, visual = _logged('p', None, 4)
    arrows = logged[0][1]
    assert isinstance(arrows, rr.Arrows2D)
    assert len(arrows.vectors.as_arrow_array()) == len(visual.sites_)


def test_prepare_rejects_a_scalar_state():
    """A scalar state has no axes."""
    with pytest.raises(ValidationError, match='no tensor or vector'):
        AxisVisual('rho', path='film', recording=_Recording()).prepare(
            _film().prepare()._problem,
        )


def test_check_rejects_a_spacing_that_is_not_positive():
    """A spacing of 0 raises an error."""
    with pytest.raises(ValidationError, match='every'):
        AxisVisual('Q', every=0, path='film').prepare()
