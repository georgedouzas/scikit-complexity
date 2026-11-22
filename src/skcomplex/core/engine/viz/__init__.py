"""Draw running simulations in Rerun.

A visual draws the values of an element at every frame, such as the points of a flock, the image of a grid, or the line
of a population over time.
"""

from ._agents import AgentsVisual
from ._axes import AxisVisual
from ._base import BaseVisual
from ._bodies import BodiesVisual
from ._cells import CellVisual
from ._expectation import ExpectationVisual
from ._histograms import HistogramVisual
from ._images import ImageVisual
from ._linear_programs import LinearProgramVisual
from ._matrices import MatrixVisual
from ._meshes import MeshVisual
from ._points import PointsVisual
from ._rerun import RerunObserver
from ._rings import PhaseRingVisual
from ._scalars import ScalarsVisual
from ._sites import SiteVisual

__all__: list[str] = [
    'AgentsVisual',
    'AxisVisual',
    'BaseVisual',
    'BodiesVisual',
    'CellVisual',
    'ExpectationVisual',
    'HistogramVisual',
    'ImageVisual',
    'LinearProgramVisual',
    'MatrixVisual',
    'MeshVisual',
    'PhaseRingVisual',
    'PointsVisual',
    'RerunObserver',
    'ScalarsVisual',
    'SiteVisual',
]
