"""Visualize running simulations in Rerun."""

from ._agents import AgentsVisual
from ._blueprints import default_blueprint, primary_view, save_blueprint
from ._bodies import BodiesVisual
from ._colors import colors_of, palette_colors
from ._images import SPACE_TIME, ImageVisual
from ._meshes import MeshVisual
from ._points import PointsVisual
from ._rerun import RerunObserver
from ._rings import PhaseRingVisual
from ._scalars import ScalarsVisual
from ._sites import SiteVisual
from ._traits import TraitsVisual, agents_visual_of
from ._utils import NODE_SPACING, UPWARD, log_scalars
from ._visuals import BaseVisual, register_visual, resolve_visual

__all__: list[str] = [
    'NODE_SPACING',
    'SPACE_TIME',
    'UPWARD',
    'AgentsVisual',
    'BaseVisual',
    'BodiesVisual',
    'ImageVisual',
    'MeshVisual',
    'PhaseRingVisual',
    'PointsVisual',
    'RerunObserver',
    'ScalarsVisual',
    'SiteVisual',
    'TraitsVisual',
    'agents_visual_of',
    'colors_of',
    'default_blueprint',
    'log_scalars',
    'palette_colors',
    'primary_view',
    'register_visual',
    'resolve_visual',
    'save_blueprint',
]
