"""Visualize running simulations in Rerun."""

from ._blueprints import default_blueprint, primary_view, save_blueprint
from ._bodies import BodiesVisual
from ._rerun import RerunObserver
from ._rings import PhaseRingVisual
from ._traits import TraitsVisual, agents_visual_of
from ._visuals import (
    NODE_SPACING,
    SPACE_TIME,
    UPWARD,
    VISUALS,
    AgentsVisual,
    ImageVisual,
    MeshVisual,
    PointsVisual,
    ScalarsVisual,
    SiteVisual,
    Visual,
    colors_of,
    log_scalars,
    palette_colors,
    register_visual,
    resolve_visual,
)

__all__: list[str] = [
    'NODE_SPACING',
    'SPACE_TIME',
    'UPWARD',
    'VISUALS',
    'AgentsVisual',
    'BodiesVisual',
    'ImageVisual',
    'MeshVisual',
    'PhaseRingVisual',
    'PointsVisual',
    'RerunObserver',
    'ScalarsVisual',
    'SiteVisual',
    'TraitsVisual',
    'Visual',
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
