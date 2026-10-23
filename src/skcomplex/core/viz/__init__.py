"""Visualize running simulations in Rerun."""

from ._blueprints import default_blueprint, primary_view, save_blueprint
from ._bodies import BodiesVisual
from ._cayley import CayleyVisual
from ._rerun import RerunObserver
from ._rings import PhaseRingVisual
from ._traits import TraitsVisual, agents_visual_of
from ._visuals import (
    SPACE_TIME,
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
    register_visual,
    resolve_visual,
)

__all__: list[str] = [
    'SPACE_TIME',
    'VISUALS',
    'AgentsVisual',
    'BodiesVisual',
    'CayleyVisual',
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
    'primary_view',
    'register_visual',
    'resolve_visual',
    'save_blueprint',
]
