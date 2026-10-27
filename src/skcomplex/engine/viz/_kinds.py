"""Name the visual of every kind of `VisualSpec`, and build the visual of a spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...core.specs import VisualSpec
from ..problems import BaseProblem, CompiledAgents
from ._agents import AgentsVisual
from ._bodies import BodiesVisual
from ._cayley import CayleyVisual
from ._images import ImageVisual
from ._meshes import MeshVisual
from ._points import PointsVisual
from ._quantum import ExpectationVisual
from ._rewriting import GeometryVisual
from ._rings import PhaseRingVisual
from ._scalars import ScalarsVisual
from ._sites import SiteVisual
from ._traits import TraitsVisual
from ._visuals import BaseVisual

VISUALS: dict[str, type[BaseVisual]] = {
    'agents': AgentsVisual,
    'bodies': BodiesVisual,
    'cayley': CayleyVisual,
    'expectations': ExpectationVisual,
    'geometry': GeometryVisual,
    'image': ImageVisual,
    'mesh': MeshVisual,
    'phase_ring': PhaseRingVisual,
    'points': PointsVisual,
    'scalars': ScalarsVisual,
    'sites': SiteVisual,
    'traits': TraitsVisual,
}


def resolve_visual(spec: VisualSpec) -> BaseVisual:
    """Build the visual a spec names.

    Args:
        spec:
            The visual spec of an element.

    Returns:
        visual:
            The visual of the spec's kind, built with the spec's options.

    Examples:
        >>> from skcomplex.core.specs import VisualSpec
        >>> from skcomplex.engine.viz import resolve_visual
        >>> type(resolve_visual(VisualSpec('image'))).__name__
        'ImageVisual'
    """
    return VISUALS[spec.kind](**dict(spec.options))


def agents_visual_of(agents: CompiledAgents, problem: BaseProblem) -> AgentsVisual:
    """Return the visual of an agent group: its declared agents visual, or the traits visual without a space.

    Args:
        agents:
            The compiled agents.

        problem:
            The compiled problem, with the space the agents live in.

    Returns:
        visual:
            The visual that logs the agents.

    Examples:
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.engine.viz import agents_visual_of
        >>> from skcomplex.models import create
        >>> problem = compile_system(create('moran_process'))
        >>> type(agents_visual_of(problem.agents[0], problem)).__name__
        'TraitsVisual'
    """
    visual = resolve_visual(agents.visual)
    if not isinstance(visual, AgentsVisual):
        visual = AgentsVisual()
    if problem.agent_space is None and type(visual) is AgentsVisual and visual.position is None:
        return TraitsVisual(color_by=visual.color_by, color_range=visual.color_range)
    return visual
