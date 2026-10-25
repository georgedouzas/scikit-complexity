"""Rewriting systems: productions on words, substitutions of tiles, graph productions, and their geometry."""

from ...core.base import Form
from .._formalism import Formalism, register
from ._components import HostGraphComponent, PatchComponent, WordComponent, structure_of
from ._geometry import WordGeometry
from ._interactions import Rewrite, Rewriting
from ._laws import Affine, GraphProductionLaw, Module, ProductionLaw, Prototile, RewritingLaw, SubstitutionLaw
from ._passes import CompiledGraph, CompiledPatch, RewritingPass, RewritingProblem, compile_rewriting
from ._solvers import RewritingSolver
from ._structures import GraphStructure, PatchStructure, StructureDomain, WordStructure
from ._visuals import GeometryVisual
from ._words import CompiledWord

__all__: list[str] = [
    'Affine',
    'CompiledGraph',
    'CompiledPatch',
    'CompiledWord',
    'GeometryVisual',
    'GraphProductionLaw',
    'GraphStructure',
    'HostGraphComponent',
    'Module',
    'PatchComponent',
    'PatchStructure',
    'ProductionLaw',
    'Prototile',
    'Rewrite',
    'Rewriting',
    'RewritingLaw',
    'RewritingPass',
    'RewritingProblem',
    'RewritingSolver',
    'StructureDomain',
    'SubstitutionLaw',
    'WordComponent',
    'WordGeometry',
    'WordStructure',
    'compile_rewriting',
    'structure_of',
]

register(
    Formalism(
        'rewriting',
        laws=(ProductionLaw, SubstitutionLaw, GraphProductionLaw),
        components=(WordComponent, PatchComponent, HostGraphComponent, Rewriting, Rewrite),
        passes=(RewritingPass(),),
        forms=(Form.REWRITING,),
        solvers={'rewriter': RewritingSolver},
        defaults=(RewritingSolver,),
        visuals={'geometry': GeometryVisual},
    ),
)
