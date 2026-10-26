"""Rewriting systems: productions on words, substitutions of tiles, graph productions, and their geometry."""

from ...core.base import Form
from .._formalism import Formalism, register
from ._components import HostGraphElement, PatchElement, StructureDomain, WordElement, structure_of
from ._geometry import WordGeometry
from ._laws import (
    Affine,
    GraphProductionLaw,
    Module,
    ProductionLaw,
    Prototile,
    RewriteLaw,
    RewritingError,
    RewritingLaw,
    SubstitutionLaw,
)
from ._passes import CompiledGraph, CompiledPatch, RewritingCompiler, RewritingProblem, compile_rewriting
from ._solvers import RewritingSolver
from ._structures import GraphStructure, PatchStructure, Structure, WordStructure
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
    'HostGraphElement',
    'Module',
    'PatchElement',
    'PatchStructure',
    'ProductionLaw',
    'Prototile',
    'RewriteLaw',
    'RewritingCompiler',
    'RewritingError',
    'RewritingLaw',
    'RewritingProblem',
    'RewritingSolver',
    'Structure',
    'StructureDomain',
    'SubstitutionLaw',
    'WordElement',
    'WordGeometry',
    'WordStructure',
    'compile_rewriting',
    'structure_of',
]

register(
    Formalism(
        'rewriting',
        laws=(ProductionLaw, SubstitutionLaw, GraphProductionLaw, RewritingLaw, RewriteLaw),
        components=(WordElement, PatchElement, HostGraphElement),
        compilers=(RewritingCompiler(),),
        forms=(Form.REWRITING,),
        solvers={'rewriter': RewritingSolver},
        defaults=(RewritingSolver,),
        visuals={'geometry': GeometryVisual},
    ),
)
