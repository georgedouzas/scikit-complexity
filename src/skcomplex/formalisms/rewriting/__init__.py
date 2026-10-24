"""Rewriting systems: productions on words, substitutions of tiles, graph productions, and their geometry."""

from ...core.base import Form
from .._formalism import Formalism, register
from ._components import HostGraph, Patch, Word, structure_of
from ._geometry import WordGeometry
from ._interactions import Rewrite, Rewriting
from ._laws import Affine, GraphProduction, Module, Production, Prototile, RewritingLaw, Substitution
from ._passes import CompiledGraph, CompiledPatch, RewritingPass, RewritingProblem, compile_rewriting
from ._solvers import Rewriter
from ._structures import GraphStructure, PatchStructure, Structure, WordStructure
from ._visuals import GeometryVisual
from ._words import CompiledWord

__all__: list[str] = [
    'Affine',
    'CompiledGraph',
    'CompiledPatch',
    'CompiledWord',
    'GeometryVisual',
    'GraphProduction',
    'GraphStructure',
    'HostGraph',
    'Module',
    'Patch',
    'PatchStructure',
    'Production',
    'Prototile',
    'Rewrite',
    'Rewriter',
    'Rewriting',
    'RewritingLaw',
    'RewritingPass',
    'RewritingProblem',
    'Structure',
    'Substitution',
    'Word',
    'WordGeometry',
    'WordStructure',
    'compile_rewriting',
    'structure_of',
]

register(
    Formalism(
        'rewriting',
        laws=(Production, Substitution, GraphProduction),
        components=(Word, Patch, HostGraph, Rewriting, Rewrite),
        passes=(RewritingPass(),),
        forms=(Form.REWRITING,),
        solvers={'rewriter': Rewriter},
        defaults=(Rewriter,),
        visuals={'geometry': GeometryVisual},
    ),
)
