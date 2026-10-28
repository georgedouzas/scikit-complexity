"""Fractals: escape-time sets, chaos-game attractors, and words and patches that grow by rewriting.

The rewriting classes are a worked example of a form defined outside the engine: elements and laws on the core bases,
kernels, a compiler, a solver, and a visual on the engine bases. A rewriting system runs with
`Simulation(system, compiler=RewritingCompiler())`.
"""

from ._affine_spec import AffineSpec
from ._chaos_game import ChaosPointElement, chaos_game
from ._geometry_spec import GeometrySpec
from ._geometry_visual import GeometryVisual
from ._module_spec import Letters, ModuleSpec
from ._patch_element import PatchElement
from ._patch_structure_spec import PatchStructureSpec
from ._production_law import ProductionLaw
from ._prototile_spec import PrototileSpec
from ._quadratic import QuadraticMapElement, julia, mandelbrot
from ._rewriting_compiler import RewritingCompiler
from ._rewriting_error import RewritingError
from ._rewriting_problem_spec import RewritingProblemSpec
from ._rewriting_solver import RewritingSolver
from ._substitution import (
    CantorElement,
    KochElement,
    SierpinskiElement,
    cantor_set,
    koch_curve,
    sierpinski_substitution,
)
from ._substitution_law import SubstitutionLaw
from ._word_element import WordElement
from ._word_structure_spec import WordStructureSpec

__all__: list[str] = [
    'AffineSpec',
    'CantorElement',
    'ChaosPointElement',
    'GeometrySpec',
    'GeometryVisual',
    'KochElement',
    'Letters',
    'ModuleSpec',
    'PatchElement',
    'PatchStructureSpec',
    'ProductionLaw',
    'PrototileSpec',
    'QuadraticMapElement',
    'RewritingCompiler',
    'RewritingError',
    'RewritingProblemSpec',
    'RewritingSolver',
    'SierpinskiElement',
    'SubstitutionLaw',
    'WordElement',
    'WordStructureSpec',
    'cantor_set',
    'chaos_game',
    'julia',
    'koch_curve',
    'mandelbrot',
    'sierpinski_substitution',
]
