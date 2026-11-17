"""Define the rewriting of words and tilings that several fields share.

A word grows by productions, as an L-system, and a patch of tiles grows by substitutions, as a fractal tiling. The
package brings the compiler, the solver, and the drawing of these systems.
"""

from ._affine_spec import AffineSpec
from ._geometry_spec import GeometrySpec
from ._geometry_visual import GeometryVisual
from ._module_spec import ModuleSpec
from ._patch import PatchElement
from ._production import ProductionLaw
from ._prototile_spec import PrototileSpec
from ._rewriting_solver import RewritingSolver
from ._substitution import SubstitutionLaw
from ._word import WordElement

__all__: list[str] = [
    'AffineSpec',
    'GeometrySpec',
    'GeometryVisual',
    'ModuleSpec',
    'PatchElement',
    'ProductionLaw',
    'PrototileSpec',
    'RewritingSolver',
    'SubstitutionLaw',
    'WordElement',
]
