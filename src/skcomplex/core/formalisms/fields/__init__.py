"""Fields: differential operators, flux laws, meshes, their discretizations, and stiff integrators."""

from .. import lattices  # noqa: F401 - registers the formalism that fields builds on
from .._formalism import Formalism, register
from ._components import Field
from ._laws import Flux, advection, divergence, gradient, inverse_laplacian, laplacian
from ._lowering import mesh_laplacian
from ._passes import (
    CHECKS,
    ConservedNoisePass,
    FieldFunctionsPass,
    FieldJacobianPass,
    FieldProblemPass,
    LowerFieldsPass,
)
from ._solvers import ETDRK4, IMEX2, SSPRK, CrankNicolson, IMEXEuler, SplitOperator
from ._spaces import Mesh

__all__: list[str] = [
    'ETDRK4',
    'IMEX2',
    'SSPRK',
    'ConservedNoisePass',
    'CrankNicolson',
    'Field',
    'FieldFunctionsPass',
    'FieldJacobianPass',
    'FieldProblemPass',
    'Flux',
    'IMEXEuler',
    'LowerFieldsPass',
    'Mesh',
    'SplitOperator',
    'advection',
    'divergence',
    'gradient',
    'inverse_laplacian',
    'laplacian',
    'mesh_laplacian',
]

register(
    Formalism(
        'fields',
        depends=('lattices',),
        laws=(Flux,),
        components=(Field,),
        spaces=(Mesh,),
        passes=(LowerFieldsPass(), ConservedNoisePass(), FieldFunctionsPass(), FieldJacobianPass(), FieldProblemPass()),
        solvers={
            'ssp-rk': SSPRK,
            'imex-euler': IMEXEuler,
            'imex2': IMEX2,
            'crank-nicolson': CrankNicolson,
            'etdrk4': ETDRK4,
        },
        checks=CHECKS,
    ),
)
