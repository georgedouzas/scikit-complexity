"""Fields: differential operators, flux laws, meshes, their discretizations, and stiff integrators."""

from .. import lattices  # noqa: F401 - registers the formalism that fields builds on
from .._formalism import Formalism, register
from ._components import FieldComponent
from ._laws import FluxLaw, advection, divergence, gradient, inverse_laplacian, laplacian
from ._lowering import mesh_laplacian
from ._passes import (
    CHECKS,
    ConservedNoisePass,
    FieldFunctionsPass,
    FieldJacobianPass,
    FieldProblemPass,
    LowerFieldsPass,
)
from ._solvers import CrankNicolsonSolver, ETDRK4Solver, IMEX2Solver, IMEXEulerSolver, SplitOperator, SSPRKSolver
from ._spaces import MeshSpace

__all__: list[str] = [
    'ConservedNoisePass',
    'CrankNicolsonSolver',
    'ETDRK4Solver',
    'FieldComponent',
    'FieldFunctionsPass',
    'FieldJacobianPass',
    'FieldProblemPass',
    'FluxLaw',
    'IMEX2Solver',
    'IMEXEulerSolver',
    'LowerFieldsPass',
    'MeshSpace',
    'SSPRKSolver',
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
        laws=(FluxLaw,),
        components=(FieldComponent,),
        spaces=(MeshSpace,),
        passes=(LowerFieldsPass(), ConservedNoisePass(), FieldFunctionsPass(), FieldJacobianPass(), FieldProblemPass()),
        solvers={
            'ssp-rk': SSPRKSolver,
            'imex-euler': IMEXEulerSolver,
            'imex2': IMEX2Solver,
            'crank-nicolson': CrankNicolsonSolver,
            'etdrk4': ETDRK4Solver,
        },
        checks=CHECKS,
    ),
)
