"""Kinetic theory of active particles: encounter laws, moments, activity levels, and direct simulation Monte Carlo."""

from .._formalism import Formalism, register
from ._checks import probability_issues
from ._components import distribution_from, members_from
from ._contractions import CompiledEncounter, TableValues
from ._laws import DestructionLaw, EncounterLaw, ProliferationLaw, outcome_sums
from ._moments import moment
from ._passes import (
    CompileEncountersPass,
    EncounterFunctionsPass,
    EncounterJacobianPass,
    EncounterProblemPass,
    KineticProblem,
)
from ._solvers import KineticSolver

__all__: list[str] = [
    'CompiledEncounter',
    'DestructionLaw',
    'EncounterLaw',
    'KineticProblem',
    'KineticSolver',
    'ProliferationLaw',
    'TableValues',
    'distribution_from',
    'members_from',
    'moment',
    'outcome_sums',
]

register(
    Formalism(
        'kinetics',
        laws=(EncounterLaw, ProliferationLaw, DestructionLaw),
        passes=(CompileEncountersPass(), EncounterFunctionsPass(), EncounterJacobianPass(), EncounterProblemPass()),
        solvers={'particles': KineticSolver},
        checks=(probability_issues,),
    ),
)
