"""Lattice dynamics: local rules, drives, relaxations, extremal updates, Monte Carlo sampling, and their steppers."""

from ...core.base import Form
from .._formalism import Formalism, register
from ._laws import DriveLaw, Energy, ExtremalLaw, LocalRuleLaw, MoveLaw, RelaxationLaw, SampleLaw, SamplerLaw
from ._passes import CHECKS, OWNER_RULES, STEPWISE_FORMS, LatticeProblemPass, UnpackSamplesPass
from ._solvers import LatticeSolver, MonteCarloSolver
from ._steps import (
    CompiledDrive,
    CompiledExtremal,
    CompiledMonteCarlo,
    CompiledRelaxation,
    CompiledRule,
    ExtremalRecord,
    LatticeState,
    LatticeStep,
    RelaxationRecord,
)

__all__: list[str] = [
    'CompiledDrive',
    'CompiledExtremal',
    'CompiledMonteCarlo',
    'CompiledRelaxation',
    'CompiledRule',
    'DriveLaw',
    'Energy',
    'ExtremalLaw',
    'ExtremalRecord',
    'LatticeProblemPass',
    'LatticeSolver',
    'LatticeState',
    'LatticeStep',
    'LocalRuleLaw',
    'MonteCarloSolver',
    'MoveLaw',
    'RelaxationLaw',
    'RelaxationRecord',
    'SampleLaw',
    'SamplerLaw',
    'UnpackSamplesPass',
]

register(
    Formalism(
        'lattices',
        laws=(LocalRuleLaw, DriveLaw, RelaxationLaw, Energy, MoveLaw, SamplerLaw, SampleLaw, ExtremalLaw),
        passes=(UnpackSamplesPass(), LatticeProblemPass()),
        forms=(Form.LATTICE, Form.MONTE_CARLO),
        stepwise_forms=STEPWISE_FORMS,
        owners=OWNER_RULES,
        solvers={'lattice': LatticeSolver, 'monte-carlo': MonteCarloSolver},
        defaults=(LatticeSolver, MonteCarloSolver),
        checks=CHECKS,
    ),
)
