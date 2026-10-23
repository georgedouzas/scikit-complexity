"""Lattice dynamics: local rules, drives, relaxations, extremal updates, Monte Carlo sampling, and their steppers."""

from ...base import Form
from .._formalism import Formalism, register
from ._laws import Drive, Energy, Extremal, LocalRule, Move, Relaxation, Sample, Sampler
from ._passes import CHECKS, OWNER_RULES, STEPWISE_FORMS, LatticeProblemPass, UnpackSamplesPass
from ._solvers import LatticeStepper, MonteCarlo
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
    'Drive',
    'Energy',
    'Extremal',
    'ExtremalRecord',
    'LatticeProblemPass',
    'LatticeState',
    'LatticeStep',
    'LatticeStepper',
    'LocalRule',
    'MonteCarlo',
    'Move',
    'Relaxation',
    'RelaxationRecord',
    'Sample',
    'Sampler',
    'UnpackSamplesPass',
]

register(
    Formalism(
        'lattices',
        laws=(LocalRule, Drive, Relaxation, Energy, Move, Sampler, Sample, Extremal),
        passes=(UnpackSamplesPass(), LatticeProblemPass()),
        forms=(Form.LATTICE, Form.MONTE_CARLO),
        stepwise_forms=STEPWISE_FORMS,
        owners=OWNER_RULES,
        solvers={'lattice': LatticeStepper, 'monte-carlo': MonteCarlo},
        defaults=(LatticeStepper, MonteCarlo),
        checks=CHECKS,
    ),
)
