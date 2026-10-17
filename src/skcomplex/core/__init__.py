"""Shared vocabulary of the framework: errors, domains, declarations, forms, and registries."""

from ._domains import Domain
from ._errors import (
    AnalysisError,
    BackendError,
    IncompatibleLawsError,
    LabelError,
    ParameterError,
    RoleError,
    SimulationError,
    SkcomplexError,
    SolverError,
    SpaceError,
    ValidationError,
)
from ._forms import Form
from ._registry import Registry
from ._variables import Variable, VariableKind, declarations_of, parameter, role, state
from ._visuals import VisualSpec

__all__: list[str] = [
    'AnalysisError',
    'BackendError',
    'Domain',
    'Form',
    'IncompatibleLawsError',
    'LabelError',
    'ParameterError',
    'Registry',
    'RoleError',
    'SimulationError',
    'SkcomplexError',
    'SolverError',
    'SpaceError',
    'ValidationError',
    'Variable',
    'VariableKind',
    'VisualSpec',
    'declarations_of',
    'parameter',
    'role',
    'state',
]
