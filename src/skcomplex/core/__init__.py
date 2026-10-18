"""Shared vocabulary of the framework: errors, domains, declarations, forms, and registries."""

from ._domains import AnyDomain, Categorical, Domain
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
from ._symbolic import (
    Expression,
    SymbolicVector,
    components_of,
    declare_domains,
    domain_of,
    symbol,
    symbols_of,
    vector,
)
from ._variables import Variable, VariableKind, declarations_of, parameter, role, state
from ._visuals import VisualSpec

__all__: list[str] = [
    'AnalysisError',
    'AnyDomain',
    'BackendError',
    'Categorical',
    'Domain',
    'Expression',
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
    'SymbolicVector',
    'ValidationError',
    'Variable',
    'VariableKind',
    'VisualSpec',
    'components_of',
    'declarations_of',
    'declare_domains',
    'domain_of',
    'parameter',
    'role',
    'state',
    'symbol',
    'symbols_of',
    'vector',
]
