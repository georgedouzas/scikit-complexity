"""Define specs, the frozen values that the classes of the library read and write.

A spec declares what a class reads, such as the domain of a state, records what a run produced, such as an event, or
holds what a compiler built.
"""

from ._base_spec import BaseSpec
from ._declared_variable_spec import DeclaredVariableSpec
from ._entry_spec import EntrySpec
from ._matrix_group_spec import MatrixGroupSpec
from ._parameter_spec import ParameterSpec
from ._power_series_spec import PowerSeriesSpec
from ._role_spec import RoleSpec, Spec
from ._state_spec import StateSpec
from ._subshift_spec import SubshiftSpec
from ._tensor_spec import TensorSpec
from ._variable_spec import VariableSpec
from ._visual_spec import VisualSpec

__all__ = [
    'BaseSpec',
    'DeclaredVariableSpec',
    'EntrySpec',
    'MatrixGroupSpec',
    'ParameterSpec',
    'PowerSeriesSpec',
    'RoleSpec',
    'Spec',
    'StateSpec',
    'SubshiftSpec',
    'TensorSpec',
    'VariableSpec',
    'VisualSpec',
]
