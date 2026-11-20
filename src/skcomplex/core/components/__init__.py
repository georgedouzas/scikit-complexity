"""Define the components: elements, interactions, laws, spaces, and systems, with the values they read.

An element holds states, an interaction reads them through roles and returns laws, a law is one equation or rule for one
state, a space places the members of a group, and a system composes them.
"""

from ._entry_spec import EntrySpec
from ._matrix_group_spec import MatrixGroupSpec
from ._observable_spec import Observable
from ._power_series_spec import PowerSeriesSpec
from ._role_spec import RoleSpec
from ._subshift_spec import SubshiftSpec
from ._tags import Tags
from ._tensor_spec import TensorSpec

__all__: list[str] = [
    'EntrySpec',
    'MatrixGroupSpec',
    'Observable',
    'PowerSeriesSpec',
    'RoleSpec',
    'SubshiftSpec',
    'Tags',
    'TensorSpec',
]
