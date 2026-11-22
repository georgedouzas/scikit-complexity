"""Define the components of a system and the specs of the values they read.

An element holds values, such as `Element('ball', values={'x': 10.0, 'm': 1.0})`, an interaction reads them through its
roles and states its laws in `laws_` when it is prepared, a law is one equation or rule for one value, and a space
places the members of a group. The specs here name observables, tensors, and the other values components read.
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
