"""Define the components: elements, interactions, laws, spaces, times, and systems.

An element holds states, an interaction reads them through roles and returns laws, a law is one equation or rule for one
state, a space places the members of a group, a time says whether the system flows or steps, and a system composes them.
"""

from ._observable_spec import Observable
from ._tags import Tags

__all__: list[str] = [
    'Observable',
    'Tags',
]
