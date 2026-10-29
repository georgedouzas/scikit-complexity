"""Define the types of systems biology: the rules of Boolean networks and their candidates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence

type Rule = Callable[..., bool]
type Candidates = Sequence[tuple[Sequence[int], float]]
