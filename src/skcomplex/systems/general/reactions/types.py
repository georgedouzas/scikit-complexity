"""Define the types of reactions, their stoichiometry and the table of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping

type Stoichiometry = Mapping[tuple[str, str] | str, float | Callable[..., object]]
type ReactionTable = Mapping[str, tuple[Mapping[str, float], Mapping[str, float], float]]
