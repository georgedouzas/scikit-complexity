"""Define the types of reactions, their stoichiometry and the table of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping

from ....core.specs import Role

type Stoichiometry = Mapping[tuple[Role, str] | str, float | Callable[..., object]]
type ReactionTable = Mapping[str, tuple[Mapping[str, float], Mapping[str, float], float]]
