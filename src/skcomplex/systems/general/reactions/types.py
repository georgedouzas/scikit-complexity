"""Define the types of the reactions.

The types are the stoichiometry of a reaction and the table of the reactions of a network.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping

type Stoichiometry = Mapping[tuple[str, str] | str, float | Callable[..., object]]
type ReactionTable = Mapping[str, tuple[Mapping[str, float], Mapping[str, float], float]]
