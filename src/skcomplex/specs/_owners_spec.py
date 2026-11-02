"""Define the owners spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _OwnersSpec(BaseSpec):
    """Law kinds of the owners of a system as declared, and law classes as lowered, with the solvers to run them."""

    declared: dict[str, set[type]]
    lowered: dict[str, set[type]]
    solvers: list[type]

    def under(self, scope: str, *, nested: bool) -> tuple[dict[str, set[type]], dict[str, set[type]]]:
        """Return the declared kinds and lowered classes of the owners of a scope, or of its own components alone."""

        def holds(owner: str) -> bool:
            return owner.startswith(scope) and (nested or '/' not in owner[len(scope) :])

        return (
            {owner: kinds for owner, kinds in self.declared.items() if holds(owner)},
            {owner: kinds for owner, kinds in self.lowered.items() if holds(owner)},
        )
