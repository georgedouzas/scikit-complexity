"""Plan the partitions of a system: the subsystems whose laws compile to one form that one solver runs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..problems import Form
from ._forms import _COMBINED, _FORMS
from ._issues import Issue
from ._laws import _FORMLESS_LAWS


@dataclass(frozen=True)
class _Owners:
    """Law kinds of the owners of a system as declared, and law classes as lowered, with the solvers to run them."""

    declared: dict[str, set[type]]
    lowered: dict[str, set[type]]
    solvers: list[Any]

    def under(self, scope: str, *, nested: bool) -> tuple[dict[str, set[type]], dict[str, set[type]]]:
        """Return the declared kinds and lowered classes of the owners of a scope, or of its own components alone."""

        def holds(owner: str) -> bool:
            return owner.startswith(scope) and (nested or '/' not in owner[len(scope) :])

        return (
            {owner: kinds for owner, kinds in self.declared.items() if holds(owner)},
            {owner: kinds for owner, kinds in self.lowered.items() if holds(owner)},
        )

    def fit(self, declared: dict[str, set[type]], lowered: dict[str, set[type]]) -> bool:
        """Check whether laws compile to one form and one solver runs them."""
        return self._fits_form(lowered.values()) and self._solved(set().union(*declared.values()), self.solvers)

    @staticmethod
    def _fits_form(owners: Iterable[set[type]]) -> bool:
        """Check whether the law classes of components compile to one form.

        Args:
            owners:
                The law classes of every component.

        Returns:
            fits:
                Whether a form takes the law classes.
        """
        return _Owners._form_kinds(owners) in _FORMS

    @staticmethod
    def _solved(kinds: set[type], solvers: list[Any]) -> bool:
        """Check whether the solver of a single form passed runs every law kind of a set, or whether none is passed.

        Co-simulation is left out, since it runs partitions that each need a solver of their own.
        """
        single = [solver for solver in solvers if solver.form != Form.COUPLED]
        return not single or any(not solver.supports.missing('laws', kinds) for solver in single)

    @staticmethod
    def _form_kinds(owners: Iterable[set[type]]) -> frozenset[type]:
        """Return the law classes that decide the form of components with their own law classes.

        A combined class, such as a renewal, drops out of the classes when the rest are its companions and no component
        has laws of that class alone. A component with laws of that class alone next to components with other laws keeps
        it, so the system does not fit one form and runs as partitions.
        """
        owned = [{kind for kind in kinds if not issubclass(kind, _FORMLESS_LAWS)} for kinds in owners]
        kinds: set[type] = set().union(*owned)
        for kind, companions in _COMBINED.items():
            rest = kinds - {kind}
            if kind not in kinds or not rest:
                continue
            alone = any(item == {kind} for item in owned)
            if not alone and rest <= companions:
                kinds = rest
        return frozenset(kinds)


@dataclass(frozen=True)
class _PlannedPartition:
    """Subsystem that runs as one partition, by the scope of the laws it holds."""

    name: str
    scope: str
    nested: bool

    def holds(self, owner: str) -> bool:
        """Check whether the partition holds the laws of an element or group.

        Args:
            owner:
                The path of the element or group.

        Returns:
            holds:
                Whether the owner lies in the partition's subsystem, or in a system nested in it for a nested partition.
        """
        return owner.startswith(self.scope) and (self.nested or '/' not in owner[len(self.scope) :])


@dataclass(frozen=True)
class _Plan:
    """Partitions of a system and the issues of its law mixes that no form or solver takes."""

    partitions: tuple[_PlannedPartition, ...]
    issues: tuple[Issue, ...]
