"""Plan the partitions of a system: the subsystems whose laws compile to one form that one solver runs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ...core.errors import IncompatibleLawsError
from ...core.system import System
from ..problems import Form
from ._census import Census
from ._forms import fits_form
from ._issues import Issue


def _solved(kinds: set[type], solvers: list[Any]) -> bool:
    """Check whether the solver of a single form passed runs every law kind of a set, or whether none is passed.

    Co-simulation is left out, since it runs partitions that each need a solver of their own.
    """
    single = [solver for solver in solvers if solver.form != Form.COUPLED]
    return not single or any(kinds <= set(solver.supports.laws) for solver in single)


def _listed(owners: Mapping[str, Iterable[object]]) -> str:
    """Return the components of every law kind or class as one list."""
    classes: dict[str, list[str]] = {}
    for owner, kinds in owners.items():
        for kind in kinds:
            classes.setdefault(getattr(kind, '__name__', f'{kind}'), []).append(owner)
    return '; '.join(f'{kind}: {", ".join(sorted(set(paths)))}' for kind, paths in sorted(classes.items()))


def _mixed_issue(
    path: str,
    declared: Mapping[str, set[type]],
    lowered: Mapping[str, set[type]],
    solvers: list[Any],
) -> Issue:
    """Return the issue of a system whose own laws no solver runs, or that compile to no form, by law kind or class."""
    fix = 'Move the components of each law kind into their own nested system, so they run as coupled partitions.'
    if not _solved(set().union(*declared.values()), solvers):
        message = f'No solver runs the laws of this system together ({_listed(declared)}).'
    else:
        message = f'No form takes the laws of this system together ({_listed(lowered)}).'
    return Issue('laws-mixed', path, message, fix, error=IncompatibleLawsError)


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
        return fits_form(lowered.values()) and _solved(set().union(*declared.values()), self.solvers)


def _planned(system: System, owners: _Owners, scope: str = '') -> tuple[list['_PlannedPartition'], list[Issue]]:
    """Return the partitions of a system tree and the issues of the subsystems whose own laws fit no form."""
    name = scope.rstrip('/') or system.label
    declared, lowered = owners.under(scope, nested=True)
    if owners.fit(declared, lowered):
        return ([_PlannedPartition(name, scope, nested=True)] if declared or lowered else []), []
    own_declared, own_lowered = owners.under(scope, nested=False)
    if not system.systems or not owners.fit(own_declared, own_lowered):
        mixed = (own_declared, own_lowered) if system.systems else (declared, lowered)
        return [], [_mixed_issue(name, *mixed, owners.solvers)]
    partitions = [_PlannedPartition(name, scope, nested=False)] if own_declared or own_lowered else []
    issues: list[Issue] = []
    for child in system.systems:
        found, mixed_issues = _planned(child, owners, f'{scope}{child.label}/')
        partitions += found
        issues += mixed_issues
    return partitions, issues


def _plan_partitions(
    census: Census,
    kinds: Mapping[str, set[type]],
    solvers: Iterable[Any] | None = None,
) -> '_Plan':
    """Return the partitions of a system, the subsystems whose laws compile to one form that one solver runs."""
    declared: dict[str, set[type]] = {}
    for record in census.records:
        declared.setdefault(record.owner, set()).add(type(record.law))
    found = list(solvers or ())
    partitions, issues = _planned(census.system, _Owners(declared, dict(kinds), found))
    return _Plan(tuple(partitions), tuple(issues))


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
