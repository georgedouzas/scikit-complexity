"""Plan the partitions of a system: the subsystems whose laws compile to one form that one solver runs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from ..base import Form, IncompatibleLawsError, LawKind, Registry
from ..system import System
from ._backends import load_backends
from ._census import Census
from ._forms import fits_form
from ._issues import Issue


def solver_registry() -> Registry:
    """Return the registry of solvers, imported late since the solvers import the compiler.

    Returns:
        solvers:
            The registry of solver classes by form and name.
    """
    from ..solvers import SOLVERS  # noqa: PLC0415

    load_backends()

    return SOLVERS


def solver_classes() -> dict[Any, list[str]]:
    """Return every registered solver class with its names, without the `'auto'` aliases.

    Returns:
        classes:
            The names of every solver class.
    """
    classes: dict[Any, list[str]] = {}
    solvers = solver_registry()
    for key in solvers:
        _, name = cast('tuple[Form, str]', key)
        if name != 'auto':
            classes.setdefault(solvers.resolve(key), []).append(name)
    return classes


def _solved(kinds: set[LawKind], solvers: list[Any]) -> bool:
    """Check whether one solver of a single form runs every law kind of a set.

    Co-simulation is left out, since it runs partitions that each need a solver of their own.
    """
    return any(kinds <= solver.supports.laws for solver in solvers if solver.form is not Form.COUPLED)


def _listed(owners: Mapping[str, Iterable[object]]) -> str:
    """Return the components of every law kind or class as one list."""
    classes: dict[str, list[str]] = {}
    for owner, kinds in owners.items():
        for kind in kinds:
            classes.setdefault(getattr(kind, '__name__', f'{kind}'), []).append(owner)
    return '; '.join(f'{kind}: {", ".join(sorted(set(paths)))}' for kind, paths in sorted(classes.items()))


def _mixed_issue(
    path: str,
    declared: Mapping[str, set[LawKind]],
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

    declared: dict[str, set[LawKind]]
    lowered: dict[str, set[type]]
    solvers: list[Any]

    def under(self, scope: str, *, nested: bool) -> tuple[dict[str, set[LawKind]], dict[str, set[type]]]:
        """Return the declared kinds and lowered classes of the owners of a scope, or of its own components alone."""

        def holds(owner: str) -> bool:
            return owner.startswith(scope) and (nested or '/' not in owner[len(scope) :])

        return (
            {owner: kinds for owner, kinds in self.declared.items() if holds(owner)},
            {owner: kinds for owner, kinds in self.lowered.items() if holds(owner)},
        )

    def fit(self, declared: dict[str, set[LawKind]], lowered: dict[str, set[type]]) -> bool:
        """Check whether laws compile to one form and one solver runs them."""
        return fits_form(lowered.values()) and _solved(set().union(*declared.values()), self.solvers)


def _planned(system: System, owners: _Owners, scope: str = '') -> tuple[list['PlannedPartition'], list[Issue]]:
    """Return the partitions of a system tree and the issues of the subsystems whose own laws fit no form."""
    name = scope.rstrip('/') or system.label
    declared, lowered = owners.under(scope, nested=True)
    if owners.fit(declared, lowered):
        return ([PlannedPartition(name, scope, nested=True)] if declared or lowered else []), []
    own_declared, own_lowered = owners.under(scope, nested=False)
    if not system.systems or not owners.fit(own_declared, own_lowered):
        mixed = (own_declared, own_lowered) if system.systems else (declared, lowered)
        return [], [_mixed_issue(name, *mixed, owners.solvers)]
    partitions = [PlannedPartition(name, scope, nested=False)] if own_declared or own_lowered else []
    issues: list[Issue] = []
    for child in system.systems:
        found, mixed_issues = _planned(child, owners, f'{scope}{child.label}/')
        partitions += found
        issues += mixed_issues
    return partitions, issues


def plan_partitions(
    census: Census,
    kinds: Mapping[str, set[type]],
    solvers: Iterable[Any] | None = None,
) -> 'Plan':
    """Return the partitions of a system, the subsystems whose laws compile to one form that one solver runs.

    A system tree whose laws compile to one registered form, and whose law kinds one solver of a single form runs, is
    one partition. Otherwise the own laws of every system form a partition, and each child system is planned in turn. A
    system without children whose laws do not fit, or whose own laws do not fit, is an issue that names its components
    by law kind or class.

    Args:
        census:
            The census of the laws of the system, as declared.

        kinds:
            The law classes of every owner, as lowered.

        solvers:
            The solver classes, each with its form and the law kinds it supports, or `None` for the registered ones.

    Returns:
        plan:
            The partitions and the issues of the mixes no form or solver takes.

    Examples:
        >>> from skcomplex.core.compile import plan_partitions, validate
        >>> from skcomplex.models import create
        >>> compilation = validate(create('lotka_volterra')).compilation
        >>> plan = plan_partitions(compilation.census, compilation.kinds)
        >>> [(partition.name, partition.nested) for partition in plan.partitions]
        [('system', True)]
    """
    declared: dict[str, set[LawKind]] = {}
    for record in census.records:
        declared.setdefault(record.owner, set()).add(record.law.law_kind)
    found = list(solver_classes() if solvers is None else solvers)
    partitions, issues = _planned(census.system, _Owners(declared, dict(kinds), found))
    return Plan(tuple(partitions), tuple(issues))


@dataclass(frozen=True)
class PlannedPartition:
    """Subsystem that runs as one partition, by the scope of the laws it holds.

    Args:
        name:
            The name of the partition, the path of its subsystem or the label of the root system.

        scope:
            The path prefix of the subsystem, empty for the root system.

        nested:
            Whether the partition holds the laws of the systems nested in its subsystem too.
    """

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
class Plan:
    """Partitions of a system and the issues of its law mixes that no form or solver takes.

    Args:
        partitions:
            The partitions in tree order. A system that runs as one partition has one, or none without laws.

        issues:
            The issues of the subsystems whose laws no form or solver takes.
    """

    partitions: tuple[PlannedPartition, ...]
    issues: tuple[Issue, ...]
