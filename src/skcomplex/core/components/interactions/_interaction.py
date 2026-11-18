"""Define the interaction whose laws run once for the elements its roles bind."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace

from ...errors import ValidationError
from ...specs import DeclaredVariableSpec, RoleSpec
from ..base import BaseComponent
from ..spaces import BaseSpace
from ._base import BaseInteraction
from ._random_partition import RandomPartition

HYPEREDGES = 'hyperedges'
NEIGHBORHOODS = 'neighborhoods'
MIN_SIZE = 2


class Interaction(BaseInteraction):
    """An interaction whose laws run once for the elements its roles bind.

    When a role binds a group, the laws run once for every member of the group, as arrays. A subclass takes its roles
    and parameters as arguments of its `__init__`, annotated with `RoleSpec` or `ParameterSpec` inside `Annotated`, and
    returns its laws from `_get_laws`, which returns no laws by default.

    Unlike `BaseInteraction`, it fixes how the laws run: once per bound element or member, or once per edge with
    `along`. Choose it when a law reads the elements it binds, such as predation between two populations, or the two
    ends of a link, such as a contact that passes on an infection.

    It appears in many fields. Predation between two populations, a reaction between two chemical species, and a payment
    from a firm to its workers are each a subclass of it.

    With `among` and `along`, the laws run once for every edge of a network instead, and read `s.source` and
    `s.target`, the members at the two ends. A subclass lists both in its own `__init__` and passes them on, such as an
    infection along the contacts of a network, `Infection('infect', among='people', along=True, beta=0.3)`. When an
    element sits on those edges, such as `Element('links', values={'sign': 1}, space=network, on='edges')`, the laws
    read its member on the edge as `s.edge`, such as `s.edge.sign`.

    With `along='hyperedges'`, the laws run once for every hyperedge of the network and read `s.member`, the member
    the laws change, `s.others`, the other members in the order of the hyperedge, and `s.members`, all of them. They
    read the element on the hyperedges as `s.hyperedge`, or else the element on the edges as `s.edge`, the link that
    joins the first two members, as the regulated links of triadic percolation do. With `along=('hyperedges', 3)`,
    they run only on the hyperedges of three members, such as the triangles of a simplicial complex.

    With `along=RandomPartition(3)`, the members of the group `among` form random groups of three anew at every step,
    and the laws run once for every member of every group, reading `s.member`, `s.others`, and `s.members` as on a
    hyperedge, such as voters who discuss in small groups. With `along='neighborhoods'`, every member of a group on a
    lattice or a network forms one group with the members around it, such as the players of a public goods game with
    their neighbours. A member then belongs to several groups, and its updates from them add up.

    A rule used once needs no class. `roles` binds each role to an element and `laws` is a function of the namespace
    that returns the laws, such as gravity on a ball,
    `Interaction('gravity', roles={'target': 'ball'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), -9.81)])`.

    Args:
        label:
            The name of the interaction, unique among the components of its system, such as `'predation'`.

        roles:
            The element each role binds, by the name of the role, such as `{'target': 'ball'}`, for a rule written
            without a class, or `None`.

        laws:
            The function of the namespace that returns the laws of a rule written without a class, such as
            `lambda s: [RateLaw(s.target.n.diff(s.t), s.target.n)]`, the laws themselves when they read no element,
            or `None`.

        among:
            The group whose members sit at the ends of the edges, such as `'people'`, or `None` to bind elements by
            their roles only. A subclass annotates it as a role, and a rule written without a class binds it as one.

        along:
            `True` for the edges of the network the group lives on, the name of a layer of a network with several, such
            as `'contacts'`, `'hyperedges'` for its hyperedges, `('hyperedges', k)` for its hyperedges of `k`
            members, `RandomPartition(k)` for random groups of `k` members drawn at every step, `'neighborhoods'` for
            every member with the members around it, or `None` for no edges.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.specs import ParameterSpec, RoleSpec
        >>> from sage.all__sagemath_symbolics import oo
        >>> from sage.sets.real_set import RealSet
        >>> class PredationInteraction(Interaction):
        ...     def __init__(
        ...         self,
        ...         label,
        ...         predator: Annotated[str, RoleSpec()],
        ...         prey: Annotated[str, RoleSpec()],
        ...         a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        ...     ):
        ...         super().__init__(label)
        ...         self.predator = predator
        ...         self.prey = prey
        ...         self.a = a
        >>> PredationInteraction('predation', predator='foxes', prey='rabbits')
        PredationInteraction(label='predation', predator='foxes', prey='rabbits')

        Growth of a population, written without a class:

        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.systems import System
        >>> growth = Interaction(
        ...     'growth',
        ...     roles={'target': 'rabbits'},
        ...     laws=lambda s: [RateLaw(s.target.n.diff(s.t), 0.5 * s.target.n)],
        ... )
        >>> system = System(elements=[Element('rabbits', values={'n': 10.0})], interactions=[growth]).prepare()
        >>> system.equations_
        {'rabbits.n': '0.5*rabbits.n'}
    """

    def __init__(
        self,
        label: str,
        *,
        roles: Mapping[str, object] | None = None,
        laws: Callable[[SimpleNamespace], Sequence[BaseComponent]] | Sequence[BaseComponent] | None = None,
        among: str | None = None,
        along: bool | str | tuple[str, int] | RandomPartition | None = None,
    ) -> None:
        super().__init__(label)
        self.roles = roles
        self.laws = laws
        self.among = among
        self.along = along

    def _get_declared(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the numeric arguments as parameters, and `roles` and `among` as roles."""
        declared = dict(super()._get_declared())
        for name, binding in (self.roles or {}).items():
            declared[name] = DeclaredVariableSpec(binding, RoleSpec())
        if self.among is not None and 'among' not in declared and 'among' not in self._get_specs():
            declared['among'] = DeclaredVariableSpec(self.among, RoleSpec())
        return declared

    def _get_binding(self, role: str) -> object:
        """Return what a role binds, from `roles` or from the argument of its name.

        Args:
            role:
                The name of the role.

        Returns:
            binding:
                The label, labels, or entries the role binds, or `None` when it binds nothing.
        """
        if self.roles is not None and role in self.roles:
            return self.roles[role]
        return super()._get_binding(role)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Check the interaction, or set the laws of `laws` for a binding.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the elements it binds.

        Raises:
            ValidationError:
                A role of `roles` takes the name of an argument, `laws` is neither a function nor a list of laws, or
                `along` names hyperedges of fewer than two members.
        """
        clashes = sorted(set(self.roles or {}) & {'label', 'roles', 'laws', 'among', 'along'})
        if clashes:
            error_msg = f'The role {clashes[0]!r} of {self.label!r} takes the name of an argument. Rename the role.'
            raise ValidationError(error_msg)
        if self.laws is not None and not callable(self.laws) and not isinstance(self.laws, list | tuple):
            error_msg = (
                f'The laws of {self.label!r} are {self.laws!r}. Give a function of the namespace that returns them, '
                'such as `laws=lambda s: [RateLaw(s.target.n.diff(s.t), s.target.n)]`.'
            )
            raise ValidationError(error_msg)
        if isinstance(self.along, tuple) and not self._is_sized(self.along):
            error_msg = (
                f'{self.label!r} runs along {self.along!r}. Give the hyperedges of one size as a whole number of 2 or '
                "more, such as `along=('hyperedges', 3)`."
            )
            raise ValidationError(error_msg)
        if isinstance(self.along, RandomPartition):
            self.along._check()
        grouped = isinstance(self.along, RandomPartition) or self.along == NEIGHBORHOODS
        if grouped and self.among is None:
            error_msg = f'{self.label!r} runs along groups of members. Name their group with `among`.'
            raise ValidationError(error_msg)
        super()._prepare(context)
        if isinstance(context, SimpleNamespace) and self.laws is not None:
            self.laws_ = list(self.laws(context) if callable(self.laws) else self.laws)

    @staticmethod
    def _is_sized(along: tuple[object, ...]) -> bool:
        """Return whether `along` names hyperedges of a whole size of at least two."""
        if len(along) != MIN_SIZE or along[0] != HYPEREDGES:
            return False
        size = along[1]
        return isinstance(size, int) and not isinstance(size, bool) and size >= MIN_SIZE

    def _get_draws(self) -> bool:
        """Return whether the interaction draws, as it does along random groups."""
        return super()._get_draws() or isinstance(self.along, RandomPartition)

    def _runs_on_edges(self) -> bool:
        """Return whether the interaction runs along edges."""
        grouped = self._runs_on_hyperedges() or self._get_grouping() is not None
        return self.along is not None and self.along is not False and not grouped

    def _runs_on_hyperedges(self) -> bool:
        """Return whether the interaction runs along the hyperedges of a network."""
        return self.along == HYPEREDGES or (isinstance(self.along, tuple) and self.along[0] == HYPEREDGES)

    def _get_hyperedge_size(self) -> int | None:
        """Return the size `along` gives the hyperedges, or `None` for every size."""
        return int(self.along[-1]) if isinstance(self.along, tuple) and self._runs_on_hyperedges() else None

    def _get_layer(self) -> str | None:
        """Return the layer `along` names, or `None` for no layer."""
        named = isinstance(self.along, str) and self.along != NEIGHBORHOODS
        return self.along if named and not self._runs_on_hyperedges() else None

    def _get_grouping(self) -> tuple[str, int, str] | None:
        """Return the kind, size, and reduction of the groups `along` names, or `None`."""
        if isinstance(self.along, RandomPartition):
            return ('partition', self.along.size, 'sum')
        if self.along == NEIGHBORHOODS:
            return ('neighborhood', 0, 'sum')
        return None
