"""Define the interaction whose laws run once for the elements its roles bind."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace

from ...errors import ValidationError
from .._declared_variable import _DeclaredVariable
from .._role_spec import RoleSpec
from ..laws import BaseLaw
from ..spaces import RadiusNeighborhood
from ._base import BaseInteraction
from ._random_partition import RandomPartition

HYPEREDGES = 'hyperedges'
NEIGHBORHOODS = 'neighborhoods'
MIN_SIZE = 2
PAIR_ROLES = ('first', 'second')


class Interaction(BaseInteraction):
    """An interaction whose laws run once for the elements its roles bind.

    It states the laws that change the values of the elements its roles bind, such as gravity on a ball in
    `Interaction('gravity', roles={'target': 'ball'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), -9.81)])`. When a
    role binds a group, the laws run once for every member, as arrays. With `among` and `along`, they run once for every
    edge of a network and read `s.source` and `s.target`, the members at its two ends, or once for every hyperedge or
    random small group and read `s.member`, `s.others`, and `s.members`. With `within`, they run once for every pair of
    members closer than a radius and read `s.first` and `s.second`. A subclass takes its roles, typed `Role`, and its
    parameters as arguments of its `__init__` and sets `laws_` in its `_prepare`. It states predation between two
    populations, a reaction between two chemicals, and an infection along the contacts of a network.

    Read more in the user guide, under Interactions.

    Args:
        label:
            The name of the interaction, unique among the components of its system, such as `'predation'`.

        roles:
            The element each role binds, by the name of the role, such as `{'target': 'ball'}`, for a rule written
            without a class, or `None`.

        laws:
            The function of the namespace that returns the laws of a rule written without a class, such as
            `lambda s: [RateLaw(s.target.n.diff(s.t), s.target.n)]`, the laws themselves when they read no element, or
            `None`.

        among:
            The group whose members sit at the ends of the edges, such as `'people'`, or `None` to bind elements by
            their roles only. A subclass annotates it as a role, and a rule written without a class binds it as one.

        along:
            `True` for the edges of the network the group lives on, the name of a layer of a network with several, such
            as `'contacts'`, `'hyperedges'` for its hyperedges, `('hyperedges', k)` for its hyperedges of `k` members,
            `RandomPartition(k)` for random groups of `k` members drawn at every step, `'neighborhoods'` for every
            member with the members around it, or `None` for no edges. On the edges, the laws read the member of an
            element that sits on them as `s.edge`, and on the hyperedges as `s.hyperedge`.

        within:
            A `RadiusNeighborhood(r)` for laws that run once for every pair of members of the roles `first` and `second`
            closer than `r`, which read `s.first` and `s.second` and give the laws of both, such as the repulsion of
            walkers of two crowds, or `None` for no pairs. With `among`, both members come from that group. The laws may
            give the energy of the pair as `ObjectiveLaw(V)` in place of its forces, and each member then feels the
            force `-dV/dx` on its position `x`.

    See Also:
        BaseInteraction: The base of every interaction.
        RandomPartition: Groups the members of a group at random at every step.
        TrainingInteraction: Trains the parameters of a nested system.

    Examples:
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.components.laws import RateLaw
        >>> birth = Interaction('birth', roles={'target': 'fish'}, laws=lambda s: [RateLaw(s.target.n.diff(s.t), 1)])
        >>> birth.roles
        {'target': 'fish'}
    """

    def __init__(
        self,
        label: str,
        *,
        roles: Mapping[str, object] | None = None,
        laws: Callable[[SimpleNamespace], Sequence[BaseLaw]] | Sequence[BaseLaw] | None = None,
        among: str | None = None,
        along: bool | str | tuple[str, int] | RandomPartition | None = None,
        within: RadiusNeighborhood | None = None,
    ) -> None:
        super().__init__(label)
        self.roles = roles
        self.laws = laws
        self.among = among
        self.along = along
        self.within = within

    def _get_declared(self) -> Mapping[str, _DeclaredVariable]:
        """Return the numeric arguments as parameters, and `roles` and `among` as roles.

        With `within`, `among` binds both members of a pair, `first` and `second`.
        """
        declared = dict(super()._get_declared())
        for name, binding in (self.roles or {}).items():
            declared[name] = _DeclaredVariable(binding, RoleSpec())
        if self.among is not None and self.within is not None:
            for name in PAIR_ROLES:
                declared.setdefault(name, _DeclaredVariable(self.among, RoleSpec()))
        elif self.among is not None and 'among' not in declared and 'among' not in self._get_specs():
            declared['among'] = _DeclaredVariable(self.among, RoleSpec())
        return declared

    @classmethod
    def _get_state_arguments(cls, role: RoleSpec) -> tuple[str, ...]:
        """Return the arguments that name the states a role reads.

        The mapping `roles` of bindings names none.
        """
        return tuple(name for name in super()._get_state_arguments(role) if name != 'roles')

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
        if self.among is not None and self.within is not None and role in PAIR_ROLES:
            return self.among
        return super()._get_binding(role)

    def _check_arguments(self) -> None:
        """Raise an error for roles, laws, or groups the interaction cannot take.

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
                f'The laws of {self.label!r} are {self.laws!r}. Give a function that takes the bound elements as '
                '`s` and returns the laws, such as `laws=lambda s: [RateLaw(s.target.n.diff(s.t), s.target.n)]`.'
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
        super()._check_arguments()

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the laws of `laws` for a binding.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        if self.laws is not None:
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

    def _get_within(self) -> RadiusNeighborhood | None:
        """Return the neighbourhood of the pairs the laws run for, or `None` for no pairs."""
        return self.within

    def _get_layer(self) -> str | None:
        """Return the layer `along` names, or `None` for no layer."""
        if not isinstance(self.along, str) or self.along == NEIGHBORHOODS or self._runs_on_hyperedges():
            return None
        return self.along

    def _get_grouping(self) -> tuple[str, int, str] | None:
        """Return the kind, size, and reduction of the groups `along` names, or `None`."""
        if isinstance(self.along, RandomPartition):
            return ('partition', self.along.size, 'sum')
        if self.along == NEIGHBORHOODS:
            return ('neighborhood', 0, 'sum')
        return None
