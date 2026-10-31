"""Define coupling, which gives every member a reduction over other members of a function of both of them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import SR, Expression, oo
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, RateLaw, UpdateLaw
from ..laws.utils import count, delayed, neighbors, shift_member, total
from ..spaces import BaseNeighborhood
from ..types import Combination, NamespaceFunction, Target
from ._neighbor import NeighborInteraction

MODES = ('edges', 'group', 'next')
REDUCTIONS = ('sum', 'mean')
FORMS = ('rate', 'map')
ENDS = ('source', 'target')


class CouplingInteraction(NeighborInteraction):
    """Coupling of every member to other members, through a reduction over them of a function of both members.

    Every member `i` gets the reduction, a sum or a mean, over other members `j` of `function(namespace)`, where
    `namespace.member` reads the states of `i`, `namespace.other` those of `j`, `namespace.edge` the member of the
    edge group that joins them, or `None`, and the namespace also holds the parameters. The function is the difference
    `x_j - x_i` of the state `read` by default. The members `j` are, by `mode`:

    - `'edges'`: the neighbours of `i` along the edges of its lattice or network, or within the `neighborhood`.
    - `'group'`: every other member of the group.
    - `'next'`: the next member of the group in member order, the first after the last.

    With `include_self`, the reduction also runs over `j = i`. A weight multiplies every term: on edges, the state
    `weights` of the edge group, which a subclass binds with a second role, and over the group, the entry `W_ij` of the
    matrix `weights`. The state `read` of `j` is read `delay` earlier, and the one of `i` shifted by the phase lag
    `lag`, when they are not 0.

    In the form `'rate'` the reduction adds to the rate of the state `write`, and in the form `'map'` it sets the next
    value of `write` every step of discrete time to `combine(x_i, reduction)`, the reduction itself by default. The
    rate on the edges of a lattice or network runs once per edge along both directions, and every other coupling runs
    once per member.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        nodes:
            The path of the group of the members.

        read:
            The name of the state that the default function, the delay, and the lag read.

        write:
            The name of the state that the coupling changes.

        function:
            The term of one other member, a callable of the namespace, or `None` for the difference of `read`.

        reduce:
            `'sum'` or `'mean'`.

        mode:
            `'edges'`, `'group'`, or `'next'`.

        weights:
            The name of the state of the edge group in the mode `'edges'`, a matrix with one row per member and one
            column per other member in the mode `'group'`, or `None` for no weights.

        include_self:
            Whether the reduction also runs over the member itself.

        lag:
            The phase lag added to the state `read` of the member.

        delay:
            The delay of the state `read` of the other member, 0 for none.

        form:
            `'rate'` or `'map'`.

        combine:
            The next value in the form `'map'`, a callable of the state of the member and the reduction, or `None` for
            the reduction.

        neighborhood:
            The neighbourhood of the mode `'edges'`, such as `RadiusNeighborhood(1.0)` in a continuous space, or `None`
            for the edges of the lattice or network.

        layer:
            The layer of the network whose edges the interaction runs on, or `None` for the graph of the network.

    Raises:
        ValidationError:
            A variant is unknown, the function or the combination is not callable, or the weights or the neighbourhood
            do not suit the mode and the form, raised by `check`.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import sin, var
        >>> from skcomplex.core.interactions import CouplingInteraction
        >>> source, target = SimpleNamespace(theta=var('a')), SimpleNamespace(theta=var('b'))
        >>> sine = lambda s: sin(s.other.theta - s.member.theta)
        >>> coupling = CouplingInteraction('coupling', nodes='oscillators', read='theta', write='theta', function=sine)
        >>> [law.value for law in coupling.check().prepare().run(SimpleNamespace(source=source, target=target)).laws_]
        [sin(a - b), sin(-a + b)]
    """

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        read: str = 'x',
        write: str = 'x',
        function: NamespaceFunction | None = None,
        reduce: str = 'sum',
        mode: str = 'edges',
        weights: str | npt.ArrayLike | scipy.sparse.sparray | None = None,
        include_self: bool = False,
        lag: Annotated[float, ParameterSpec()] = 0.0,
        delay: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        form: str = 'rate',
        combine: Combination | None = None,
        neighborhood: BaseNeighborhood | None = None,
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.nodes = nodes
        self.read = read
        self.write = write
        self.function = function
        self.reduce = reduce
        self.mode = mode
        self.weights = weights
        self.include_self = include_self
        self.lag = lag
        self.delay = delay
        self.form = form
        self.combine = combine
        self.neighborhood = neighborhood

    def _check_variants(self) -> None:
        """Raise an error for an unknown mode, reduction, or form, or a function or combination that is not callable."""
        for name, value, known in (('mode', self.mode, MODES), ('reduce', self.reduce, REDUCTIONS)):
            if value not in known:
                error_msg = f'The {name} of coupling is {value!r}. Use one of {", ".join(known)}.'
                raise ValidationError(error_msg)
        if self.form not in FORMS:
            error_msg = f'The form of coupling is {self.form!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)
        for role, item in (('function', self.function), ('combine', self.combine)):
            if item is not None and not callable(item):
                error_msg = f'The {role} of coupling must be callable or None, but it is {item!r}.'
                raise ValidationError(error_msg)

    def _check_structure(self) -> None:
        """Raise an error for weights or a neighbourhood that do not suit the mode and the form."""
        if self.neighborhood is not None and (
            self.mode != 'edges' or not isinstance(self.neighborhood, BaseNeighborhood)
        ):
            error_msg = f'The neighbourhood {self.neighborhood!r} of coupling needs a neighbourhood in the mode edges.'
            raise ValidationError(error_msg)
        if isinstance(self.weights, str) and not self._is_per_edge():
            error_msg = (
                f'The weights {self.weights!r} of coupling name a state of the edges, which only the rate on the edges '
                'of a lattice or network reads. Give a matrix in the mode group.'
            )
            raise ValidationError(error_msg)
        if self.weights is not None and not isinstance(self.weights, str) and self.mode != 'group':
            error_msg = 'A matrix of weights of coupling needs the mode group. Name an edge state on the edges.'
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for unknown variants, functions that cannot be called, or weights that do not suit the mode.

        Raises:
            ValidationError:
                A variant is unknown, the function or the combination is not callable, or the weights or the
                neighbourhood do not suit the mode and the form.
        """
        super()._check()
        self._check_variants()
        self._check_structure()

    def _is_per_edge(self) -> bool:
        """Return whether the coupling is a rate written once per edge of a lattice or network."""
        return self.mode == 'edges' and self.form == 'rate' and self.neighborhood is None

    def _get_term(
        self,
        namespace: SimpleNamespace,
        member: SimpleNamespace,
        other: SimpleNamespace,
        edge: SimpleNamespace | None = None,
    ) -> Target:
        """Return the term of one other member, with the delay and the lag applied to the read state."""
        read_member, read_other = getattr(member, self.read), getattr(other, self.read)
        if self.delay != 0:
            read_other = delayed(read_other, namespace.delay)
        if self.lag != 0:
            read_member = read_member + namespace.lag
        if self.function is None:
            return read_other - read_member
        parameters = {name: value for name, value in vars(namespace).items() if name not in {*ENDS, 'edge'}}
        return self.function(
            SimpleNamespace(
                **parameters,
                member=SimpleNamespace(**{**vars(member), self.read: read_member}),
                other=SimpleNamespace(**{**vars(other), self.read: read_other}),
                edge=edge,
            ),
        )

    def _get_matrix(self) -> npt.ArrayLike | scipy.sparse.sparray | None:
        """Return the matrix of weights, with a zero diagonal unless the member is included."""
        if self.weights is None or self.include_self:
            return self.weights
        if scipy.sparse.issparse(self.weights):
            matrix = scipy.sparse.csr_array(self.weights, dtype=float)
            matrix.setdiag(0)
            matrix.eliminate_zeros()
            return matrix
        matrix = np.array(self.weights, dtype=float)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    @staticmethod
    def _map(value: Target, operation: Callable[[Expression], Expression]) -> Target:
        """Return an operation applied to an expression, or to every component of a vector."""
        if isinstance(value, FreeModuleElement):
            return value.parent()([operation(SR(item)) for item in value])
        return operation(SR(value))

    @staticmethod
    def _shifted(member: SimpleNamespace) -> SimpleNamespace:
        """Return the namespace of the next member of the group in member order."""
        return SimpleNamespace(
            **{
                name: CouplingInteraction._map(value, lambda item: shift_member(item, 1))
                for name, value in vars(member).items()
                if isinstance(value, Expression | FreeModuleElement)
            },
        )

    def _get_group_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over every other member of the group."""
        source, target = namespace.source, namespace.target
        matrix = self._get_matrix()
        reduction = self._map(
            self._get_term(namespace, target, source),
            lambda item: total(item, over=source, weights=matrix),
        )
        if not self.include_self and self.weights is None:
            reduction = reduction - self._get_term(namespace, target, target)
        if self.reduce == 'mean':
            reduction = reduction / (count(over=source) - (0 if self.include_self else 1))
        return reduction

    def _get_next_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over the next member of the group."""
        target = namespace.target
        reduction = self._get_term(namespace, target, self._shifted(target))
        if self.include_self:
            reduction = reduction + self._get_term(namespace, target, target)
        return reduction / 2 if self.reduce == 'mean' and self.include_self else reduction

    def _get_neighbor_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over the neighbours of the member."""
        source, target = namespace.source, namespace.target
        term = self._get_term(namespace, target, source)
        if not self.include_self:
            return self._map(term, lambda item: neighbors(item, over=self.neighborhood, reduce=self.reduce))
        reduction = self._map(term, lambda item: neighbors(item, over=self.neighborhood)) + self._get_term(
            namespace,
            target,
            target,
        )
        if self.reduce == 'sum':
            return reduction
        return reduction / (neighbors(SR(1), over=self.neighborhood, reduce='count') + 1)

    def _get_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction that one member gets, by mode."""
        if self.mode == 'group':
            return self._get_group_reduction(namespace)
        if self.mode == 'next':
            return self._get_next_reduction(namespace)
        return self._get_neighbor_reduction(namespace)

    def _get_edge_term(self, namespace: SimpleNamespace, member: SimpleNamespace, other: SimpleNamespace) -> Target:
        """Return the term one end of an edge gets from the other end, weighted and divided for a mean."""
        edge = getattr(namespace, 'edge', None)
        term = self._get_term(namespace, member, other, edge)
        if isinstance(self.weights, str):
            term = term * getattr(edge, self.weights)
        if self.reduce == 'mean':
            term = term / (member.degree + int(self.include_self))
        return term

    def _get_edge_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of both ends of an edge, with the rate of the member itself when it is included."""
        source, target = namespace.source, namespace.target
        laws: list[BaseLaw] = [
            RateLaw(getattr(target, self.write), self._get_edge_term(namespace, target, source)),
            RateLaw(getattr(source, self.write), self._get_edge_term(namespace, source, target)),
        ]
        if self.include_self:
            own = self._get_term(namespace, target, target)
            divisor = target.degree + 1 if self.reduce == 'mean' else 1
            laws.append(RateLaw(getattr(target, self.write), own / divisor))
        return laws

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of both ends of every edge, or the rate or next value of every member.

        Args:
            namespace:
                The namespace of an edge, with the ends `source` and `target`.

        Returns:
            laws:
                The laws of the coupling.
        """
        if self._is_per_edge():
            return self._get_edge_laws(namespace)
        target = namespace.target
        reduction = self._get_reduction(namespace)
        if self.form == 'rate':
            return [RateLaw(getattr(target, self.write), reduction)]
        state = getattr(target, self.write)
        return [UpdateLaw(state, reduction if self.combine is None else self.combine(state, reduction))]
