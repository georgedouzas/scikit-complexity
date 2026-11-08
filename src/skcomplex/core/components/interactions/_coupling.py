"""Define coupling.

Coupling gives every member a reduction over other members of a function of both of them.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import SR, Expression, arctan2, cos, oo, sin
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import BaseSpec, ClampSpec, ParameterSpec, RadiusSpec, RoleSpec, Tags
from ...types import Array, ElementKind
from ..laws import BaseLaw, RateLaw, RuleLaw, UpdateLaw
from ..laws.utils import count, delayed, hyperedges, maximum, minimum, neighbors, shift_member, total
from ..spaces import BaseNeighborhood, BaseSpace
from ..types import AgentCombination, AgentViewProtocol, Combination, NamespaceFunction, SystemViewProtocol, Target
from ._neighbor import NeighborInteraction

MODES = ('edges', 'group', 'next', 'hyperedges')
MIN_SIZE = 2
REDUCTIONS = ('sum', 'mean', 'max', 'min', 'normalized-mean', 'circular-mean')
SUMS = ('sum', 'mean')
EXTREMES = ('max', 'min')
FORMS = ('rate', 'map', 'agent')
ENDS = ('source', 'target')

type Weights = str | npt.ArrayLike | scipy.sparse.sparray | NamespaceFunction | None
type Part = Callable[[Target, Target], Target]


class CouplingInteraction(NeighborInteraction):
    """Coupling of every member to other members.

    The coupling is a reduction over the other members of a function of both members. Every member `i` gets the
    reduction over other members `j` of `function(namespace)`, where `namespace.member` reads the states of `i`,
    `namespace.other` those of `j`, `namespace.edge` the member of the edge group that joins them, or `None`, and the
    namespace also holds the parameters and the time `t`. The function is the difference `x_j - x_i` of the state `read`
    by default. The members `j` are, by `mode`:

    - `'edges'`: the neighbours of `i` along the edges of its lattice or network, or within the `neighborhood`.
    - `'group'`: every other member of the group, or every member of the group `source` when it is given.

    In the mode `'group'` the roles may also bind separate elements. A target that lists elements couples every element
    to the other elements of the list, a single target element is a hub that reduces over every member of the group
    `source`, and a source that is one element or a list of elements couples every member of the target to them. The
    reduction over separate elements is written out term by term.
    - `'next'`: the next member of the group in member order, the first after the last.
    - `'hyperedges'`: the other members of every hyperedge around `i` of `size` members, or of every size, on a network
      with hyperedges, or with a `neighborhood` in a continuous space, every set of `size - 1` current neighbours of
      `i`. The function then reads `namespace.member`, `namespace.others`, the other members of one hyperedge in its
      order, and `namespace.members`, all its members, which reductions over them read, and the reduction runs over the
      hyperedges around `i`. The default function is the sum over the other members of the difference of `read`.

    With `include_self`, the reduction also runs over `j = i`. Every term `f_ij` has a weight `w_ij`: on edges, the
    state `weights` of the edge group, which a subclass binds with a second role, over the group, the entry `W_ij` of
    the matrix `weights` or of the shaped parameter of the interaction that `weights` names, which a subclass declares,
    and in every mode a callable `weights(namespace)` of the namespace of the term, such as an
    indicator of the states of both members. The reductions are `'sum'`, `sum_j w_ij f_ij`, `'mean'`, the sum divided
    by the number of members `j`, `'max'` and `'min'`, the largest and smallest `w_ij f_ij`, `'normalized-mean'`,
    `sum_j w_ij f_ij / sum_j w_ij`, and `'circular-mean'`, the angle `arctan2(sum_j w_ij sin f_ij, sum_j w_ij cos
    f_ij)`. The state `read` of `j` is read `delay` earlier, and the one of `i` shifted by the phase lag `lag`, when
    they are not 0.

    In the form `'rate'` the reduction adds to the rate of the state `write`, and in the form `'map'` it sets the next
    value of `write` every step of discrete time to `combine(namespace, reduction)`, where `namespace.member` reads the
    states of `i` and the namespace holds the parameters and the time, or to the reduction itself by default. The sum
    and the mean of the rate on the edges of a lattice or network run once per edge along both directions, and every
    other coupling runs once per member. On a `FieldElement`, a rate that is a constant times the difference of the
    ends becomes that constant times the field's Laplacian times its squared spacing, and on a lattice the other end is
    read across the field's boundary conditions, so beyond fixed edges it reads the fixed value.

    In the form `'agent'` the target is a group of agents, and every agent runs the coupling in its own activation in
    the phase `phase`. Every `RadiusSpec` of `radii` gives the vector `function(agent, other, system)` of every other
    agent of the source group, or of the target group, within its radius, reduced by its sum or mean. Only the radii
    of the highest priority that hold another agent count, and their reductions add up. The states `write` of the
    agent form a vector `v`, which becomes `combine(agent, system, reduction, priority)`, or `v` plus the reduction by
    default, where `reduction` and `priority` are `None` when no radius holds another agent. `clamp` then bounds the
    angle the vector turns and its norm.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of the members, or in the mode `'group'` the path of one element or the paths of
            separate elements.

        source:
            The path of the group, the element, or the paths of the separate elements of the other members in the mode
            `'group'`, or `None` for the members of the target.

        read:
            The name of the state that the default function, the delay, and the lag read.

        write:
            The name of the state that the coupling changes, or the names of the components of the vector of the agent
            form.

        function:
            The term of one other member, a callable of the namespace, or `None` for the difference of `read`.

        reduce:
            `'sum'`, `'mean'`, `'max'`, `'min'`, `'normalized-mean'`, or `'circular-mean'`.

        mode:
            `'edges'`, `'group'`, `'next'`, or `'hyperedges'`.

        size:
            The number of members of the hyperedges of the mode `'hyperedges'`, or `None` for every size.

        weights:
            The name of the state of the edge group in the mode `'edges'`, a matrix with one row per member and one
            column per other member or the name of a parameter of the interaction declared
            `ParameterSpec(shape=('target', 'source'))` in the mode `'group'`, a callable of the namespace of a term in
            any mode, or `None` for no weights.

        include_self:
            Whether the reduction also runs over the member itself.

        lag:
            The phase lag added to the state `read` of the member.

        delay:
            The delay of the state `read` of the other member, 0 for none.

        form:
            `'rate'`, `'map'`, or `'agent'`.

        combine:
            The next value in the form `'map'`, a callable of the namespace of the member and the reduction, or in the
            form `'agent'` a callable of the agent, the system view, the reduction, and its priority, or `None` for the
            default.

        neighborhood:
            The neighbourhood of the mode `'edges'`, such as `RadiusNeighborhood(1.0)` in a continuous space, or `None`
            for the edges of the lattice or network. In the mode `'hyperedges'`, the neighbourhood of a continuous
            space whose current neighbours form the hyperedges, or `None` for the hyperedges of the network.

        layer:
            The layer of the network whose edges the interaction runs on, or `None` for the graph of the network.

        radii:
            The terms of the agent form over the radii around an agent.

        clamp:
            The clamp of the turn and the norm of the vector of the agent form, or `None`.

        phase:
            The phase of the step the agent and the map forms run in, or `None` for the default phase.

    Raises:
        ValidationError:
            A variant is unknown, the function or the combination is not callable, the weights, the source, the
            reduction, or the neighbourhood do not suit the mode and the form, or the radii, the clamp, the phase, or
            the written states do not suit the form, raised by `prepare`.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import sin, var
        >>> from skcomplex.core.components.interactions import CouplingInteraction
        >>> source, target = SimpleNamespace(theta=var('a')), SimpleNamespace(theta=var('b'))
        >>> sine = lambda s: sin(s.other.theta - s.member.theta)
        >>> coupling = CouplingInteraction('coupling', target='oscillators', read='theta', write='theta', function=sine)
        >>> [law.value for law in coupling.prepare().run(SimpleNamespace(source=source, target=target)).laws_]
        [sin(a - b), sin(-a + b)]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw, RuleLaw, UpdateLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of the agent rules, the map updates, or the rates otherwise.

        These are the rules of the agent form, the updates of the map form, or the rates.
        """
        forms: dict[object, tuple[type[BaseLaw], ...]] = {'agent': (RuleLaw,), 'map': (UpdateLaw,)}
        return self._get_variant_tags(forms.get(self._get_argument('form'), (RateLaw,)))

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str | Sequence[str], RoleSpec(elements=(ElementKind.ANY,), states=('write',))],
        source: Annotated[str | Sequence[str] | None, RoleSpec(optional=True, elements=(ElementKind.ANY,))] = None,
        read: str = 'x',
        write: str | Sequence[str] = 'x',
        function: NamespaceFunction | None = None,
        reduce: str = 'sum',
        mode: str = 'edges',
        size: int | None = None,
        weights: Weights = None,
        include_self: bool = False,
        lag: Annotated[float, ParameterSpec()] = 0.0,
        delay: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        form: str = 'rate',
        combine: Combination | AgentCombination | None = None,
        neighborhood: BaseNeighborhood | None = None,
        layer: str | None = None,
        radii: Sequence[RadiusSpec] = (),
        clamp: ClampSpec | None = None,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.target = target
        self.source = source
        self.read = read
        self.write = write
        self.function = function
        self.reduce = reduce
        self.mode = mode
        self.size = size
        self.weights = weights
        self.include_self = include_self
        self.lag = lag
        self.delay = delay
        self.form = form
        self.combine = combine
        self.neighborhood = neighborhood
        self.radii = radii
        self.clamp = clamp
        self.phase = phase

    def _check_variants(self) -> None:
        """Raise an error for an unknown variant or a function that is not callable.

        The unknown variant is a mode, a reduction, or a form, and the function may also be the combination.
        """
        self._check_variant('mode', self.mode, MODES, (BaseSpec,))
        for name, value, known in (
            ('reduce', self.reduce, REDUCTIONS),
            ('form', self.form, FORMS),
        ):
            if value not in known:
                error_msg = f'The {name} of coupling is {value!r}. Use one of {", ".join(known)}.'
                raise ValidationError(error_msg)
        for role, item in (('function', self.function), ('combine', self.combine)):
            if item is not None and not callable(item):
                error_msg = f'The {role} of coupling must be callable or None, but it is {item!r}.'
                raise ValidationError(error_msg)

    def _check_reduction(self) -> None:
        """Raise an error for a source outside the mode group, or a bad reduction.

        The bad reduction is one that its mode cannot run.
        """
        if self.source is not None and self.mode != 'group':
            error_msg = f'The source {self.source!r} of coupling needs the mode group, but the mode is {self.mode!r}.'
            raise ValidationError(error_msg)
        if not isinstance(self.target, str) and self.mode != 'group':
            error_msg = (
                f'The coupling of the separate elements {self.target!r} needs the mode group, not {self.mode!r}.'
            )
            raise ValidationError(error_msg)
        if self.mode == 'next' and self.reduce not in SUMS:
            error_msg = f'The coupling to the next member reduces by sum or mean, not {self.reduce!r}.'
            raise ValidationError(error_msg)
        if self.reduce in EXTREMES and self.mode == 'group' and self._excludes_self() and isinstance(self.target, str):
            error_msg = (
                f'The {self.reduce} of coupling over its own group runs over every member. Pass include_self=True or '
                'a separate source group.'
            )
            raise ValidationError(error_msg)
        if self.reduce in EXTREMES and (self._has_matrix() or self._names_matrix()):
            error_msg = f'The {self.reduce} of coupling takes no matrix of weights. Give the weights as a callable.'
            raise ValidationError(error_msg)

    def _check_structure(self) -> None:
        """Raise an error for weights or a neighbourhood that do not suit the mode.

        They must also suit the form.
        """
        if self.neighborhood is not None and (
            self.mode not in {'edges', 'hyperedges'} or not isinstance(self.neighborhood, BaseNeighborhood)
        ):
            error_msg = (
                f'The neighbourhood {self.neighborhood!r} of coupling needs a neighbourhood in the mode edges or '
                'hyperedges.'
            )
            raise ValidationError(error_msg)
        if isinstance(self.weights, str) and not self._is_per_edge() and not self._names_matrix():
            error_msg = (
                f'The weights {self.weights!r} of coupling name a state of the edges, which only the sum or mean rate '
                'on the edges of a lattice or network reads, or a shaped parameter in the mode group. Give a matrix '
                'in the mode group, or a callable.'
            )
            raise ValidationError(error_msg)
        if self._names_matrix() and (self._excludes_self() or not isinstance(self.target, str) or self._is_separate()):
            error_msg = (
                f'The weights {self.weights!r} of coupling name a shaped parameter, whose matrix product runs over '
                'every member of a group. Pass include_self=True or a separate source group, and bind groups.'
            )
            raise ValidationError(error_msg)
        if self._has_matrix() and self.mode != 'group':
            error_msg = 'A matrix of weights of coupling needs the mode group. Name an edge state on the edges.'
            raise ValidationError(error_msg)

    def _check_hyperedges(self) -> None:
        """Raise an error for a size or a setting the mode hyperedges does not take.

        The size is outside the mode hyperedges.
        """
        if self.size is not None and (
            self.mode != 'hyperedges'
            or isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < MIN_SIZE
        ):
            error_msg = (
                f'The size {self.size!r} of coupling needs the mode hyperedges and a whole number of 2 or more, but '
                f'the mode is {self.mode!r}.'
            )
            raise ValidationError(error_msg)
        if self.mode == 'hyperedges' and (self.include_self or self._is_nonzero('delay') or self.form == 'agent'):
            error_msg = (
                'The coupling over hyperedges takes no include_self, no delay, and no agent form. Couple the member '
                'to itself with another interaction.'
            )
            raise ValidationError(error_msg)

    def _check_agent(self) -> None:
        """Raise an error for agent settings outside the agent form, or no radii.

        The settings are radii, a clamp, or several written states.
        """
        if self.form != 'agent':
            if self.radii or self.clamp is not None or not isinstance(self.write, str):
                error_msg = (
                    f'The coupling {self.label!r} takes radii, a clamp, and several written states in the agent form '
                    f'only, but its form is {self.form!r}.'
                )
                raise ValidationError(error_msg)
            if self.phase is not None and self.form != 'map':
                error_msg = (
                    f'The coupling {self.label!r} takes a phase in the agent and the map forms, but its form is '
                    f'{self.form!r}.'
                )
                raise ValidationError(error_msg)
            return
        if not self.radii or not isinstance(self.target, str):
            error_msg = f'The agent form of coupling {self.label!r} needs one group of agents and at least one radius.'
            raise ValidationError(error_msg)
        for spec in self.radii:
            spec._check()
        if self.clamp is not None:
            self.clamp._check()

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for unknown variants, bad functions, or unsuitable weights.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                A variant is unknown, the function or the combination is not callable, or the weights, the source, the
                reduction, or the neighbourhood do not suit the mode and the form.
        """
        super()._prepare(context)
        self._check_variants()
        self._check_reduction()
        self._check_structure()
        self._check_hyperedges()
        self._check_agent()

    def _get_written(self) -> str:
        """Return the name of the one state that the symbolic forms write."""
        return cast(str, self.write)

    def _has_matrix(self) -> bool:
        """Return whether the weights are a matrix."""
        return self.weights is not None and not isinstance(self.weights, str) and not callable(self.weights)

    def _names_matrix(self) -> bool:
        """Return whether the weights name a shaped parameter in the mode group."""
        return isinstance(self.weights, str) and self.mode == 'group'

    def _is_separate(self) -> bool:
        """Return whether the source lists separate elements."""
        return self.source is not None and not isinstance(self.source, str)

    def _excludes_self(self) -> bool:
        """Return whether a reduction over the own group leaves the member out."""
        return self.source is None and not self.include_self

    def _is_nonzero(self, name: str) -> bool:
        """Return whether a parameter has a value other than 0.

        It reads the parameter under the name `_get_parameter` returns.
        """
        return bool(getattr(self, self._get_parameter(name)) != 0)

    def _is_per_edge(self) -> bool:
        """Return whether the coupling is a rate written once per edge.

        The edges are those of a lattice or a network.
        """
        return self.mode == 'edges' and self.form == 'rate' and self.neighborhood is None and self.reduce in SUMS

    def _parameters(self, namespace: SimpleNamespace) -> dict[str, object]:
        """Return the parameters and the time of a namespace.

        It leaves out the ends and the edge.
        """
        ends = {self._get_role(end) for end in (*ENDS, 'edge')}
        return {name: value for name, value in vars(namespace).items() if name not in ends}

    def _get_term(
        self,
        namespace: SimpleNamespace,
        member: SimpleNamespace,
        other: SimpleNamespace,
        edge: SimpleNamespace | None = None,
    ) -> Target:
        """Return the term of one other member, with the delay and lag applied.

        The delay and the lag apply to the read state.
        """
        member_values, other_values = dict(vars(member)), dict(vars(other))
        if self._is_nonzero('delay'):
            other_values[self.read] = delayed(other_values[self.read], self._get_parameter_symbol('delay', namespace))
        if self._is_nonzero('lag'):
            member_values[self.read] = member_values[self.read] + self._get_parameter_symbol('lag', namespace)
        if self.function is None:
            return other_values[self.read] - member_values[self.read]
        return self.function(
            SimpleNamespace(
                **self._parameters(namespace),
                member=SimpleNamespace(**member_values),
                other=SimpleNamespace(**other_values),
                edge=edge,
            ),
        )

    def _get_weight(
        self,
        namespace: SimpleNamespace,
        member: SimpleNamespace,
        other: SimpleNamespace,
        edge: SimpleNamespace | None = None,
    ) -> Target:
        """Return the weight of the term of one other member.

        It comes from callable weights, or is 1 for other weights.
        """
        if not callable(self.weights):
            return SR(1)
        return self.weights(SimpleNamespace(**self._parameters(namespace), member=member, other=other, edge=edge))

    def _get_matrix(self, namespace: SimpleNamespace) -> npt.ArrayLike | scipy.sparse.sparray | Expression | None:
        """Return the matrix of weights, or `None`.

        The diagonal is zero unless the member is included. Weights that name a shaped parameter give the symbol of that
        parameter.
        """
        if self._names_matrix():
            name = cast(str, self.weights)
            if not hasattr(namespace, name):
                error_msg = (
                    f'The weights {name!r} of coupling {self.label!r} name no parameter of the interaction. Declare '
                    f"it as `ParameterSpec(shape=('target', 'source'))`."
                )
                raise ValidationError(error_msg)
            return cast(Expression, getattr(namespace, name))
        if not self._has_matrix():
            return None
        weights = cast(npt.ArrayLike | scipy.sparse.sparray, self.weights)
        if not self._excludes_self():
            return weights
        if scipy.sparse.issparse(weights):
            matrix = scipy.sparse.csr_array(weights, dtype=float)
            matrix.setdiag(0)
            matrix.eliminate_zeros()
            return matrix
        matrix = np.array(weights, dtype=float)
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

    def _get_reduced(self, total_of: Callable[[Part], Target], number: Target) -> Target:
        """Return the reduction from sums of parts of the weighted terms.

        It also uses the number of members, by reduction.
        """
        if self.reduce == 'sum':
            return total_of(lambda value, weight: weight * value)
        if self.reduce == 'mean':
            return total_of(lambda value, weight: weight * value) / number
        if self.reduce == 'normalized-mean':
            return total_of(lambda value, weight: weight * value) / total_of(lambda value, weight: weight)
        sines = total_of(lambda value, weight: weight * sin(value))
        return arctan2(sines, total_of(lambda value, weight: weight * cos(value)))

    def _get_group_total(self, namespace: SimpleNamespace, part: Part) -> Target:
        """Return the sum of a part of the weighted term over the other group members."""
        source, target = getattr(namespace, self._get_role('source')), self._get_member(namespace, 'target')
        matrix = self._get_matrix(namespace)
        value, weight = self._get_term(namespace, target, source), self._get_weight(namespace, target, source)
        reduction = self._map(part(value, weight), lambda item: total(item, over=source, weights=matrix))
        if self._excludes_self() and matrix is None:
            reduction = reduction - part(
                self._get_term(namespace, target, target),
                self._get_weight(namespace, target, target),
            )
        return reduction

    @staticmethod
    def _get_extreme(values: list[Target], extreme: Callable[..., Expression]) -> Target:
        """Return the largest or smallest of several values.

        It works component by component for vectors.
        """
        if len(values) == 1:
            return values[0]
        first = values[0]
        if isinstance(first, FreeModuleElement):
            columns = zip(*[list(value) for value in values], strict=True)
            return first.parent()([extreme(*[SR(item) for item in column]) for column in columns])
        return extreme(*[SR(value) for value in values])

    @staticmethod
    def _get_sum(pairs: list[tuple[Target, Target]], part: Part) -> Target:
        """Return the sum of a part of the weighted terms of separate elements."""
        parts = [part(value, weight) for value, weight in pairs]
        return sum(parts[1:], parts[0])

    def _get_separate_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over the separate elements of `namespace.source`.

        It is written out term by term.
        """
        if self._has_matrix():
            error_msg = (
                'A matrix of weights of coupling needs groups. Give the weights of separate elements as a callable.'
            )
            raise ValidationError(error_msg)
        target = self._get_member(namespace, 'target')
        others = [*getattr(namespace, self._get_role('source')), *([target] if self.include_self else [])]
        pairs = [
            (self._get_term(namespace, target, other), self._get_weight(namespace, target, other)) for other in others
        ]
        if not pairs:
            error_msg = f'The coupling {self.label!r} of separate elements has no other element. Bind two or more.'
            raise ValidationError(error_msg)
        if self.reduce in EXTREMES:
            extreme = maximum if self.reduce == 'max' else minimum
            return self._get_extreme([value * weight for value, weight in pairs], extreme)
        return self._get_reduced(lambda part: self._get_sum(pairs, part), SR(len(pairs)))

    def _get_group_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over every other member of the group.

        For a source group, it is over every member of the source group.
        """
        source, target = getattr(namespace, self._get_role('source')), self._get_member(namespace, 'target')
        if isinstance(source, tuple):
            return self._get_separate_reduction(namespace)
        if self.reduce in EXTREMES:
            extreme = maximum if self.reduce == 'max' else minimum
            value = self._get_term(namespace, target, source) * self._get_weight(namespace, target, source)
            return self._map(value, lambda item: extreme(item, over=source))
        number = count(over=source) - (1 if self._excludes_self() else 0)
        return self._get_reduced(lambda part: self._get_group_total(namespace, part), number)

    def _get_next_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over the next member of the group."""
        target = self._get_member(namespace, 'target')
        shifted = self._shifted(target)
        reduction = self._get_term(namespace, target, shifted) * self._get_weight(namespace, target, shifted)
        if self.include_self:
            own = self._get_term(namespace, target, target) * self._get_weight(namespace, target, target)
            reduction = reduction + own
        return reduction / 2 if self.reduce == 'mean' and self.include_self else reduction

    def _get_neighbor_total(self, namespace: SimpleNamespace, part: Part) -> Target:
        """Return the sum of a part of the weighted term over the neighbourhood.

        The sum is over the neighbours and the member itself.
        """
        source, target = getattr(namespace, self._get_role('source')), self._get_member(namespace, 'target')
        value, weight = self._get_term(namespace, target, source), self._get_weight(namespace, target, source)
        reduction = self._map(part(value, weight), lambda item: neighbors(item, over=self.neighborhood))
        return reduction + part(self._get_term(namespace, target, target), self._get_weight(namespace, target, target))

    def _get_neighbor_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over the neighbours of the member."""
        source, target = getattr(namespace, self._get_role('source')), self._get_member(namespace, 'target')
        term = self._get_term(namespace, target, source)
        weight = self._get_weight(namespace, target, source) if callable(self.weights) else None
        if not self.include_self:
            return self._map(
                term,
                lambda item: neighbors(item, over=self.neighborhood, reduce=self.reduce, weight=weight),
            )
        if self.reduce in EXTREMES:
            extreme = maximum if self.reduce == 'max' else minimum
            others = neighbors(term, over=self.neighborhood, reduce=self.reduce, weight=weight)
            own = self._get_term(namespace, target, target) * self._get_weight(namespace, target, target)
            return extreme(others, own)
        number = neighbors(SR(1), over=self.neighborhood, reduce='count') + 1
        return self._get_reduced(lambda part: self._get_neighbor_total(namespace, part), number)

    def _get_hyperedge_term(self, namespace: SimpleNamespace, around: SimpleNamespace) -> Target:
        """Return the term of one hyperedge around a member.

        The lag applies to the read state of the member.
        """
        member = around.member
        if self._is_nonzero('lag'):
            lag = self._get_parameter_symbol('lag', namespace)
            member = SimpleNamespace(**{**vars(member), self.read: getattr(member, self.read) + lag})
        if self.function is None:
            members = around.members
            own = getattr(around.member, self.read)
            others = total(getattr(members, self.read), over=members) - own
            return others - (count(over=members) - 1) * getattr(member, self.read)
        return self.function(SimpleNamespace(**self._parameters(namespace), **{**vars(around), 'member': member}))

    def _get_hyperedge_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over the hyperedges around the member.

        Args:
            namespace:
                The namespace of the states and parameters of the interaction.

        Returns:
            reduction:
                The reduction over the hyperedges around the member.

        Raises:
            ValidationError:
                The members live on no network with hyperedges and in no continuous space.
        """
        target = self._get_member(namespace, 'target')
        others, members = getattr(target, 'others', None), getattr(target, 'members', None)
        if others is None or members is None:
            error_msg = (
                f'The coupling {self.label!r} over hyperedges binds {self.target!r}, whose members live on no network '
                'with hyperedges and in no continuous space. Give the group a network space with hyperedges, or a '
                'continuous space and a neighbourhood.'
            )
            raise ValidationError(error_msg)
        if self.size is not None:
            others = others[: self.size - 1]
        around = SimpleNamespace(member=target, others=others, members=members)
        term = self._get_hyperedge_term(namespace, around)
        weight = None
        if callable(self.weights):
            weight = self.weights(SimpleNamespace(**self._parameters(namespace), **vars(around)))
        return self._map(
            term,
            lambda item: hyperedges(item, reduce=self.reduce, size=self.size, weight=weight, over=self.neighborhood),
        )

    def _get_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction that one member gets, by mode."""
        if self.mode == 'hyperedges':
            return self._get_hyperedge_reduction(namespace)
        if self.mode == 'group':
            return self._get_group_reduction(namespace)
        if self.mode == 'next':
            return self._get_next_reduction(namespace)
        if self.mode == 'edges':
            return self._get_neighbor_reduction(namespace)
        raise self._get_variant_error(self.mode, 'mode', '_get_reduction')

    def _get_edge_term(self, namespace: SimpleNamespace, member: SimpleNamespace, other: SimpleNamespace) -> Target:
        """Return the term one end of an edge gets from the other end.

        It is weighted and divided for a mean.
        """
        edge = getattr(namespace, self._get_role('edge'), None)
        term = self._get_term(namespace, member, other, edge)
        if isinstance(self.weights, str):
            term = term * getattr(edge, self.weights)
        elif callable(self.weights):
            term = term * self._get_weight(namespace, member, other, edge)
        if self.reduce == 'mean':
            term = term / (member.degree + int(self.include_self))
        return term

    def _get_edge_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of both ends of an edge.

        They include the rate of the member itself when it is included.
        """
        source, target = getattr(namespace, self._get_role('source')), self._get_member(namespace, 'target')
        laws: list[BaseLaw] = [
            RateLaw(getattr(target, self._get_written()), self._get_edge_term(namespace, target, source)),
            RateLaw(getattr(source, self._get_written()), self._get_edge_term(namespace, source, target)),
        ]
        if self.include_self:
            own = self._get_term(namespace, target, target) * self._get_weight(namespace, target, target)
            divisor = target.degree + 1 if self.reduce == 'mean' else 1
            laws.append(RateLaw(getattr(target, self._get_written()), own / divisor))
        return laws

    @staticmethod
    def _get_bound(agent: AgentViewProtocol, bound: float | str) -> float:
        """Return a bound given as a number or as the name of a parameter of the agent."""
        return float(cast(float, getattr(agent, bound))) if isinstance(bound, str) else float(bound)

    def _get_agent_reduction(
        self,
        agent: AgentViewProtocol,
        system: SystemViewProtocol,
    ) -> tuple[Array | None, int | None]:
        """Return the sum of the reduced terms of the top-priority radii holding an agent.

        The radii are those of the highest priority that hold another agent.
        """
        groups = set(system.groups(self._get_role('source' if self.source is not None else 'target')))
        found: dict[float, list[AgentViewProtocol]] = {}
        levels: dict[int, list[Array]] = {}
        for spec in self.radii:
            radius = self._get_bound(agent, spec.radius)
            if radius not in found:
                found[radius] = [other for other in agent.neighbors(radius=radius) if other.group in groups]
            others = found[radius]
            if not others:
                continue
            terms = np.array([np.asarray(spec.function(agent, other, system), dtype=float) for other in others])
            levels.setdefault(spec.priority, []).append(
                terms.mean(axis=0) if spec.reduce == 'mean' else terms.sum(axis=0),
            )
        if not levels:
            return None, None
        priority = max(levels)
        return np.sum(levels[priority], axis=0), priority

    @staticmethod
    def _turn(current: Array, value: Array, max_angle: float) -> Array:
        """Return a vector turned from the current direction toward its own by an angle.

        The turn is at most the angle and keeps the norm.
        """
        norm, size = float(np.linalg.norm(value)), float(np.linalg.norm(current))
        if norm == 0 or size == 0:
            return value
        direction, facing = value / norm, current / size
        angle = float(np.arccos(np.clip(facing @ direction, -1.0, 1.0)))
        if angle <= max_angle:
            return value
        if angle == np.pi:
            direction = np.eye(len(facing))[int(np.argmin(np.abs(facing)))]
        normal = direction - (facing @ direction) * facing
        normal = normal / np.linalg.norm(normal)
        return norm * (np.cos(max_angle) * facing + np.sin(max_angle) * normal)

    def _get_clamped(self, agent: AgentViewProtocol, current: Array, value: Array) -> Array:
        """Return a new vector with its turn and its norm clamped."""
        clamp = self.clamp
        if clamp is None:
            return value
        if clamp.max_angle is not None:
            value = self._turn(current, value, self._get_bound(agent, clamp.max_angle))
        norm = float(np.linalg.norm(value))
        if norm == 0:
            return value
        low = 0.0 if clamp.min_norm is None else self._get_bound(agent, clamp.min_norm)
        high = np.inf if clamp.max_norm is None else self._get_bound(agent, clamp.max_norm)
        return value * min(max(norm, low), high) / norm

    def _update_member(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Set the written states of one agent to the clamped combination of the reduction.

        The reduction is over its radii.
        """
        if not agent.alive:
            return
        names = [self.write] if isinstance(self.write, str) else list(self.write)
        current = np.array([float(cast(float, getattr(agent, name))) for name in names])
        reduction, priority = self._get_agent_reduction(agent, system)
        if self.combine is not None:
            combine = cast(AgentCombination, self.combine)
            value = np.asarray(combine(agent, system, reduction, priority), dtype=float)
        else:
            value = current if reduction is None else current + reduction
        for name, item in zip(names, self._get_clamped(agent, current, value), strict=True):
            setattr(agent, name, float(item))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of both ends of every edge, or the laws of every member.

        The laws of a member are its rate or its next value.

        Args:
            namespace:
                The namespace of an edge, with the ends `source` and `target`.

        Returns:
            laws:
                The laws of the coupling.
        """
        if self.form == 'agent':
            return [RuleLaw(self._update_member, phase=self.phase, members=self._get_role('target'))]
        if self._is_per_edge():
            return self._get_edge_laws(namespace)
        target = self._get_member(namespace, 'target')
        reduction = self._get_reduction(namespace)
        if self.form == 'rate':
            return [RateLaw(getattr(target, self._get_written()), reduction)]
        state = getattr(target, self._get_written())
        if self.combine is None:
            return [UpdateLaw(state, reduction, phase=self.phase)]
        member = SimpleNamespace(**self._parameters(namespace), member=target)
        return [UpdateLaw(state, cast(Combination, self.combine)(member, reduction), phase=self.phase)]
