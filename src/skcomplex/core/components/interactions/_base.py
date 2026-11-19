"""Define the abstract base of every interaction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import inspect
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass, replace
from operator import lt as operator_lt
from types import SimpleNamespace, UnionType
from typing import ClassVar, Union, cast, get_args, get_origin

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import CC, RR, SR, ZZ, Expression, function, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.structure.element import Matrix
from sage.symbolic.function import SymbolicFunction
from sage.symbolic.operators import FDerivativeOperator
from sklearn.base import clone

from ...enums import DomainKind
from ...errors import SpaceError, ValidationError
from ...specs import BaseSpec, DeclaredVariableSpec, ParameterSpec, RoleSpec, StateSpec
from .._tags import Tags
from ..base import BaseComponent
from ..laws import BaseLaw, NoiseLaw, PartnerSpec, RateLaw, RuleLaw, UpdateLaw
from ..laws._flow_term import FlowTermLaw
from ..laws.utils import delayed, derivative, lag
from ..laws.utils._fractional import fractional_of
from ..spaces import BaseSpace, ContinuousSpace
from ..types import AgentViewProtocol, SystemViewProtocol
from ._interaction_requirement_spec import _InteractionRequirementSpec
from ._wiener_process_spec import WienerProcessSpec

GRID_CANDIDATES = ('neighbors', 'cell', 'local')
RATE_DOMAINS = (DomainKind.REAL, DomainKind.INTEGER, DomainKind.COMPLEX, DomainKind.VECTOR)


class BaseInteraction(BaseComponent[BaseSpace | SimpleNamespace, SimpleNamespace]):
    """An abstract interaction, the component that changes the states of elements.

    An interaction binds elements through roles, such as `'source'` and `'target'`, reads their states, and returns laws
    that change them. It holds no state of its own. A state on a coupling, such as the weight of an edge, lives in a
    group on the edges or hyperedges of a network. A subclass takes each role and parameter as an argument of its
    `__init__`, annotated with `RoleSpec` or `ParameterSpec` inside `Annotated`, and stores it under the same name, as a
    scikit-learn estimator does. It returns its laws from `_get_laws`, where `namespace.<role>` reads the states of the
    element a role binds and `namespace.<parameter>` is a symbol.

    A role states the element kinds and states it needs in its `RoleSpec`, and the class lists the laws it may return in
    `law_classes`, so that `get_requirements` and `get_tags` can name them before a run. A subclass whose arguments
    choose among its laws overrides `_get_tags`. A subclass that names its roles or parameters otherwise returns their
    names from `_get_role` and `_get_parameter`. An interaction that draws a random order of agents, breaks ties at
    random, or picks members at random is stochastic whatever its laws.

    Unlike `BaseComponent`, it changes the states of other components and holds none of its own. Choose it when a new
    kind of interaction runs its laws in a way of its own, as `NeighborInteraction` runs them once per edge. Choose
    `Interaction` for laws that run once for the elements the roles bind.

    It appears in every field through its subclasses, such as infection in an epidemic, a chemical reaction, a payment
    between accounts, and the coupling of oscillators.

    Args:
        label:
            The name of the interaction, unique among the components of its system, such as `'growth'`.

    Attributes:
        laws_:
            The laws the interaction contributes for the namespace of its last run.

    Examples:
        >>> from skcomplex.core.components.interactions import BaseInteraction
        >>> from skcomplex.core.specs import Role
        >>> class GrowthInteraction(BaseInteraction):
        ...     def __init__(self, label, target: Role, r=1.0):
        ...         super().__init__(label)
        ...         self.target = target
        ...         self.r = r
        ...
        ...     def _get_laws(self, namespace):
        ...         return []
        >>> GrowthInteraction('growth', target='rabbits', r=0.5).get_params()
        {'label': 'growth', 'r': 0.5, 'target': 'rabbits'}
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = ()
    laws_: Sequence[BaseComponent]

    def _get_argument(self, name: str) -> object:
        """Return the value of an argument of the general interaction.

        It reads the argument under its subclass name, and returns `None` without one.

        Args:
            name:
                The name of the argument in the general interaction, such as `'noise'`.

        Returns:
            value:
                The value of the argument, or `None` when the interaction has no such attribute.
        """
        return getattr(self, self._get_parameter(name), None)

    def _get_variant_tags(
        self,
        classes: Sequence[type[BaseLaw]],
        *,
        stochastic: bool = False,
        structure: frozenset[str] = frozenset(),
    ) -> Tags:
        """Return the tags of the laws an interaction returns for its arguments.

        The tags are unknown below its own `_get_laws`.

        Args:
            classes:
                The law classes the arguments choose.

            stochastic:
                Whether the arguments make laws of deterministic kinds draw, such as an update with a probability.

            structure:
                How the laws change the structure of the system, such as `'changing topology'`.

        Returns:
            tags:
                The tags of the chosen laws, or unknown law kinds when a subclass writes laws it does not list.
        """
        if self._get_declared_laws() is None:
            return replace(self._get_law_tags(None), structure=structure)
        tags = self._get_law_tags(classes)
        return replace(tags, randomness='stochastic' if stochastic else tags.randomness, structure=structure)

    def _get_tags(self) -> Tags:
        """Return the tags of the laws the class lists.

        The tags are unknown law kinds when the class writes laws it does not list.
        """
        return self._get_law_tags(self._get_declared_laws())

    def _get_draws(self) -> bool:
        """Return whether the interaction draws.

        It draws, by `draws`, a random order of its agents, or random ties or picks.
        """
        return (
            self.draws
            or self._get_argument('schedule') in ('random-sequential', 'member')
            or 'random' in (self._get_argument('ties'), self._get_argument('pick'))
        )

    @classmethod
    def _is_preparing_laws(cls) -> bool:
        """Return whether the class sets its laws in `_prepare`, not in `_get_laws`."""
        owner = next(klass for klass in cls.__mro__ if '_get_laws' in vars(klass))
        return owner is BaseInteraction

    def _get_declared(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the numeric arguments of an interaction as parameters, by name.

        An argument annotated with a spec keeps its spec, and a role or a setting that is no number declares nothing.
        An `int` is an integer, a `float` a real, and a `complex` a complex parameter, such as the `c` of `z**2 + c`. A
        list or an array of numbers is a vector parameter of its length, such as the stiffness of a trap along each
        axis, and a square matrix of numbers a matrix parameter. The allowed values of an argument come from
        `Annotated`, as for the values of an element, such as `k: Annotated[float, RealSet.open(0, oo)] = 1.0` for a
        positive rate. An argument whose value is a `WienerProcessSpec` binds those processes, as the market noise
        that drives the prices of several assets does. The parameters keep the order of the arguments.
        """
        if not self._is_preparing_laws():
            return {}
        specs = self._get_specs()
        allowed = self._get_allowed()
        params = self.get_params(deep=False)
        declared = {}
        for name in inspect.signature(type(self)).parameters:
            value = params.get(name)
            if isinstance(value, WienerProcessSpec) and name not in specs:
                declared[name] = DeclaredVariableSpec(value, RoleSpec(noise=True))
                continue
            if name in specs or name == 'label' or not self._is_numeric(value):
                continue
            vector, n_dim, tensor = self._get_shape_of(value)
            first = np.asarray(value).ravel()[0].item() if vector or tensor else value
            domain = allowed.get(name, self._get_domain_of(first))
            spec = ParameterSpec(domain=domain, vector=vector, n_dim=n_dim, tensor=tensor)
            declared[name] = DeclaredVariableSpec(self._get_matrix_entries(value, tensor), spec)
        return declared

    @classmethod
    def _get_roles(cls, *, single: bool = False) -> tuple[str, ...]:
        """Return the names of the arguments that bind elements.

        They come in the order of the constructor.

        With `single`, only the roles that bind one element, such as `target: Role`, and not several, such as
        `reactants: tuple[Role, ...]`, or Wiener processes.
        """
        return tuple(
            name
            for name, spec in cls._get_specs().items()
            if isinstance(spec, RoleSpec) and not (single and (spec.variadic or spec.noise))
        )

    @staticmethod
    def _is_numeric(value: object) -> bool:
        """Return whether a value is a number, or a vector or a square matrix of numbers.

        A boolean is no number. A vector or a matrix is a list or an array, so a tuple stays a setting, such as the
        offsets of a stencil.
        """
        if isinstance(value, bool):
            return False
        if isinstance(value, int | float | complex):
            return True
        if not isinstance(value, list | np.ndarray):
            return False
        try:
            array = np.asarray(value)
        except ValueError:
            return False
        numeric = np.issubdtype(array.dtype, np.number) and array.size > 0
        return numeric and BaseInteraction._get_shape_of(array)[1] is not None

    @classmethod
    def _get_allowed(cls) -> dict[str, object]:
        """Return the allowed values `Annotated` gives each argument, by name.

        A set, such as `{0, 1}`, is a finite set of values, and any other container of values, such as
        `RealSet.open(0, oo)`, is taken as it is.
        """
        allowed: dict[str, object] = {}
        for name, argument in inspect.signature(cls.__init__, eval_str=True).parameters.items():
            for item in getattr(cls._get_annotated(argument.annotation), '__metadata__', ()):
                if isinstance(item, set | frozenset):
                    allowed[name] = FiniteEnumeratedSet(sorted(item, key=str))
                elif hasattr(item, '__contains__') and not isinstance(item, str | BaseSpec):
                    allowed[name] = item
        return allowed

    @staticmethod
    def _get_domain_of(value: complex) -> object:
        """Return the domain of a numeric argument: integers, reals, or complex numbers."""
        if isinstance(value, int):
            return ZZ
        return RR if isinstance(value, float) else CC

    @classmethod
    def _get_declared_laws(cls) -> tuple[type, ...] | None:
        """Return the law classes a class lists, or `None` when they are unknown."""
        if not cls._is_preparing_laws():
            return super()._get_declared_laws()
        for klass in cls.__mro__:
            if klass is BaseInteraction:
                return None
            if 'law_classes' in vars(klass):
                classes: tuple[type, ...] = vars(klass)['law_classes']
                return classes
        return None

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return no laws, since an interaction sets its laws in `_prepare`.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                No laws.
        """
        return []

    @staticmethod
    def _get_functional(value: object, time: Expression, functions: dict[str, Expression]) -> object:
        """Return the symbols of states in a value of a role as functions of time."""
        if isinstance(value, SimpleNamespace):
            return SimpleNamespace(
                **{name: BaseInteraction._get_functional(item, time, functions) for name, item in vars(value).items()},
            )
        if isinstance(value, tuple):
            return tuple(BaseInteraction._get_functional(item, time, functions) for item in value)
        if isinstance(value, FreeModuleElement):
            return vector(SR, [BaseInteraction._get_functional(item, time, functions) for item in value])
        if isinstance(value, Expression) and value.is_symbol() and value is not time:
            functions[str(value)] = value
            return function(str(value))(time)
        return value

    @staticmethod
    def _get_state_of(value: Expression, time: Expression, symbols: Mapping[str, Expression], discrete: bool) -> object:
        """Return the state, derivative, lag, or delay a function of time stands for."""
        operator = value.operator()
        if isinstance(operator, FDerivativeOperator) and operator.function().name() in symbols:
            state = derivative(symbols[operator.function().name()], len(operator.parameter_set()))
        else:
            name = str(getattr(operator, 'name', lambda: '')())
            if name not in symbols:
                return None
            state = symbols[name]
        if len(value.operands()) != 1:
            return None
        shift = SR(time - value.operands()[0]).expand()
        if shift.is_trivial_zero():
            return state
        shift = SR(BaseInteraction._get_translated(shift, time, symbols, discrete))
        if discrete:
            return lag(state, shift)
        shortest = BaseInteraction._get_shortest(shift)
        return delayed(state, shift) if shortest is None else delayed(state, shift, shortest)

    @staticmethod
    def _get_shortest(shift: Expression) -> Expression | None:
        """Return the lower bound of a delay that reads states, or `None` for none.

        A delay clipped from below, such as `x(t=t - clip(tau, low=0.1))` for a delay `tau` that changes with the
        state, is never shorter than its bound, which bounds the steps of the solver. The shift is expanded and never
        simplified, since a clipped delay holds conditions that simplifying cannot read.
        """
        operator = shift.operator()
        if getattr(operator, 'name', lambda: '')() != 'where':
            return None
        condition, then, _ = shift.operands()
        bounded = condition.operator() is operator_lt and bool(SR(condition.rhs() - then).is_trivial_zero())
        return then if bounded else None

    @staticmethod
    def _get_translated(value: object, time: Expression, symbols: Mapping[str, Expression], discrete: bool) -> object:
        """Return a value with derivatives and shifts in time of states translated."""
        if isinstance(value, list | tuple):
            return type(value)(BaseInteraction._get_translated(item, time, symbols, discrete) for item in value)
        if isinstance(value, Mapping):
            return {
                BaseInteraction._get_translated(key, time, symbols, discrete): BaseInteraction._get_translated(
                    item,
                    time,
                    symbols,
                    discrete,
                )
                for key, item in value.items()
            }
        if isinstance(value, FreeModuleElement | Matrix):
            return value.apply_map(lambda item: BaseInteraction._get_translated(SR(item), time, symbols, discrete))
        if not isinstance(value, Expression) or value.operator() is None:
            return value
        state = BaseInteraction._get_state_of(value, time, symbols, discrete)
        if state is not None:
            return state
        operands = [BaseInteraction._get_translated(item, time, symbols, discrete) for item in value.operands()]
        return value.operator()(*operands)

    @staticmethod
    def _get_symbolic[ValueT](value: ValueT, time: Expression) -> ValueT:
        """Return a value with every state `x(t)` read as its symbol `x`.

        A derivative `x.diff(t, k)` becomes `derivative(x, k)`. Sage differentiates by symbols only, so a recipe that
        differentiates a function of the states, such as the gradient of a potential or the Euler-Lagrange equations of
        a Lagrangian, reads it in symbols first. A value is an expression, a vector, a tuple, or a namespace of them.
        """
        symbolic: object = value
        if isinstance(value, SimpleNamespace):
            symbolic = SimpleNamespace(
                **{name: BaseInteraction._get_symbolic(item, time) for name, item in vars(value).items()},
            )
        elif isinstance(value, tuple):
            symbolic = tuple(BaseInteraction._get_symbolic(item, time) for item in value)
        else:
            symbols: dict[str, Expression] = {}
            BaseInteraction._collect_states(value, time, symbols)
            if symbols:
                symbolic = BaseInteraction._get_translated(value, time, symbols, discrete=False)
        return cast('ValueT', symbolic)

    @staticmethod
    def _collect_states(value: object, time: Expression, symbols: dict[str, Expression]) -> None:
        """Add the symbol of every state a value reads as a function of time, by name."""
        if isinstance(value, FreeModuleElement | list | tuple | Matrix):
            for item in value.list() if isinstance(value, Matrix) else value:
                BaseInteraction._collect_states(item, time, symbols)
            return
        if not isinstance(value, Expression) or value.operator() is None:
            return
        operator = value.operator()
        if isinstance(operator, FDerivativeOperator):
            operator = operator.function()
        if isinstance(operator, SymbolicFunction) and value.operands() == [time]:
            symbols[operator.name()] = SR.var(operator.name())
            return
        for item in value.operands():
            BaseInteraction._collect_states(item, time, symbols)

    @staticmethod
    def _get_translated_spec[SpecT: BaseSpec](
        spec: SpecT,
        time: Expression,
        symbols: Mapping[str, Expression],
        discrete: bool,
    ) -> SpecT:
        """Return a spec of a law with the states of its fields translated.

        The arguments of a `CallSpec` are such fields.
        """
        fields = {item.name: getattr(spec, item.name) for item in dataclass_fields(spec) if item.init}
        translated = {
            name: BaseInteraction._get_translated(item, time, symbols, discrete) for name, item in fields.items()
        }
        changed = {name: item for name, item in translated.items() if item is not fields[name]}
        return replace(spec, **changed) if changed else spec

    @staticmethod
    def _get_replaced(law: BaseComponent, **params: object) -> BaseComponent:
        """Return a copy of a law with some parameters replaced and the others shared.

        A deep copy would rebuild every Sage expression from its pickle, which can change its numbers, such as
        `1.0 * x` into `x`, and so merge branches of a table function that were apart. The copy shares them instead.
        """
        return copy.copy(law).set_params(**params)

    @staticmethod
    def _get_law_in_states(law: BaseComponent, time: Expression, symbols: Mapping[str, Expression]) -> BaseComponent:
        """Return a law written with functions of time as a law of the states."""
        if not symbols:
            return law
        discrete = isinstance(law, UpdateLaw)
        params = law.get_params(deep=False)
        translated = {
            name: (
                BaseInteraction._get_translated_spec(item, time, symbols, discrete)
                if isinstance(item, BaseSpec) and is_dataclass(item)
                else BaseInteraction._get_translated(item, time, symbols, discrete)
            )
            for name, item in params.items()
            if name != 'label'
        }
        changed = {name: item for name, item in translated.items() if item is not params[name]}
        return BaseInteraction._get_replaced(law, **changed) if changed else law

    @staticmethod
    def _get_normalized(law: BaseComponent) -> BaseComponent:
        """Return a law on a derivative of order `k` as a law on the state of order `k`.

        A rate law on the fractional derivative `fractional(x, alpha)` becomes a rate law on `x` of the order `alpha`,
        and a law on the derivatives of the components of a vector, such as `x.diff(t, 2)` for a position in the
        plane, becomes a law on the vector of the order of its derivatives.

        Raises:
            ValidationError:
                A fractional derivative is the target of a law other than a rate law, or of no state.
        """
        if isinstance(law, RateLaw | NoiseLaw) and isinstance(law.target, FreeModuleElement):
            return BaseInteraction._get_normalized_vector(law)
        if not isinstance(law, RateLaw | NoiseLaw) or not isinstance(law.target, Expression):
            return law
        found = fractional_of(law.target)
        if found is not None:
            state, order = found
            if not isinstance(law, RateLaw) or not state.is_symbol():
                error_msg = (
                    f'A {type(law).__name__} targets a fractional derivative, but only a RateLaw of a state takes one, '
                    'such as `RateLaw(fractional(s.target.x, 0.5), -s.k * s.target.x)`.'
                )
                raise ValidationError(error_msg)
            return BaseInteraction._get_replaced(law, target=state, order=1 if order == 1 else order)
        if law.target.operator() is not derivative:
            return law if law.order is not None else BaseInteraction._get_replaced(law, order=1)
        state, order = law.target.operands()
        return BaseInteraction._get_replaced(law, target=state, order=int(order))

    @staticmethod
    def _get_normalized_vector(law: RateLaw | NoiseLaw) -> BaseComponent:
        """Return a law on the derivatives of a vector as a law on the vector.

        The derivatives of order `k` of the components, such as `x.diff(t, 2)`, give the vector the order `k`. A target
        whose components are no derivatives of one order is kept as it is.
        """
        found = [
            item.operands() if isinstance(item, Expression) and item.operator() is derivative else None
            for item in law.target
        ]
        orders = {int(item[1]) for item in found if item is not None}
        if not found or None in found or len(orders) != 1:
            return law
        states = vector(SR, [item[0] for item in found if item is not None])
        return clone(law).set_params(target=states, order=orders.pop())

    def _get_laws_of(self, inputs: object) -> Sequence[BaseComponent]:
        """Return the laws for a namespace, from a clone prepared on it if needed."""
        if not self._is_preparing_laws():
            return super()._get_laws_of(inputs)
        if not isinstance(inputs, SimpleNamespace):
            error_msg = f'{type(self).__name__} learns its laws from a namespace of symbols, but got {inputs!r}.'
            raise ValidationError(error_msg)
        time = getattr(inputs, 't', SR.var('t'))
        functions: dict[str, Expression] = {}
        roles = {
            name: self._get_functional(item, time, functions) if isinstance(item, SimpleNamespace | tuple) else item
            for name, item in vars(inputs).items()
        }
        functional = SimpleNamespace(**roles)
        prepared = copy.copy(self)
        prepared.prepare(functional)
        laws: list[BaseComponent] = []
        for law in prepared.laws_:
            rule = self._get_rule(law, roles, functional)
            if rule is not None:
                laws.append(rule)
                continue
            translated = self._get_law_in_states(law, time, functions)
            flow = isinstance(translated, RateLaw | UpdateLaw)
            terms = FlowTermLaw._get_terms(translated, self.label) if flow else [translated]
            laws.extend(self._get_local_rule(self._get_normalized(term)) for term in terms)
        return laws

    @staticmethod
    def _get_local_rule(law: BaseComponent) -> BaseComponent:
        """Return the local rule of an update law with a neighbourhood, or the law itself.

        An update law with a neighbourhood, such as `UpdateLaw(s.target.alive, life(), neighborhood=MOORE)`, runs as a
        local rule of the cells, with its lookup tables and fused lattice kernels.
        """
        if isinstance(law, UpdateLaw) and law.neighborhood is not None:
            law.prepare()
            return law._get_local_rule()
        return law

    def _get_rule(
        self,
        law: BaseComponent,
        roles: Mapping[str, object],
        system: SimpleNamespace | None = None,
    ) -> RuleLaw | None:
        """Return the rule of agents or of the system that an update law states, or `None`.

        An update law whose target is a bound group, such as `UpdateLaw(s.target, move)`, is an agent rule: its value
        is a function `move(agent, system)` that reads and changes one member, and every member runs it at each step.
        An update law whose target is the namespace of the interaction itself, such as `UpdateLaw(s, brake)`, is a rule
        of the whole system: its value is a function `brake(system)` that runs once per step after the agents' rules.

        Raises:
            ValidationError:
                The target is a group but the value is no function of an agent and the system, the target is the
                namespace of the interaction but the value is no function of the system, or the target is a namespace
                that no role binds.
        """
        if not isinstance(law, UpdateLaw) or not isinstance(law.target, SimpleNamespace):
            return None
        if law.target is system:
            rule = law.value
            if not callable(rule) or not self._is_rule(rule, 1):
                error_msg = (
                    f'The update law of {self.label!r} on the whole system has the value {rule!r}. A rule of the '
                    'system is a function of the system, such as def brake(system): ...'
                )
                raise ValidationError(error_msg)
            return RuleLaw(rule, law.phase, label=law.label)
        role = next((name for name, item in roles.items() if item is law.target), None)
        if role is None:
            error_msg = (
                f'An update law of {self.label!r} targets {law.target!r}, which no role binds. A rule targets the '
                'group of a role, such as UpdateLaw(s.target, move).'
            )
            raise ValidationError(error_msg)
        rule = law.value
        if not callable(rule) or not self._is_rule(rule):
            error_msg = (
                f'The update law of {self.label!r} on the group of {role!r} has the value {rule!r}. A rule is a '
                'function of one agent and the system, such as def move(agent, system): agent.x += 1.'
            )
            raise ValidationError(error_msg)
        return RuleLaw(rule, law.phase, members=role, label=law.label)

    @staticmethod
    def _is_rule(rule: Callable[..., object], arguments: int = 2) -> bool:
        """Return whether a function takes a number of arguments, two by default.

        The two are the agent and the system.
        """
        try:
            inspect.signature(rule).bind(*[None] * arguments)
        except (TypeError, ValueError):
            return False
        return True

    def _prepare_on(self, space: BaseSpace | SimpleNamespace | None) -> None:
        """Check the interaction on the space of its system, learning no laws."""
        if not self._is_preparing_laws():
            super()._prepare_on(space)
            return
        BaseInteraction._prepare(self, space if isinstance(space, BaseSpace) else None)
        self._prepared = True

    def _get_prepared_laws(self, namespace: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return the laws of the interaction for a namespace of symbols."""
        if self._is_preparing_laws():
            return self._get_laws_of(namespace)
        return super()._get_prepared_laws(namespace)

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Set `laws_`, the laws of the interaction for a namespace of symbols.

        Args:
            inputs:
                The namespace of the symbols the laws can use.
        """
        self.laws_ = self._get_laws_of(inputs)

    def _get_binding(self, role: str) -> object:
        """Return what a role binds, such as the label of an element.

        It is the argument of the same name, such as `target='ball'`.

        Args:
            role:
                The name of the role.

        Returns:
            binding:
                The label, labels, or entries the role binds, or `None` when it binds nothing.
        """
        return getattr(self, role, None)

    def _get_role(self, role: str) -> str:
        """Return the role a subclass binds in place of a role of its general interaction.

        It returns the name of the role.

        Args:
            role:
                The name of the role in the general interaction, such as `'source'` or `'target'`.

        Returns:
            name:
                The name of the role of the subclass, the role itself by default.
        """
        return role

    def _get_member(self, namespace: SimpleNamespace, role: str) -> SimpleNamespace:
        """Return the namespace of the element bound to a role of the general interaction.

        The role is read under its subclass name.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

            role:
                The name of the role in the general interaction, such as `'source'` or `'target'`.

        Returns:
            member:
                The namespace bound to the name `_get_role` returns for the role.
        """
        member: SimpleNamespace = getattr(namespace, self._get_role(role))
        return member

    def _get_parameter(self, name: str) -> str:
        """Return the parameter a subclass reads in place of a general one.

        It returns the name of the parameter that the subclass reads in place of a parameter of its general interaction.

        Args:
            name:
                The name of the parameter in the general interaction, such as `'probability'`.

        Returns:
            parameter:
                The name of the parameter of the subclass, the name itself by default.
        """
        return name

    def _declares(self, name: str) -> bool:
        """Return whether the class declares a name as a parameter."""
        return isinstance(type(self)._get_specs().get(name), ParameterSpec)

    def _get_parameter_value(self, name: str, system: SystemViewProtocol) -> float:
        """Return the value of a parameter in an agent form.

        It is the plain attribute when the class does not declare the parameter.

        Args:
            name:
                The name of the parameter in the general interaction.

            system:
                The system view of the interaction.

        Returns:
            value:
                The value of the parameter of the system, or of the attribute of the interaction.
        """
        parameter = self._get_parameter(name)
        if self._declares(parameter):
            return system.get_parameter(parameter)
        return float(getattr(self, parameter))

    def _get_parameter_symbol(self, name: str, namespace: SimpleNamespace) -> object:
        """Return the symbol of a parameter in a symbolic form.

        It is the plain attribute when the class does not declare the parameter.

        Args:
            name:
                The name of the parameter in the general interaction.

            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            symbol:
                The symbol of the parameter in the namespace, read also when a component other than the interaction
                declares it, or else the value of the attribute of the interaction.
        """
        parameter = self._get_parameter(name)
        if self._declares(parameter) or hasattr(namespace, parameter):
            return getattr(namespace, parameter)
        return getattr(self, parameter)

    def _check_variant(
        self,
        name: str,
        value: object,
        values: Collection[object],
        kinds: tuple[type, ...] = (),
        *,
        functions: bool = False,
    ) -> None:
        """Raise an error for a variant of no named value and of no kind it also takes.

        The variant is none of its named values and of none of the kinds it also takes.

        Args:
            name:
                The name of the variant argument, such as `'pairing'`.

            value:
                The value of the variant argument.

            values:
                The named values of the variant.

            kinds:
                The types the variant takes besides its named values, such as spec classes.

            functions:
                Whether the variant takes a function.

        Raises:
            ValidationError:
                The value is none of the named values and of none of the kinds.
        """
        if isinstance(value, kinds) or (functions and callable(value)) or any(value == item for item in values):
            return
        named = [kind.__name__ for kind in kinds if kind is not BaseSpec]
        accepted = [*named, *(repr(str(item)) for item in values), *(['a function'] if functions else [])]
        spec = ', or a spec that the component handles' if BaseSpec in kinds else ''
        error_msg = f'Unknown {name} {value!r} of {self.label!r}. Use one of {", ".join(accepted)}{spec}.'
        raise ValidationError(error_msg)

    def _get_variant_error(self, value: object, name: str, method: str) -> ValidationError:
        """Return the error for a spec of a variant that no subclass runs.

        Args:
            value:
                The value of the variant argument.

            name:
                The name of the variant argument, such as `'rule'`.

            method:
                The private method a subclass overrides to run the spec.

        Returns:
            error:
                The error that names the variant and the method.
        """
        error_msg = (
            f'{type(self).__name__} {self.label!r} has no way to run the {name} {value!r}. Override `{method}` in a '
            'subclass to run it.'
        )
        return ValidationError(error_msg)

    def _update_agents[T](
        self,
        system: SystemViewProtocol,
        schedule: str,
        get_next: Callable[[AgentViewProtocol, SystemViewProtocol], T],
        set_next: Callable[[AgentViewProtocol, T], None],
    ) -> None:
        """Set the next value of every agent of the target by the schedule.

        The agents update all at once or one after the other.

        Args:
            system:
                The system view of the interaction.

            schedule:
                `'synchronous'` for every agent from the values at the start of the rule, or `'sequential'` or
                `'random-sequential'` for one agent after the other, in their order or in a new random order, each
                reading the values the agents before it set.

            get_next:
                The function of an agent and the system view that returns its next value.

            set_next:
                The function that sets the next value of an agent.
        """
        agents = list(system.get_agents(self._get_role('target')))
        if schedule == 'synchronous':
            values = [get_next(agent, system) for agent in agents]
            for agent, value in zip(agents, values, strict=True):
                set_next(agent, value)
            return
        order = system.rng.permutation(len(agents)) if schedule == 'random-sequential' else range(len(agents))
        for index in order:
            agent = agents[int(index)]
            if agent.alive:
                set_next(agent, get_next(agent, system))

    def _pick_best(self, scores: npt.ArrayLike, ties: str, system: SystemViewProtocol) -> int:
        """Return the position of the largest score.

        It is the first one or one drawn uniformly among ties.

        Args:
            scores:
                The scores of the options, in their order.

            ties:
                `'first'` for the first of the largest scores, or `'random'` for one of them drawn uniformly.

            system:
                The system view of the interaction, whose random stream draws among ties.

        Returns:
            position:
                The position of the chosen option.
        """
        values = np.asarray(scores, dtype=float)
        best = np.flatnonzero(values == values.max())
        if ties == 'first':
            return int(best[0])
        return int(best[int(system.rng.integers(len(best)))])

    def _check_states(self) -> None:
        """Raise an error for a type with states or adaptive parameters without a shape.

        The type declares the states or the adaptive parameters.
        """
        specs = self._get_specs()
        states = [name for name, spec in specs.items() if isinstance(spec, StateSpec)]
        adaptive = [
            name
            for name, spec in specs.items()
            if isinstance(spec, ParameterSpec) and spec.adaptive and spec.shape is None
        ]
        if adaptive:
            error_msg = (
                f'{type(self).__name__} declares the adaptive parameters {", ".join(adaptive)}, which laws change as '
                'states, but interactions hold no state. Declare them on the element whose parameters they are, or '
                'give them the shape of the groups the interaction couples, `ParameterSpec(shape=...)`.'
            )
            raise ValidationError(error_msg)
        if states:
            error_msg = (
                f'{type(self).__name__} declares the states {", ".join(states)}, but interactions hold no state. Put '
                'state on couplings in a group on the edges of a network space, `GroupElement(..., on="edges")`.'
            )
            raise ValidationError(error_msg)

    def _check_grid(self, name: str, value: object, space: BaseSpace | None, alternative: str) -> None:
        """Raise an error for a variant that needs cells or nodes in a continuous space.

        The variant needs the cells of a grid or the nodes of a network.

        Args:
            name:
                The name of the variant argument, such as `'pairing'`.

            value:
                The value of the variant argument.

            space:
                The space of the system.

            alternative:
                The variant or argument that runs in a continuous space.

        Raises:
            SpaceError:
                The space is continuous.
        """
        if not isinstance(space, ContinuousSpace):
            return
        error_msg = (
            f'The {name} {value!r} of {self.label!r} needs the cells of a grid or the nodes of a network, but the '
            f'system has a continuous space. Use {alternative}.'
        )
        raise SpaceError(error_msg)

    def _check_partners(self, partners: object, space: BaseSpace | None) -> None:
        """Raise an error for a selection of partners among cells or nodes.

        The error is for a continuous space.
        """
        if isinstance(partners, PartnerSpec) and partners.candidates in GRID_CANDIDATES:
            self._check_grid('candidates', partners.candidates, space, 'a radius, `PartnerSpec(candidates=<radius>)`')

    @classmethod
    def _get_law_classes(cls) -> tuple[type[BaseLaw], ...]:
        """Return the law classes the class lists.

        There are none when a subclass writes its own laws below the list.
        """
        for klass in cls.__mro__:
            declared = vars(klass)
            if 'law_classes' in declared:
                classes: tuple[type[BaseLaw], ...] = declared['law_classes']
                return classes
            if '_get_laws' in declared:
                return ()
        return ()

    @staticmethod
    def _get_accepted(kinds: Sequence[tuple[type, ...]]) -> tuple[type, ...]:
        """Return the classes that some law accepts, none when a law accepts any class."""
        if not all(kinds):
            return ()
        return tuple(dict.fromkeys(kind for classes in kinds for kind in classes))

    @classmethod
    def _get_state_arguments(cls, role: RoleSpec) -> tuple[str, ...]:
        """Return the arguments that name the states a role of one element reads.

        They follow from the type hints.

        Such an argument holds names of states, such as `states: tuple[str, ...]`, which may name derivatives with
        primes, such as `("x", "x'")`. A role of several elements or of Wiener processes reads none.
        """
        if role.variadic or role.noise:
            return ()
        hints = inspect.signature(cls.__init__, eval_str=True).parameters.items()
        return tuple(name for name, argument in hints if cls._is_state_names(argument.annotation))

    @classmethod
    def _is_state_names(cls, annotation: object) -> bool:
        """Return whether a type hint holds names of states, as `tuple[str, ...]` does."""
        hint = cls._get_annotated(annotation)
        origin, args = get_origin(hint), get_args(hint)
        if origin is Union or origin is UnionType:
            return any(cls._is_state_names(arg) for arg in args)
        return origin is tuple and args == (str, Ellipsis)
        if origin in (tuple, Mapping, dict) and args:
            first = args[0]
            keys = get_args(first) if get_origin(first) in (Union, UnionType) else (first,)
            return str in keys
        return False

    def _get_state_names(self, names: Sequence[str]) -> tuple[str, ...]:
        """Return the states that declared names mean.

        They are the names an argument holds, or else the names themselves.
        """
        found: list[str] = []
        for name in names:
            value = getattr(self, name, name)
            if isinstance(value, str):
                found.append(value)
            elif isinstance(value, Mapping | list | tuple):
                found += [item for item in value if isinstance(item, str)]
        return tuple(dict.fromkeys(found))

    def _runs_on_edges(self) -> bool:
        """Return whether the laws run once per edge of a network, false by default."""
        return False

    def _runs_on_hyperedges(self) -> bool:
        """Return whether the laws run once per hyperedge of a network, false by default."""
        return False

    def _get_hyperedge_size(self) -> int | None:
        """Return the size of the hyperedges the laws run on, or `None` for every size."""
        return None

    def _get_layer(self) -> str | None:
        """Return the layer of the network the laws run along, or `None` for its graph."""
        return None

    def _get_grouping(self) -> tuple[str, int, str] | None:
        """Return the kind, size, and reduction of the groups the laws run on, or `None`."""
        return None

    def _check_space(self, space: BaseSpace | None) -> None:
        """Raise an error for a variant the space of the system cannot run.

        There is none by default.

        Args:
            space:
                The space of the system, or `None`.
        """

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Check the interaction on its space, or start the laws of a binding.

        It first raises an error for declarations the interaction cannot take.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the elements it binds, whose
                laws a subclass then sets in `laws_`.

        Raises:
            SpaceError:
                A variant needs a kind of space the system does not have.
            ValidationError:
                The type declares states.
        """
        super()._prepare(context)
        self._check_states()
        if isinstance(context, SimpleNamespace):
            self.laws_: Sequence[BaseComponent] = []
            return
        self._check_space(context)

    def get_requirements(self) -> _InteractionRequirementSpec:
        """Return what the interaction needs.

        It needs the element kinds and states of its roles, and its laws' time and space.

        Returns:
            requirements:
                The roles by name, with the states each reads for the values of this interaction, the law classes
                the class lists, and the classes of time and of space these laws run in.

        Examples:
            >>> from skcomplex.core.components.interactions import RateInteraction
            >>> interaction = RateInteraction('growth', target='rabbits', state='n', rate=lambda s: s.target.n)
            >>> interaction.get_requirements().roles['target'].states
            ('n',)
        """
        laws = self._get_law_classes()
        needs = [law.get_requirements() for law in laws]
        return _InteractionRequirementSpec(
            time=self._get_accepted([need.time for need in needs]),
            space=self._get_accepted([need.space for need in needs]),
            sites=any(need.sites for need in needs),
            roles={
                name: replace(spec, states=self._get_state_names(spec.states or self._get_state_arguments(spec)))
                for name, spec in self._get_component_specs().items()
                if isinstance(spec, RoleSpec)
            },
            laws=laws,
        )
