"""Automata over infinite or group alphabets.

They are a self-indexed shift, power series, naturals, reals, and group values.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_gap import SymmetricGroup
from sage.all__sagemath_symbolics import RR, ZZ
from sage.groups.perm_gps.permgroup import PermutationGroup_generic

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import DeclaredElement, Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import LocalRuleLaw
from ....core.components.spaces import BaseSpace, LatticeNeighborhood, LatticeSpace
from ....core.errors import ParameterError
from ....core.specs import DeclaredVariableSpec, PowerSeriesSpec, Role, StateSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array
from ..common.automata import CayleySpace, StateTranslationNeighborhood
from ._groups import free_group
from ._pascal_cell import PascalCellElement

RATIONAL_SERIES = PowerSeriesSpec()
FORWARD = LatticeNeighborhood(((0,), (1,)))
NO_ELEMENT = -1


class GroupCellElement(Element):
    """Cell of an automaton whose alphabet is a group.

    It holds a group element by its number. The number is the site of the element in a group space, or its position in
    the list of a finite group, and a negative number names no element, the fill value read outside a ball. With
    `space`, the element holds one cell on every site.

    Args:
        label:
            The label of the cells.

        value:
            The number of the initial group element, 0 for the identity, or one number per cell.

        space:
            The space with one cell on every site, such as a lattice or a group, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GroupCellElement
        >>> GroupCellElement('cell', value=2).value
        2
    """

    def __init__(
        self,
        label: str,
        *,
        value: Annotated[float, ZZ] = 0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.value = value


class PowerSeriesCellElement(DeclaredElement):
    """Cell of an automaton over a ring `K[[t]]` of truncated formal power series.

    The series are truncated at an order. The cell declares the state `value` in the ring it is given, so it holds the
    coefficients `a_0, ..., a_{order - 1}` of a series, which a local rule reads as one row per cell.

    Args:
        label:
            The label of the cell or the lattice group.

        ring:
            The truncated power series ring of the values.

        variables:
            Declared states and parameters besides `value`, or values that replace its default, by name.

    Examples:
        >>> from skcomplex.core.specs import PowerSeriesSpec
        >>> from skcomplex.systems.computer_science.automata import PowerSeriesCellElement
        >>> PowerSeriesCellElement('cell', ring=PowerSeriesSpec(3)).get_value('value').tolist()
        [0.0, 0.0, 0.0]
    """

    def __init__(
        self,
        label: str,
        *,
        ring: PowerSeriesSpec = RATIONAL_SERIES,
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
    ) -> None:
        super().__init__(label, variables=variables)
        self.ring = ring

    def _declare_variables(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the state `value` in the ring and the given variables.

        The state has the zero series as its default.
        """
        value = DeclaredVariableSpec(np.zeros(self.ring.order), StateSpec(domain=self.ring))
        return {'value': value, **(self.variables or {})}


class RealCellElement(Element):
    """Cell of an automaton over the real numbers.

    With `space`, the element holds one cell on every site.

    Args:
        label:
            The label of the cells.

        value:
            The initial value, or one value per cell.

        space:
            The space with one cell on every site, such as a lattice, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import RealCellElement
        >>> RealCellElement('cell', value=0.5).value
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        value: Annotated[float, RR] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.value = value


class SelfIndexedShiftInteraction(Interaction):
    """Map `τ(x)(g) = x(g · x(g))` of cells whose values are elements of the group of their sites.

    Every cell takes the value of the cell at the product of its site and its own value, so the cell it reads moves
    with the configuration.

    Args:
        label:
            The label of the interaction.

        target:
            The cells on a group space, whose state `value` the map changes.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import SelfIndexedShiftInteraction
        >>> SelfIndexedShiftInteraction('translation', target='cells').target
        'cells'
    """

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _translate(value: Array) -> Array:
        """Return the value read at the product of a site and its own value, `x(g · x(g))`."""
        return value

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the read at the product of every site and its value as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            value = context.target.value
            self.laws_: list[BaseComponent] = [
                LocalRuleLaw(value, StateTranslationNeighborhood(), self._translate, label=self.label),
            ]


class PowerSeriesDifferenceInteraction(Interaction):
    """Difference `τ(x)(n) = x(n) - t x(n + 1)` of cells holding truncated power series.

    The coefficients of every series sit on the last axis of the state `value`, and the product with `t` shifts them
    by one place in the ring.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `value` the difference changes.

        ring:
            The truncated power series ring of the values.

    Examples:
        >>> from skcomplex.core.specs import PowerSeriesSpec
        >>> from skcomplex.systems.computer_science.automata import PowerSeriesDifferenceInteraction
        >>> PowerSeriesDifferenceInteraction('difference', target='cells', ring=PowerSeriesSpec(3)).ring.order
        3
    """

    def __init__(self, label: str, *, target: Role = 'cells', ring: PowerSeriesSpec = RATIONAL_SERIES) -> None:
        super().__init__(label)
        self.target = target
        self.ring = ring

    def _get_rule(self) -> Callable[[Array, Array], Array]:
        """Return `x(n) - t x(n + 1)` in the ring, with the coefficients on the last axis."""
        ring = self.ring
        variable = ring.get_coefficients_of(ring.ring.gen())

        def difference(here: Array, right: Array) -> Array:
            return ring.reduce(here - ring.multiply(variable, right))

        return difference

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the difference with the cell to the right as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            value = context.target.value
            self.laws_: list[BaseComponent] = [LocalRuleLaw(value, FORWARD, self._get_rule(), label=self.label)]


class SubtractionRuleInteraction(Interaction):
    """Truncated difference `τ(x)(n) = max(0, x(n) - x(n + 1))` of cells holding natural numbers.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `value` the difference changes.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import SubtractionRuleInteraction
        >>> SubtractionRuleInteraction('subtraction', target='cells').target
        'cells'
    """

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _subtract(here: Array, right: Array) -> Array:
        """Return the truncated difference `max(0, x(n) - x(n + 1))` of the naturals."""
        return np.maximum(0.0, here - right)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the truncated difference with the cell to the right as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            value = context.target.value
            self.laws_: list[BaseComponent] = [LocalRuleLaw(value, FORWARD, self._subtract, label=self.label)]


class QuadraticRuleInteraction(Interaction):
    """Rule `τ(x)(n) = x(n + 1) - x(n)^2` of cells holding real numbers.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `value` the rule changes.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import QuadraticRuleInteraction
        >>> QuadraticRuleInteraction('quadratic', target='cells').target
        'cells'
    """

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _get_quadratic(here: Array, right: Array) -> Array:
        """Return `x(n + 1) - x(n)^2`."""
        return right - here**2

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rule on a cell and the cell to its right as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            value = context.target.value
            self.laws_: list[BaseComponent] = [LocalRuleLaw(value, FORWARD, self._get_quadratic, label=self.label)]


class GroupDifferenceInteraction(Interaction):
    """Difference `τ(x)(n) = x(n)^-1 x(n + 1)` of cells holding elements of a finite group.

    Cells hold group elements by their number in the Cayley table of Sage, the identity first.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `value` the difference changes.

        group:
            The finite group, a Sage permutation group, or `None` for the symmetric group of degree 3.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GroupDifferenceInteraction
        >>> GroupDifferenceInteraction('difference', target='cells').group is None
        True
    """

    def __init__(self, label: str, *, target: Role = 'cells', group: PermutationGroup_generic | None = None) -> None:
        super().__init__(label)
        self.target = target
        self.group = group

    def _get_rule(self) -> Callable[[Array, Array], Array]:
        """Return the number of `x(n)^-1 x(n + 1)`, read from the Cayley table and the inverses."""
        alphabet = SymmetricGroup(3) if self.group is None else self.group
        product = np.asarray(alphabet.cayley_table().table(), dtype=np.intp)
        inverse = np.argmin(product, axis=1).astype(np.intp)

        def difference(here: Array, right: Array) -> Array:
            return product[inverse[here.astype(np.intp)], right.astype(np.intp)].astype(float)

        return difference

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the difference with the cell to the right as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            value = context.target.value
            self.laws_: list[BaseComponent] = [LocalRuleLaw(value, FORWARD, self._get_rule(), label=self.label)]


class SelfIndexedShiftSystem(BaseSystem):
    """The map `τ(x)(g) = x(g · x(g))` over the group as alphabet.

    It is continuous and equivariant but no automaton. Every cell holds a group element, the number of its site, and
    takes the value of the cell at the product of its site and its value, so the cell it reads moves with the
    configuration. On an infinite group no finite memory set describes the map, which is continuous and commutes with
    the shifts but is no cellular automaton (Ceccherini-Silberstein and Coornaert 2023, Example 1.8.2). The simulation
    truncates the group: on a ball, a product outside it reads `-1`, a value that names no element and stays, and a
    finite group, without a radius, keeps every product. The system has the group elements `cells.value`.

    Args:
        group:
            The group, a Sage or GAP group, or `None` for the free group of rank 2.

        radius:
            The radius of the ball, or `None` for a finite group.

        initial:
            The numbers of the initial elements, one per site, or `None` for random ones.

        random_state:
            The seed or generator of the random initial elements, drawn uniformly among the sites.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import SelfIndexedShiftSystem
        >>> SelfIndexedShiftSystem(radius=2).prepare().simulate(until=1).get_values('cells.value').shape
        (2, 17)
    """

    field = 'computer_science/automata'
    topics = ('group', 'infinite alphabet', 'equivariant', 'free group')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 6
    dt = 1

    def __init__(
        self,
        group: object | None = None,
        *,
        radius: int | None = 4,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.group = group
        self.radius = radius
        self.initial = initial
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the cells on a ball of the group and the self-indexed shift."""
        super()._prepare(context)
        space = CayleySpace(free_group(2) if self.group is None else self.group, ball=self.radius, fill=NO_ELEMENT)
        size = space.prepare().size_
        if self.initial is None:
            values = np.random.default_rng(self.random_state).integers(0, size, size).astype(float)
        else:
            values = np.asarray(self.initial, dtype=float)
        self.system_ = System(
            self.label,
            elements=[GroupCellElement('cells', value=values, space=space)],
            interactions=[SelfIndexedShiftInteraction('translation', target='cells')],
        )


class PowerSeriesAutomatonSystem(BaseSystem):
    """The automaton `τ(x)(n) = x(n) - t x(n + 1)` over the power series `K[[t]]`.

    The power series are formal, on a ring. The alphabet `K[[t]]` has infinite dimension, and the automaton is bijective
    with the inverse `sigma(y)(n) = sum_j t^j y(n + j)`, which is not continuous, so it is no cellular automaton
    (Ceccherini-Silberstein and Coornaert 2023, Exercise 1.37). The cells hold the series truncated at the order, which
    keeps every coefficient below it exact, since the coefficient of `t^i` of the image reads coefficients of `t^i` and
    `t^(i - 1)` alone. From the constant series 1 at one cell, the cells at step `s` hold `C(s, k) (-t)^k` at the `k`-th
    cell to the left, as long as `k` stays below the order. The system has the coefficients `cells.value`.

    Args:
        size:
            The number of cells of the ring.

        order:
            The number of coefficients kept.

        field:
            The field `K`, `'QQ'`, `'RR'`, or `'GF(p)'`.

        initial:
            The initial coefficients, an array of shape `(size, order)`, or `None` for the series 1 in the middle cell.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import PowerSeriesAutomatonSystem
        >>> PowerSeriesAutomatonSystem(5, order=3).prepare().system_.elements[0].values['value'].shape
        (5, 3)
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'infinite alphabet', 'power series', 'bijective')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 12
    dt = 1

    def __init__(
        self,
        size: int = 31,
        *,
        order: int = 8,
        field: str = 'QQ',
        initial: npt.ArrayLike | None = None,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.order = order
        self.field = field
        self.initial = initial

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the cells of truncated series and their difference."""
        super()._prepare(context)
        ring = PowerSeriesSpec(self.order, self.field)
        if self.initial is None:
            values = np.zeros((self.size, self.order))
            values[self.size // 2, 0] = 1.0
        else:
            values = np.asarray(self.initial, dtype=float)
        cells = GroupElement(
            'cells',
            element=PowerSeriesCellElement,
            space=LatticeSpace(self.size),
            values={'ring': ring, 'value': values},
            on='sites',
        )
        self.system_ = System(
            self.label,
            elements=[cells],
            interactions=[PowerSeriesDifferenceInteraction('difference', target='cells', ring=ring)],
        )


class SubtractionAutomatonSystem(BaseSystem):
    """The automaton `τ(x)(n) = max(0, x(n) - x(n + 1))` over the naturals.

    Its image is not closed. The image of the configurations `x_M`, equal to `M` left of the origin and to
    `max(0, M - n)` from it, is the block of `M` ones from the origin. These blocks converge to the configuration of
    ones on the half line, which is no image, so the image of the automaton over the infinite alphabet `N` is not closed
    (Ceccherini-Silberstein and Coornaert 2023, Example 3.3.3). The cells sit on a segment with fixed ends that read 0,
    and start at `x_M` with `M = height`, its origin a quarter of the way along. The system has the cells `cells.value`.

    Args:
        size:
            The number of cells.

        height:
            The value `M` of the staircase.

        initial:
            The initial values, or `None` for the staircase.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ParameterError:
            The height is negative.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import SubtractionAutomatonSystem
        >>> SubtractionAutomatonSystem(12, height=3).prepare().system_.elements[0].value.tolist()
        [3.0, 3.0, 3.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'infinite alphabet', 'naturals', 'non-closed image')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 10
    dt = 1

    def __init__(
        self,
        size: int = 41,
        *,
        height: int = 10,
        initial: npt.ArrayLike | None = None,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.height = height
        self.initial = initial

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the staircase on a segment and the truncated difference."""
        super()._prepare(context)
        if self.height < 0:
            error_msg = f'The height of the staircase is a natural number, but it is {self.height}.'
            raise ParameterError(error_msg)
        if self.initial is None:
            origin = self.size // 4
            steps = np.arange(self.size) - origin
            values = np.where(steps < 0, self.height, np.maximum(0, self.height - steps)).astype(float)
        else:
            values = np.asarray(self.initial, dtype=float)
        self.system_ = System(
            self.label,
            elements=[PascalCellElement('cells', value=values, space=LatticeSpace(self.size, boundary='fixed'))],
            interactions=[SubtractionRuleInteraction('subtraction', target='cells')],
        )


class RealQuadraticAutomatonSystem(BaseSystem):
    """The automaton `τ(x)(n) = x(n + 1) - x(n)^2` over the reals on a ring.

    It is pre-injective but not injective, since the constant configurations 0 and 1 both map to 0, and its image is
    dense but neither closed nor everything (Ceccherini-Silberstein and Coornaert 2023, Exercise 5.64). The square pulls
    the cells down at every step, and cells far from the origin run off to minus infinity. The system has the cells
    `cells.value`.

    Args:
        size:
            The number of cells of the ring.

        amplitude:
            The half width of the uniform initial values.

        initial:
            The initial values, or `None` for random ones.

        random_state:
            The seed or generator of the random initial values.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import RealQuadraticAutomatonSystem
        >>> RealQuadraticAutomatonSystem(8).prepare().simulate(until=1).get_values('cells.value').shape
        (2, 8)
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'infinite alphabet', 'reals', 'pre-injective')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 6
    dt = 1

    def __init__(
        self,
        size: int = 64,
        *,
        amplitude: float = 0.25,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.amplitude = amplitude
        self.initial = initial
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a ring of real cells and the quadratic rule."""
        super()._prepare(context)
        if self.initial is None:
            values = np.random.default_rng(self.random_state).uniform(-self.amplitude, self.amplitude, self.size)
        else:
            values = np.asarray(self.initial, dtype=float)
        self.system_ = System(
            self.label,
            elements=[RealCellElement('cells', value=values, space=LatticeSpace(self.size))],
            interactions=[QuadraticRuleInteraction('quadratic', target='cells')],
        )


class GroupDifferenceAutomatonSystem(BaseSystem):
    """The automaton `τ(x)(n) = x(n)^-1 x(n + 1)` whose alphabet is a finite group.

    It runs on a ring. Over the integers, with the generator `s0 = 1` of infinite order and a nontrivial group alphabet,
    the automaton is surjective but not injective (Ceccherini-Silberstein and Coornaert 2023, Exercise 3.24): every
    constant configuration maps to the identity. The ring is the finite quotient of order `size`, where the product of
    the image around the ring is the identity. Cells hold group elements by their number in the Cayley table of Sage,
    the identity first. The system has the group elements `cells.value`.

    Args:
        size:
            The number of cells of the ring.

        group:
            The finite group, a Sage permutation group, or `None` for the symmetric group of degree 3.

        initial:
            The initial numbers of the elements, or `None` for random ones.

        random_state:
            The seed or generator of the random initial elements.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GroupDifferenceAutomatonSystem
        >>> GroupDifferenceAutomatonSystem(10).prepare().simulate(until=1).get_values('cells.value').shape
        (2, 10)
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'group alphabet', 'surjective', 'symmetric group')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 30
    dt = 1

    def __init__(
        self,
        size: int = 60,
        *,
        group: PermutationGroup_generic | None = None,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.group = group
        self.initial = initial
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a ring of group elements and their difference."""
        super()._prepare(context)
        order = int((SymmetricGroup(3) if self.group is None else self.group).order())
        if self.initial is None:
            values = np.random.default_rng(self.random_state).integers(0, order, self.size).astype(float)
        else:
            values = np.asarray(self.initial, dtype=float)
        self.system_ = System(
            self.label,
            elements=[GroupCellElement('cells', value=values, space=LatticeSpace(self.size))],
            interactions=[GroupDifferenceInteraction('difference', target='cells', group=self.group)],
        )
