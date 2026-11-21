"""Electrical components that join the nodes of a circuit.

Each component adds its current into the balance of every node it joins.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import SR, Expression, exp, oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, ConstraintLaw
from .....core.components.spaces import BaseSpace
from .....core.types import Role

POSITIVE = RealSet.open(0, oo)


class BranchInteraction(Interaction):
    """Component between the nodes `a` and `b` whose current flows from `a` to `b`.

    Every node holds its potential `v`, and the currents of the components that join a node add into its balance,
    which is zero, as Kirchhoff's current law says. A subclass returns its current from `_get_current`, or its own laws
    from `_get_branch_laws` when the current is a value of its own. A resistor, a diode, and a source in a circuit are
    such components, as a pipe between two tanks or a road between two towns would be.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        a:
            The path of the node the current leaves.

        b:
            The path of the node the current enters.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import ResistorInteraction
        >>> ResistorInteraction('load', a='top', b='bottom', R=2.0).b
        'bottom'
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (ConstraintLaw,)

    def __init__(self, label: str, *, a: Role, b: Role) -> None:
        super().__init__(label)
        self.a = a
        self.b = b

    def _get_current(self, s: SimpleNamespace) -> Expression:
        """Return the current from `a` to `b`, zero for a component that sets its laws in `_get_branch_laws`."""
        del s
        return SR.zero()

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the component, the current into the balances of both nodes by default."""
        current = self._get_current(s)
        return [ConstraintLaw(-current, 0, target=s.a.v), ConstraintLaw(current, 0, target=s.b.v)]

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the laws of the component.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the nodes it joins.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = self._get_branch_laws(context)


class ResistorInteraction(BranchInteraction):
    """Resistor that obeys Ohm's law, with the current `(a.v - b.v) / R`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        a:
            The path of the node the current leaves.

        b:
            The path of the node the current enters.

        R:
            The resistance.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import ResistorInteraction
        >>> ResistorInteraction('load', a='top', b='bottom', R=2.0).R
        2.0
    """

    def __init__(self, label: str, *, a: Role, b: Role, R: Annotated[float, POSITIVE] = 1.0) -> None:
        super().__init__(label, a=a, b=b)
        self.R = R

    def _get_current(self, s: SimpleNamespace) -> Expression:
        """Return the current of Ohm's law."""
        current: Expression = (s.a.v - s.b.v) / s.R
        return current


class CapacitorInteraction(BranchInteraction):
    """Capacitor whose voltage `u` and current `i` live on the element `branch`.

    It has `C u' = i` and `u = a.v - b.v`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        a:
            The path of the node the current leaves.

        b:
            The path of the node the current enters.

        branch:
            The path of the element that holds the voltage `u` and the current `i` of the capacitor.

        C:
            The capacitance.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.engineering.common.electrical import (
        ...     CapacitorInteraction,
        ...     GroundInteraction,
        ...     ResistorInteraction,
        ... )
        >>> nodes = [Element(name, values={'v': 0.0}) for name in ('top', 'bottom')]
        >>> branches = [Element('tank', values={'u': 1.0, 'i': 0.0}), Element('earth', values={'i': 0.0})]
        >>> interactions = [
        ...     CapacitorInteraction('capacitor', a='top', b='bottom', branch='tank', C=0.5),
        ...     ResistorInteraction('load', a='top', b='bottom', R=2.0),
        ...     GroundInteraction('ground', node='bottom', branch='earth'),
        ... ]
        >>> system = System(elements=nodes + branches, interactions=interactions).prepare()
        >>> round(float(system.simulate(until=1.0, dt=1.0).get_values('tank.u')[-1]), 4)
        0.3679
    """

    def __init__(
        self,
        label: str,
        *,
        a: Role,
        b: Role,
        branch: Role,
        C: Annotated[float, POSITIVE] = 1.0,
    ) -> None:
        super().__init__(label, a=a, b=b)
        self.branch = branch
        self.C = C

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the charge balance, the voltage, and the current into the balances of both nodes."""
        u, i = s.branch.u, s.branch.i
        return [
            ConstraintLaw(s.C * u.diff(s.t), i, target=u),
            ConstraintLaw(u, s.a.v - s.b.v, target=i),
            ConstraintLaw(-i, 0, target=s.a.v),
            ConstraintLaw(i, 0, target=s.b.v),
        ]


class InductorInteraction(BranchInteraction):
    """Inductor whose current `i` lives on the element `branch`, with `L i' = a.v - b.v`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        a:
            The path of the node the current leaves.

        b:
            The path of the node the current enters.

        branch:
            The path of the element that holds the current `i` of the inductor.

        L:
            The inductance.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import InductorInteraction
        >>> InductorInteraction('coil', a='top', b='bottom', branch='coil', L=0.5).L
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        a: Role,
        b: Role,
        branch: Role,
        L: Annotated[float, POSITIVE] = 1.0,
    ) -> None:
        super().__init__(label, a=a, b=b)
        self.branch = branch
        self.L = L

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the voltage of the inductor and its current into the balances of both nodes."""
        i = s.branch.i
        return [
            ConstraintLaw(s.L * i.diff(s.t), s.a.v - s.b.v, target=i),
            ConstraintLaw(-i, 0, target=s.a.v),
            ConstraintLaw(i, 0, target=s.b.v),
        ]


class VoltageSourceInteraction(BranchInteraction):
    """Ideal source that holds the node `a` at the voltage `V` above the node `b`.

    Its current `i`, from `a` through the source to `b`, lives on the element `branch`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        a:
            The path of the positive node.

        b:
            The path of the negative node.

        branch:
            The path of the element that holds the current `i` of the source.

        V:
            The voltage.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import VoltageSourceInteraction
        >>> VoltageSourceInteraction('battery', a='plus', b='minus', branch='cell', V=9.0).V
        9.0
    """

    def __init__(self, label: str, *, a: Role, b: Role, branch: Role, V: float = 1.0) -> None:
        super().__init__(label, a=a, b=b)
        self.branch = branch
        self.V = V

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the voltage of the source as the equation of its current, and the current into both balances."""
        i = s.branch.i
        return [
            ConstraintLaw(s.a.v - s.b.v, s.V, target=i),
            ConstraintLaw(-i, 0, target=s.a.v),
            ConstraintLaw(i, 0, target=s.b.v),
        ]


class CurrentSourceInteraction(BranchInteraction):
    """Ideal source that drives the constant current `current` from `b` through the source to `a`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        a:
            The path of the node the source drives the current into.

        b:
            The path of the node the source draws the current from.

        current:
            The current the source drives into `a`.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import CurrentSourceInteraction
        >>> CurrentSourceInteraction('source', a='top', b='bottom', current=0.5).current
        0.5
    """

    def __init__(self, label: str, *, a: Role, b: Role, current: float = 1.0) -> None:
        super().__init__(label, a=a, b=b)
        self.current = current

    def _get_current(self, s: SimpleNamespace) -> Expression:
        """Return the current from `a` to `b`, the opposite of the current the source drives."""
        current: Expression = -s.current
        return current


class GroundInteraction(Interaction):
    """Reference that holds the potential of a node at zero.

    The current that the ground feeds into the node, `i` of the element `branch`, balances the node.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        node:
            The path of the node held at zero potential.

        branch:
            The path of the element that holds the current `i` from the ground into the node.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import GroundInteraction
        >>> GroundInteraction('ground', node='bottom', branch='earth').node
        'bottom'
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (ConstraintLaw,)

    def __init__(self, label: str, *, node: Role, branch: Role) -> None:
        super().__init__(label)
        self.node = node
        self.branch = branch

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the zero potential as the equation of the current, and the current into the balance of the node.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the node and the branch.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            i = context.branch.i
            self.laws_: list[BaseLaw] = [
                ConstraintLaw(context.node.v, 0, target=i),
                ConstraintLaw(i, 0, target=context.node.v),
            ]


class DiodeInteraction(BranchInteraction):
    """Exponential diode, with the current `Is (exp((a.v - b.v) / Vt) - 1)`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        a:
            The path of the anode.

        b:
            The path of the cathode.

        Is:
            The saturation current.

        Vt:
            The thermal voltage.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import DiodeInteraction
        >>> DiodeInteraction('diode', a='anode', b='cathode').Vt
        0.025
    """

    def __init__(
        self,
        label: str,
        *,
        a: Role,
        b: Role,
        Is: Annotated[float, POSITIVE] = 1e-6,
        Vt: Annotated[float, POSITIVE] = 0.025,
    ) -> None:
        super().__init__(label, a=a, b=b)
        self.Is = Is
        self.Vt = Vt

    def _get_current(self, s: SimpleNamespace) -> Expression:
        """Return the current of the diode."""
        current: Expression = s.Is * (exp((s.a.v - s.b.v) / s.Vt) - 1)
        return current


class ChuaDiodeInteraction(BranchInteraction):
    """Nonlinear resistor of Chua's circuit.

    At the voltage `u = a.v - b.v` it has the current `m1 u + (m0 - m1) (|u + 1| - |u - 1|) / 2`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        a:
            The path of the node the current leaves.

        b:
            The path of the node the current enters.

        m0:
            The slope of the characteristic inside the breakpoints.

        m1:
            The slope of the characteristic outside the breakpoints.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import ChuaDiodeInteraction
        >>> round(ChuaDiodeInteraction('diode', a='top', b='bottom').m0, 3)
        -1.143
    """

    def __init__(self, label: str, *, a: Role, b: Role, m0: float = -8 / 7, m1: float = -5 / 7) -> None:
        super().__init__(label, a=a, b=b)
        self.m0 = m0
        self.m1 = m1

    def _get_current(self, s: SimpleNamespace) -> Expression:
        """Return the current of the nonlinear resistor."""
        voltage = s.a.v - s.b.v
        current: Expression = s.m1 * voltage + (s.m0 - s.m1) * (abs(voltage + 1) - abs(voltage - 1)) / 2
        return current


class OperationalAmplifierInteraction(Interaction):
    """Ideal operational amplifier.

    Its inputs `p` and `n` draw no current, and its output drives the node `out` with the current `i` of the element
    `branch` that holds both inputs at equal potentials. The output current returns through the supply of the
    amplifier, so no balance takes it back.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        p:
            The path of the noninverting input node.

        n:
            The path of the inverting input node.

        out:
            The path of the output node.

        branch:
            The path of the element that holds the output current `i`.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import OperationalAmplifierInteraction
        >>> OperationalAmplifierInteraction('amplifier', p='ground', n='sum', out='output', branch='drive').out
        'output'
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (ConstraintLaw,)

    def __init__(self, label: str, *, p: Role, n: Role, out: Role, branch: Role) -> None:
        super().__init__(label)
        self.p = p
        self.n = n
        self.out = out
        self.branch = branch

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the equal potentials of the inputs as the equation of the output current, which feeds the output node.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the nodes it joins.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            i = context.branch.i
            self.laws_: list[BaseLaw] = [
                ConstraintLaw(context.p.v, context.n.v, target=i),
                ConstraintLaw(i, 0, target=context.out.v),
            ]
