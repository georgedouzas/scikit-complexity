"""Three-enzyme circuits that adapt to a step of their input."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import step
from ....core.components.spaces import BaseSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

NODES = ('A', 'B', 'C')
NEGATIVE_FEEDBACK_BUFFER = ((0, -1, 0), (0, 0, 1), (1, 0, 0))
RATES = ((0.419, 0.3, 5.061), (7.809, 9.355, 4.852), (0.735, 1.176, 1.928), (7.282, 0.182, 7.914))
CONSTANTS = ((0.1633, 2.5556, 0.0323), (0.0154, 0.3059, 0.0641), (4.3698, 0.0021, 0.3178), (0.0379, 0.0202, 1.802))
BASAL_RATES = (3.092, 0.712, 0.337)
BASAL_CONSTANTS = (0.143, 0.0031, 0.1074)


class EnzymeNodeElement(Element):
    """Enzyme of a signalling circuit, the active fraction `X` of a pool.

    Other enzymes activate and deactivate the pool.

    Args:
        label:
            The label of the enzyme.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        X:
            The initial active fraction.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import EnzymeNodeElement
        >>> EnzymeNodeElement('A', X=0.2).X
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        X: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.X = X


class EnzymeRegulationInteraction(Interaction):
    """Michaelis-Menten activation or deactivation of an enzyme by another enzyme.

    Both enzymes belong to one circuit.

    An activating source converts the inactive fraction of the target at the rate `k X_s (1 - X) / (1 - X + K)`, and a
    deactivating source the active fraction at the rate `k X_s X / (X + K)`, the kinetics of Ma et al. (2009).

    Args:
        label:
            The label of the interaction.

        target:
            The path of the regulated enzyme.

        source:
            The path of the regulating enzyme.

        sign:
            1 for activation or -1 for deactivation.

        k:
            The catalytic rate.

        K:
            The Michaelis constant.

    Raises:
        ValidationError:
            The sign is neither 1 nor -1.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import EnzymeRegulationInteraction
        >>> EnzymeRegulationInteraction('ab', target='B', source='A', sign=-1).sign
        -1
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        sign: int = 1,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.sign = sign
        self.k = k
        self.K = K

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Check the sign.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The sign is neither 1 nor -1.
        """
        super()._prepare(context)
        if self.sign not in {1, -1}:
            error_msg = f'The sign of a regulation must be 1 or -1, not {self.sign}.'
            raise ValidationError(error_msg)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the conversion of the target by the source.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.X, self._get_conversion(s.source.X, s.target.X, s.k, s.K))]

    def _get_conversion(self, regulator: Expression, X: Expression, k: Expression, K: Expression) -> Expression:
        """Return the Michaelis-Menten conversion of the target by a regulator.

        The conversion follows the sign of the interaction.
        """
        if self.sign > 0:
            return k * regulator * (1 - X) / (1 - X + K)
        return -k * regulator * X / (X + K)


class EnzymeStimulusInteraction(Interaction):
    """Activation or deactivation of an enzyme by an outside stimulus or a basal enzyme.

    A basal enzyme has a fixed concentration.

    The regulator has the concentration `level`, which rises by `change` at the time `at`, as the input `I` of a
    circuit, and stays fixed for a basal enzyme `F` with no change. It activates the target at the rate `k I (1 - X) /
    (1 - X + K)` or deactivates it at the rate `k I X / (X + K)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the regulated enzyme.

        sign:
            1 for activation or -1 for deactivation.

        k:
            The catalytic rate.

        K:
            The Michaelis constant.

        level:
            The concentration of the regulator before the change.

        change:
            The step of the concentration at the time `at`.

        at:
            The time of the step.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import EnzymeStimulusInteraction
        >>> EnzymeStimulusInteraction('input', target='A', level=0.5, change=0.1, at=300.0).change
        0.1
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        sign: int = 1,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        level: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        change: float = 0.0,
        at: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.sign = sign
        self.k = k
        self.K = K
        self.level = level
        self.change = change
        self.at = at

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the conversion of the target by the stimulus.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        regulator = s.level + step(s.t, self.at, self.change)
        return [RateLaw(s.target.X, self._get_conversion(regulator, s.target.X, s.k, s.K))]

    def _get_conversion(self, regulator: Expression, X: Expression, k: Expression, K: Expression) -> Expression:
        """Return the Michaelis-Menten conversion of the target by the stimulus.

        The conversion follows the sign of the interaction.
        """
        if self.sign > 0:
            return k * regulator * (1 - X) / (1 - X + K)
        return -k * regulator * X / (X + K)


class EnzymeAdaptationCircuitSystem(BaseSystem):
    """A circuit of three enzymes A, B, and C of Ma et al.

    The output C of the circuit may adapt to a step of the input.

    The input `I` activates A, and `links[i][j]`, 1, -1, or 0, says whether enzyme `j` activates, deactivates, or leaves
    enzyme `i`, with Michaelis-Menten kinetics of the rate `k[j][i]` and the constant `K[j][i]`, the last row of `k` and
    `K` for the input. A node with no activator gets a basal activating enzyme and one with no deactivator a basal
    deactivating enzyme, both of concentration 0.5 with the constants `k_basal` and `K_basal`. The input steps from
    `level` by `change` at the time `at`. Ma et al. enumerated the 16,038 topologies of three nodes whose input reaches
    C and found that adaptation, a sensitive transient followed by a precise return, needs a negative feedback loop with
    a buffer node or an incoherent feedforward loop with a proportioner node. The default is the negative feedback loop
    A to C to B, B deactivating A, with a parameter set that adapts: the buffer B works in saturation, so its steady
    state fixes the output. The system has the active fractions `A.X`, `B.X`, and `C.X`.

    Args:
        links:
            The signs of the regulations, one row per regulated node and one column per regulating node.

        k:
            The catalytic rates, one row per regulating node and the input last, one column per regulated node.

        K:
            The Michaelis constants, in the layout of `k`.

        k_basal:
            The catalytic rates of the basal enzymes of every node.

        K_basal:
            The Michaelis constants of the basal enzymes of every node.

        level:
            The input before the step.

        change:
            The step of the input.

        at:
            The time of the step.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The links are not a three by three matrix of 1, -1, and 0.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import EnzymeAdaptationCircuitSystem
        >>> [interaction.label for interaction in EnzymeAdaptationCircuitSystem().prepare().system_.interactions][:3]
        ['input', 'B_A', 'C_B']
    """

    field = 'biology/systems_biology'
    topics = ('adaptation', 'network motifs', 'negative feedback', 'incoherent feedforward', 'ode')
    references = ('Ma et al. 2009', 'Raman 2021')
    until = 800.0
    dt = 1.0

    def __init__(
        self,
        links: Sequence[Sequence[int]] = NEGATIVE_FEEDBACK_BUFFER,
        *,
        k: Sequence[Sequence[float]] = RATES,
        K: Sequence[Sequence[float]] = CONSTANTS,
        k_basal: Sequence[float] = BASAL_RATES,
        K_basal: Sequence[float] = BASAL_CONSTANTS,
        level: float = 0.5,
        change: float = 0.1,
        at: float = 300.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.links = links
        self.k = k
        self.K = K
        self.k_basal = k_basal
        self.K_basal = K_basal
        self.level = level
        self.change = change
        self.at = at

    def _build_system(self) -> CompositeSystem:
        """Return the parts of a circuit of three enzymes A, B, and C of Ma et al."""
        signs = np.asarray(self.links)
        if signs.shape != (3, 3) or not np.isin(signs, (-1, 0, 1)).all():
            error_msg = f'The links must be a 3 by 3 matrix of 1, -1, and 0, not {self.links!r}.'
            raise ValidationError(error_msg)
        rates, constants = np.asarray(self.k, dtype=float), np.asarray(self.K, dtype=float)
        interactions: list[BaseInteraction] = [
            EnzymeStimulusInteraction(
                'input',
                target='A',
                k=float(rates[3, 0]),
                K=float(constants[3, 0]),
                level=self.level,
                change=self.change,
                at=self.at,
            ),
        ]
        for target, source in zip(*np.nonzero(signs), strict=True):
            interactions.append(
                EnzymeRegulationInteraction(
                    f'{NODES[source]}_{NODES[target]}',
                    target=NODES[target],
                    source=NODES[source],
                    sign=int(signs[target, source]),
                    k=float(rates[source, target]),
                    K=float(constants[source, target]),
                ),
            )
        for index, node in enumerate(NODES):
            activated = index == 0 or (signs[index] > 0).any()
            for sign, present in ((1, activated), (-1, (signs[index] < 0).any())):
                if not present:
                    interactions.append(
                        EnzymeStimulusInteraction(
                            f'basal_{"activation" if sign > 0 else "deactivation"}_{node}',
                            target=node,
                            sign=sign,
                            k=float(self.k_basal[index]),
                            K=float(self.K_basal[index]),
                        ),
                    )
        return CompositeSystem(
            self.label,
            elements=[EnzymeNodeElement(node) for node in NODES],
            interactions=interactions,
        )
