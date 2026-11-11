"""Signalling cascades of a hormone pulse.

A kinase cascade amplifies the pulse, and the G-protein timer switches it off.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws.utils import pulse
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ...general.reactions import ReactionInteraction
from ..common.kinetics import SpeciesElement

STAGES = ('adenylate_cyclase', 'camp', 'protein_kinase_a', 'phosphorylase_kinase', 'phosphorylase')


class GProteinElement(Element):
    """G protein activated by a hormone pulse and inactivated by its own GTPase.

    The GTPase is the timer that ends the signal.

    The active fraction follows `dg/dt = k_on H(t) (1 - g) - k_gtp g`, where the hormone `H(t)` is a rectangular pulse
    of height `hormone` from `start` for `width`. Once the hormone is gone, the signal ends after the time `1 / k_gtp`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `HormoneCascadeAmplifierSystem` read its parameters.

    Args:
        label:
            The label of the G protein.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        g:
            The initial active fraction.

        k_on:
            The activation rate per unit of hormone.

        k_gtp:
            The rate of GTP hydrolysis, which inactivates the protein.

        hormone:
            The height of the hormone pulse.

        start:
            The time the hormone pulse starts.

        width:
            The duration of the hormone pulse.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import GProteinElement
        >>> GProteinElement('g_protein').k_gtp
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        g: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        k_on: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        k_gtp: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        hormone: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        start: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.g = g
        self.k_on = k_on
        self.k_gtp = k_gtp
        self.hormone = hormone
        self.start = start
        self.width = width


class HormoneCascadeAmplifierSystem(BaseSystem):
    """The adenylate cyclase cascade, which amplifies a hormone pulse.

    The G-protein timer ends the pulse.

    A hormone pulse activates the G protein, which activates adenylate cyclase, which makes cyclic AMP, which activates
    protein kinase A, phosphorylase kinase, and glycogen phosphorylase in turn (Konieczny et al. 4.4.3). Every stage is
    activated at the rate `alpha_i` times the active amount of the stage above it and inactivated at the rate `beta_i`,
    so the integral of every stage over the signal is the integral of the stage above it times the gain `alpha_i /
    beta_i`, and the cascade multiplies the integral of the G-protein signal by the product of the gains, 100,000 with
    the defaults. The G protein inactivates itself by GTP hydrolysis at the rate `k_gtp`, the timer of Konieczny et al.
    4.4.6, so blocking it, as cholera toxin does, prolongs the signal. The system has the active G protein `g_protein.g`
    and the active stages `adenylate_cyclase.x`, `camp.x`, `protein_kinase_a.x`, `phosphorylase_kinase.x`, and
    `phosphorylase.x`.

    Args:
        alpha:
            The activation rates of adenylate cyclase, cyclic AMP, protein kinase A, phosphorylase kinase, and
            phosphorylase.

        beta:
            The inactivation rates of the five stages.

        k_on:
            The activation rate of the G protein per unit of hormone.

        k_gtp:
            The rate of GTP hydrolysis of the G protein.

        hormone:
            The height of the hormone pulse.

        start:
            The time the hormone pulse starts.

        width:
            The duration of the hormone pulse.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The activation and inactivation rates do not give one value per stage.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import HormoneCascadeAmplifierSystem
        >>> [element.label for element in HormoneCascadeAmplifierSystem().prepare().system_.elements][-1]
        'phosphorylase'
    """

    field = 'biology/systems_biology'
    topics = ('signalling', 'cascade', 'amplification', 'hormone', 'g protein', 'ode')
    references = ('Konieczny et al. 2023', 'Heinrich, Neel, and Rapoport 2002')
    until = 30.0
    dt = 0.2

    def __init__(
        self,
        alpha: Sequence[float] = (10.0, 100.0, 1.0, 10.0, 10.0),
        beta: Sequence[float] = (1.0, 2.0, 0.5, 1.0, 1.0),
        *,
        k_on: float = 1.0,
        k_gtp: float = 0.5,
        hormone: float = 1.0,
        start: float = 1.0,
        width: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.alpha = alpha
        self.beta = beta
        self.k_on = k_on
        self.k_gtp = k_gtp
        self.hormone = hormone
        self.start = start
        self.width = width

    def _build_system(self) -> CompositeSystem:
        """Return the components of the adenylate cyclase cascade."""
        if len(self.alpha) != len(STAGES) or len(self.beta) != len(STAGES):
            error_msg = f'Give one activation and one inactivation rate per stage, {len(STAGES)} of each.'
            raise ValidationError(error_msg)
        g_protein = GProteinElement(
            'g_protein',
            k_on=self.k_on,
            k_gtp=self.k_gtp,
            hormone=self.hormone,
            start=self.start,
            width=self.width,
        )
        stages = [SpeciesElement(label) for label in STAGES]
        upstreams = (('g_protein', 'g'), *((label, 'x') for label in STAGES[:-1]))
        activations = [
            ReactionInteraction(f'activation_of_{label}', products={(label, 'x'): 1}, modifiers={upstream: 1}, k=rate)
            for upstream, label, rate in zip(upstreams, STAGES, self.alpha, strict=True)
        ]
        inactivations = [
            ReactionInteraction(f'inactivation_of_{label}', reactants={(label, 'x'): 1}, k=rate)
            for label, rate in zip(STAGES, self.beta, strict=True)
        ]
        switching = [
            ReactionInteraction(
                'g_protein_activation',
                target='g_protein',
                products={'g': 1},
                propensity=lambda s: s.target.k_on
                * pulse(s.t, s.target.start, s.target.width, s.target.hormone)
                * (1 - s.target.g),
            ),
            ReactionInteraction(
                'gtp_hydrolysis',
                target='g_protein',
                reactants={'g': 1},
                propensity=lambda s: s.target.k_gtp * s.target.g,
            ),
        ]
        return CompositeSystem(
            self.label,
            elements=[g_protein, *stages],
            interactions=[*switching, *activations, *inactivations],
        )
