"""Signalling cascades: a hormone pulse amplified by a kinase cascade and switched off by the G-protein timer."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....core.base import Domain, ValidationError, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Law, Rate, Symbols, pulse
from ....core.runtime import register_model
from ....core.system import System

STAGES = ('adenylate_cyclase', 'camp', 'protein_kinase_a', 'phosphorylase_kinase', 'phosphorylase')


class GProtein(Element):
    """G protein activated by a hormone pulse and inactivated by its own GTPase, the timer that ends the signal.

    The active fraction follows `dg/dt = k_on H(t) (1 - g) - k_gtp g`, where the hormone `H(t)` is a rectangular pulse
    of height `hormone` from `start` for `width`. Once the hormone is gone, the signal ends after the time `1 / k_gtp`.

    Args:
        label:
            The label of the G protein.

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
        >>> from skcomplex.models.biology.systems_biology import GProtein
        >>> GProtein('g_protein').k_gtp
        0.5
    """

    g: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    k_on: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    k_gtp: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    hormone: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    start: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    width: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the activation by the hormone pulse and the inactivation by GTP hydrolysis.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        hormone = pulse(s.t, s.start, s.width, s.hormone)
        return [Rate(s.g, s.k_on * hormone * (1 - s.g) - s.k_gtp * s.g)]


class CascadeStage(Element):
    """Stage of a signalling cascade, an active enzyme or a messenger, which is inactivated at a constant rate.

    Args:
        label:
            The label of the stage.

        x:
            The initial amount of the active form.

        beta:
            The inactivation rate.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import CascadeStage
        >>> CascadeStage('camp', beta=2.0).beta
        2.0
    """

    x: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    beta: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the inactivation `-beta x`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.x, -s.beta * s.x)]


class Activation(Interaction):
    """Activation of a cascade stage by the active form of the stage above it, at the rate `alpha` times its amount.

    Args:
        label:
            The label of the interaction.

        upstream:
            The path of the activating stage.

        downstream:
            The path of the activated stage.

        state_name:
            The state of the activating stage, `'x'` for a stage and `'g'` for the G protein.

        alpha:
            The activation rate per unit of the activating stage.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import Activation
        >>> Activation('activation', upstream='camp', downstream='protein_kinase_a', alpha=5.0).alpha
        5.0
    """

    upstream: str = role()
    downstream: str = role()
    state_name: str = 'x'
    alpha: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the activation `alpha` times the active upstream amount.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.downstream.x, s.alpha * getattr(s.upstream, self.state_name))]


@register_model(
    field='biology/systems_biology',
    tags=('signalling', 'cascade', 'amplification', 'hormone', 'g protein', 'ode'),
    references=('Konieczny et al. 2023', 'Heinrich, Neel, and Rapoport 2002'),
    defaults={'until': 30.0, 'dt': 0.2},
)
def hormone_cascade_amplifier(
    alpha: Sequence[float] = (10.0, 100.0, 1.0, 10.0, 10.0),
    beta: Sequence[float] = (1.0, 2.0, 0.5, 1.0, 1.0),
    *,
    k_on: float = 1.0,
    k_gtp: float = 0.5,
    hormone: float = 1.0,
    start: float = 1.0,
    width: float = 1.0,
) -> System:
    """Return the adenylate cyclase cascade, which amplifies a hormone pulse and ends it by the G-protein timer.

    A hormone pulse activates the G protein, which activates adenylate cyclase, which makes cyclic AMP, which activates
    protein kinase A, phosphorylase kinase, and glycogen phosphorylase in turn (Konieczny et al. 4.4.3). Every stage is
    activated at the rate `alpha_i` times the active amount of the stage above it and inactivated at the rate
    `beta_i`, so the integral of every stage over the signal is the integral of the stage above it times the gain
    `alpha_i / beta_i`, and the cascade multiplies the integral of the G-protein signal by the product of the gains,
    100,000 with the defaults. The G protein inactivates itself by GTP hydrolysis at the rate `k_gtp`, the timer of
    Konieczny et al. 4.4.6, so blocking it, as cholera toxin does, prolongs the signal.

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

    Returns:
        system:
            The model, with the active G protein `g_protein.g` and the active stages `adenylate_cyclase.x`, `camp.x`,
            `protein_kinase_a.x`, `phosphorylase_kinase.x`, and `phosphorylase.x`.

    Raises:
        ValidationError:
            The activation and inactivation rates do not give one value per stage.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import hormone_cascade_amplifier
        >>> [element.label for element in hormone_cascade_amplifier().elements][-1]
        'phosphorylase'
    """
    if len(alpha) != len(STAGES) or len(beta) != len(STAGES):
        error_msg = f'Give one activation and one inactivation rate per stage, {len(STAGES)} of each.'
        raise ValidationError(error_msg)
    g_protein = GProtein('g_protein', k_on=k_on, k_gtp=k_gtp, hormone=hormone, start=start, width=width)
    stages = [CascadeStage(label, beta=rate) for label, rate in zip(STAGES, beta, strict=True)]
    upstreams = ('g_protein', *STAGES[:-1])
    activations = [
        Activation(
            f'activation_of_{label}',
            upstream=upstream,
            downstream=label,
            state_name='g' if upstream == 'g_protein' else 'x',
            alpha=rate,
        )
        for upstream, label, rate in zip(upstreams, STAGES, alpha, strict=True)
    ]
    return System(components=[g_protein, *stages], interactions=activations)
