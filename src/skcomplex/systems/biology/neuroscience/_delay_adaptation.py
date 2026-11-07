"""Phase coupling through conduction delays that adapt to the phase differences of the units they join."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sin, tanh
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import clip, delayed
from ....core.specs import ParameterSpec, RoleSpec


class DelayAdaptationInteraction(Interaction):
    """Phase coupling through conduction delays that adapt to the phase differences of the units they join.

    The target unit reads the phase of the source unit along the axon as it was one conduction delay ago,
    `dtheta_i/dt += kappa sin(theta_j(t - tau_ij) - theta_i)`. The delay relaxes to its baseline and is pushed by the
    phase difference, `dtau_ij/dt = epsilon H(tau_ij) (-(tau_ij - tau0_ij) + rho sin(theta_j - theta_i))`, where the
    smooth step `H(tau) = (1 + tanh(tau / width)) / 2` keeps the delay nonnegative. The coupling reads the delay no
    shorter than `shortest`, which bounds the steps of the solver.

    Args:
        label:
            The label of the interaction.

        source:
            The path of the unit that sends the signal.

        target:
            The path of the unit that receives the signal.

        axon:
            The path of the axon that conducts it.

        kappa:
            The coupling strength of an edge.

        epsilon:
            The adaptation rate of the delays.

        rho:
            The strength of the push of the phase difference on the delay.

        width:
            The width of the smooth step.

        shortest:
            The shortest delay the coupling reads.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import DelayAdaptationInteraction
        >>> DelayAdaptationInteraction('myelination', source='a', target='b', axon='ab', rho=0.5).rho
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        target: Annotated[str, RoleSpec()],
        axon: Annotated[str, RoleSpec()],
        kappa: Annotated[float, ParameterSpec()] = 0.1,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        rho: Annotated[float, ParameterSpec()] = 1.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
        shortest: float = 0.1,
    ) -> None:
        super().__init__(label)
        self.source = source
        self.target = target
        self.axon = axon
        self.kappa = kappa
        self.epsilon = epsilon
        self.rho = rho
        self.width = width
        self.shortest = shortest

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the delayed coupling of the target and the adaptation of the delay.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        source, target, tau = s.source.theta, s.target.theta, s.axon.tau
        step = (1 + tanh(tau / s.width)) / 2
        return [
            RateLaw(target, s.kappa * sin(delayed(source, clip(tau, low=self.shortest), self.shortest) - target)),
            RateLaw(tau, s.epsilon * step * (-(tau - s.axon.tau0) + s.rho * sin(source - target))),
        ]
