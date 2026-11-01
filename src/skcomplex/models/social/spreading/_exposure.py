"""Exposure contagion: SIS in the environments of a hypergraph, with infective doses from the infected members."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import xgi
from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.elements import LatticeElement
from ....core.interactions import CouplingInteraction
from ....core.laws.utils import count, total, uniform, where
from ....core.spaces import NetworkSpace
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, RoleSpec, VisualSpec
from ..._catalog import register_model
from ._participant import ParticipantElement


class ExposureInteraction(CouplingInteraction):
    """Infection of susceptible participants by the doses of their environments, and recovery, in discrete time.

    It is a `CouplingInteraction` over the hyperedges of a network in the map form. An environment, a hyperedge, gives
    each susceptible member the dose `rate * rho ** exponent`, where `rho` is the share of the other members that are
    infected, and the doses of the environments of a participant add up to `D`. In every step a susceptible
    participant is infected with probability `1 - exp(-D)`, and an infected one recovers with probability `recovery`,
    a parameter of the participants.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the participants on the network.

        rate:
            The dose of an environment whose other members are all infected.

        exponent:
            The exponent of the infection kernel, sublinear below 1 and supralinear above.

    Examples:
        >>> from skcomplex.models.social.spreading import ExposureInteraction
        >>> ExposureInteraction('exposure', target='participants', exponent=1.5).check().exponent
        1.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.06,
        exponent: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            mode='hyperedges',
            form='map',
            write='infected',
            function=self._get_dose,
            combine=self._get_next,
        )
        self.rate = rate
        self.exponent = exponent

    def _get_dose(self, environment: SimpleNamespace) -> Expression:
        """Return the dose an environment gives its member, from the share of its other members that are infected."""
        others = total(environment.members.infected, over=environment.members) - environment.member.infected
        share = others / (count(over=environment.members) - 1)
        return environment.rate * share**environment.exponent

    @staticmethod
    def _get_next(participant: SimpleNamespace, dose: Expression) -> Expression:
        """Return the next state of a participant, infected by its dose or recovered with the recovery probability."""
        member, draw = participant.member, uniform('exposure')
        infection = where(draw < 1 - exp(-dose), 1, 0)
        return where(member.infected == 1, where(draw < member.recovery, 0, 1), infection)


@register_model(
    field='social/spreading',
    tags=('contagion', 'hypergraph', 'exposure', 'bistability'),
    references=('St-Onge et al. 2021', 'Majhi et al. 2022'),
    defaults={'until': 200, 'dt': 1},
    observables=lambda system: [ObservableSpec('prevalence', group='participants', variable='infected')],
)
def hypergraph_exposure_sis(
    n_nodes: int = 1000,
    *,
    sizes: Sequence[int] = (10, 20, 30),
    hyperdegree: float = 5.0,
    rate: float = 0.06,
    exponent: float = 2.0,
    recovery: float = 0.05,
    share: float = 0.3,
    seed: int = 0,
) -> System:
    """Return the SIS contagion of St-Onge et al. in the environments of a hypergraph, with a nonlinear kernel.

    Participants meet in environments, the hyperedges of a random hypergraph. In every step an environment gives a
    susceptible member the dose `rate * rho ** exponent` from the share `rho` of its other members that are infected,
    the member is infected with probability `1 - exp(-D)` of the sum `D` of its doses, and an infected participant
    recovers with probability `recovery`. Heterogeneous exposure times give the power law `rho ** exponent` to the
    infection kernel. With a sublinear or linear kernel the epidemic sets in continuously, and with a supralinear
    kernel, `exponent > 1`, the healthy state stays stable: an endemic state coexists with it above a critical rate,
    so a small seed dies out and a large one persists. In mean field the prevalence `I` is endemic where `k rate
    I ** (exponent - 1) (1 - I) = recovery` for `k` environments per participant.

    Args:
        n_nodes:
            The number of participants.

        sizes:
            The sizes of the environments.

        hyperdegree:
            The mean number of environments of a participant, split evenly over the sizes, whose members are drawn at
            random.

        rate:
            The dose of an environment whose other members are all infected.

        exponent:
            The exponent of the infection kernel.

        recovery:
            The probability of recovery in a step.

        share:
            The share of infected participants at the start.

        seed:
            The seed of the hypergraph and the infected participants.

    Returns:
        system:
            The model, with the states `participants.infected` and the prevalence `prevalence`.

    Examples:
        >>> from skcomplex.models.social.spreading import hypergraph_exposure_sis
        >>> hypergraph_exposure_sis(40).elements[0].check().prepare().size_
        40
    """
    rng = np.random.default_rng(seed)
    hypergraph = xgi.Hypergraph()
    hypergraph.add_nodes_from(range(n_nodes))
    for size in sizes:
        number = round(n_nodes * hyperdegree / (len(sizes) * size))
        hypergraph.add_edges_from(rng.choice(n_nodes, size, replace=False).tolist() for _ in range(number))
    infected = (rng.permutation(n_nodes) < round(share * n_nodes)).astype(float)
    participants = LatticeElement(
        'participants',
        element=ParticipantElement,
        space=NetworkSpace(xgi.to_graph(hypergraph), hyperedges=hypergraph),
        values={'infected': infected, 'recovery': recovery},
        visual=VisualSpec('sites', {'states': ['infected'], 'low': 0.0, 'high': 1.0}),
    )
    exposure = ExposureInteraction('exposure', target='participants', rate=rate, exponent=exponent)
    return System(elements=[participants], interactions=[exposure])
