"""Crowd systems: the kinetic theory of criminality in a society.

It includes the security forces that act against it.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import DensityElement, Element, GroupElement
from ....core.components.interactions import EncounterInteraction, RateInteraction
from ....core.components.laws.utils import moment
from ....core.components.spaces import ContinuousTime, LatticeSpace
from ....core.components.systems import BaseSystem, RunSettings, System
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec


class SecurityAgentsElement(Element):
    """Density of security agents at one efficacy class.

    It holds the share of new recruits that join the class.

    Args:
        label:
            The label of the distribution.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        f:
            The initial density.

        share:
            The share of new recruits that join the efficacy class.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.crowds import SecurityAgentsElement
        >>> GroupElement('security', element=SecurityAgentsElement, size=3, values={'share': [0.5, 0.3, 0.2]}).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        f: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        share: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.f = f
        self.share = share


def contagion_table(classes: int, attraction: object, redemption: object) -> npt.NDArray[np.object_]:
    """Return the transition table of encounters between citizens, `A[h, k, i]`.

    A citizen of class `h` who meets a more criminal citizen moves one class up with probability `attraction`, and one
    who meets a more honest citizen moves one class down with probability `redemption`.

    Args:
        classes:
            The number of criminality classes.

        attraction:
            The probability that crime attracts, a number or a parameter.

        redemption:
            The probability that honesty redeems, a number or a parameter.

    Returns:
        table:
            The transition probabilities, which sum to one over the outcomes.

    Examples:
        >>> from skcomplex.systems.social.crowds import contagion_table
        >>> contagion_table(3, 0.3, 0.1)[0, 2].astype(float).tolist()
        [0.7, 0.3, 0.0]
    """
    table = np.zeros((classes, classes, classes), dtype=object)
    for candidate in range(classes):
        for field in range(classes):
            if field > candidate:
                table[candidate, field, candidate + 1] += attraction
            elif field < candidate:
                table[candidate, field, candidate - 1] += redemption
            table[candidate, field, candidate] += 1 - table[candidate, field].sum()
    return table


class CriminalContagionInteraction(EncounterInteraction):
    """Conservative encounters between citizens.

    In them crime attracts and honesty redeems.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the distribution of citizens whose members move.

        field:
            The path of the distribution of citizens they meet, usually the same distribution.

        classes:
            The number of criminality classes.

        eta:
            The encounter rate.

        attraction:
            The probability that a citizen who meets a more criminal one moves one class up.

        redemption:
            The probability that a citizen who meets a more honest one moves one class down.

    Examples:
        >>> from skcomplex.systems.social.crowds import CriminalContagionInteraction
        >>> CriminalContagionInteraction('contagion', candidate='citizens', field='citizens').attraction
        0.3
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec()],
        field: Annotated[str, RoleSpec()],
        classes: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 5,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        attraction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        redemption: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.classes = classes
        self.eta = eta
        self.attraction = attraction
        self.redemption = redemption

    def _get_rate(self, s: SimpleNamespace) -> object:
        """Return the encounter rate."""
        return s.eta

    def _get_kernel(self, s: SimpleNamespace) -> object:
        """Return the table of the attraction of crime and the redemption of honesty."""
        return contagion_table(int(self.classes), s.attraction, s.redemption)


def deterrence_table(classes: int, forces: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the rate pattern and the transition table of deterrence encounters.

    They are the encounters of citizens with security forces. A citizen of a criminal class `h > 0` who meets a security
    agent of efficacy `v_k` moves one class down at a rate proportional to `v_k`. Honest citizens are left alone.

    Args:
        classes:
            The number of criminality classes.

        forces:
            The number of efficacy classes of the security forces.

    Returns:
        pattern:
            The rate pattern `eta[h, k]`, the efficacy `v_k` for criminal classes and 0 for honest ones.

        table:
            The transition table `A[h, k, i]`.

    Examples:
        >>> from skcomplex.systems.social.crowds import deterrence_table
        >>> deterrence_table(2, 2)[0].tolist()
        [[0.0, 0.0], [0.5, 1.0]]
    """
    efficacy = np.linspace(1.0 / forces, 1.0, forces)
    pattern = np.zeros((classes, forces))
    pattern[1:] = efficacy
    table = np.zeros((classes, forces, classes))
    for candidate in range(classes):
        table[candidate, :, max(candidate - 1, 0)] = 1.0
    return pattern, table


class DeterrenceInteraction(EncounterInteraction):
    """Encounters of citizens with security forces.

    They move criminal citizens toward honesty.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the distribution of citizens.

        field:
            The path of the distribution of the security forces over efficacy.

        classes:
            The number of criminality classes.

        forces:
            The number of efficacy classes of the security forces.

        deterrence:
            The rate at which the most effective agents deter a criminal citizen.

    Examples:
        >>> from skcomplex.systems.social.crowds import DeterrenceInteraction
        >>> DeterrenceInteraction('deterrence', candidate='citizens', field='security').deterrence
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec()],
        field: Annotated[str, RoleSpec()],
        classes: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 5,
        forces: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 3,
        deterrence: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.classes = classes
        self.forces = forces
        self.deterrence = deterrence

    def _get_rate(self, s: SimpleNamespace) -> object:
        """Return the deterrence rate of every pair of classes.

        A pair holds a criminality class and an efficacy class.
        """
        return deterrence_table(int(self.classes), int(self.forces))[0] * s.deterrence

    def _get_kernel(self, s: SimpleNamespace) -> object:
        """Return the table that moves a deterred citizen one class down."""
        return deterrence_table(int(self.classes), int(self.forces))[1]


def criminality_grid(classes: int) -> npt.NDArray[np.float64]:
    """Return the criminality of every class of citizens.

    It runs from 0 for honest to 1 for criminal.

    Args:
        classes:
            The number of criminality classes.

    Returns:
        criminality:
            The criminality of every class.

    Examples:
        >>> from skcomplex.systems.social.crowds import criminality_grid
        >>> criminality_grid(3).tolist()
        [0.0, 0.5, 1.0]
    """
    return np.linspace(0.0, 1.0, classes)


class RecruitmentInteraction(RateInteraction):
    """Recruitment of security agents in proportion to the mean criminality.

    It is a macro-scale external action. It is a `RateInteraction` of the density `f` of the security forces that reads
    the citizens. The security forces grow at the rate `recruitment m share_k` in efficacy class `k`, where `m` is the
    mean criminality of the citizens, a moment of their distribution, and retire at the rate `retirement`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the distribution of the security forces over efficacy.

        citizens:
            The path of the distribution of citizens.

        classes:
            The number of criminality classes.

        recruitment:
            The recruitment rate per unit of mean criminality.

        retirement:
            The retirement rate.

    Examples:
        >>> from skcomplex.systems.social.crowds import RecruitmentInteraction
        >>> RecruitmentInteraction('recruitment', target='security', citizens='citizens').retirement
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        citizens: Annotated[str, RoleSpec()],
        classes: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 5,
        recruitment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        retirement: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, target=target, state='f', rate=self._get_rate)
        self.citizens = citizens
        self.classes = classes
        self.recruitment = recruitment
        self.retirement = retirement

    def _get_rate(self, s: SimpleNamespace) -> Expression:
        """Return the recruitment of security agents, less the retirement.

        The recruitment is in proportion to the mean criminality. The criminality is that of the citizens.
        """
        security, citizens = s.target, s.citizens
        criminality = moment(citizens.f, 1, criminality_grid(int(self.classes)), normalized=True)
        return s.recruitment * criminality * security.share - s.retirement * security.f


CRIMINAL_THRESHOLD = 0.5


CRIMINALITY_REFERENCES = ('Bellomo et al. 2017', 'Bellomo, Colasuonno, Knopoff, and Soler 2015')


def mean_criminality(s: SimpleNamespace, criminality: npt.NDArray[np.float64]) -> Expression:
    """Return the mean criminality of a distribution of citizens.

    Args:
        s:
            The namespace of one member of the distribution.

        criminality:
            The criminality of every class.

    Returns:
        mean:
            The normalized first moment.
    """
    return moment(s.f, 1, criminality, normalized=True)


def criminal_fraction(s: SimpleNamespace, criminality: npt.NDArray[np.float64]) -> Expression:
    """Return the fraction of citizens whose criminality is at least one half.

    Args:
        s:
            The namespace of one member of the distribution.

        criminality:
            The criminality of every class.

    Returns:
        fraction:
            The moment of the indicator of the criminal classes, divided by the density.
    """
    return moment(s.f, 1, (criminality >= CRIMINAL_THRESHOLD).astype(float), normalized=True)


def security_density(s: SimpleNamespace, forces: int) -> Expression:
    """Return the density of the security forces.

    Args:
        s:
            The namespace of one member of the distribution of the security forces.

        forces:
            The number of efficacy classes.

    Returns:
        density:
            The order-0 moment.
    """
    return moment(s.f, 0, np.arange(forces, dtype=float))


def _sizes_of(system: System) -> tuple[int, int]:
    """Return the number of criminality classes and of efficacy classes of a system."""
    sizes = {item.label: int(item.size or 0) for item in system.elements}
    return sizes['citizens'], sizes['security']


def criminality_observables(classes: int, forces: int) -> list[ObservableSpec]:
    """Return the mean criminality, the criminal fraction, and the security density.

    The density is that of the security forces of the system.

    Args:
        classes:
            The number of criminality classes.

        forces:
            The number of efficacy classes of the security forces.

    Returns:
        observables:
            The observables, each reading the moments of a distribution.

    Examples:
        >>> from skcomplex.systems.social.crowds import criminality_observables
        >>> [item.label for item in criminality_observables(5, 3)]
        ['mean_criminality', 'criminal_fraction', 'security_forces']
    """
    criminality = criminality_grid(classes)
    return [
        ObservableSpec('mean_criminality', partial(mean_criminality, criminality=criminality), group='citizens'),
        ObservableSpec('criminal_fraction', partial(criminal_fraction, criminality=criminality), group='citizens'),
        ObservableSpec('security_forces', partial(security_density, forces=forces), group='security'),
    ]


class CriminalitySystem(BaseSystem):
    """The kinetic system of criminality in a society.

    It includes the security forces that act against it. Citizens are a distribution over criminality classes and
    security agents a distribution over efficacy classes, held by site groups on lattices of the classes. Encounters
    between citizens spread crime when it attracts more than honesty redeems, encounters with security agents move
    criminal citizens toward honesty, a micro-scale action, and the security forces grow in proportion to the mean
    criminality, a macro-scale action that reads a moment of the citizens. The security forces grow while crime is high,
    crime recedes, and the forces retire, so crime and security oscillate toward a balance where some crime persists.

    The system has the distributions `citizens` and `security` and the interactions `contagion`, `deterrence`, and
    `recruitment`. Its standard observables are the mean criminality, the criminal fraction, and the density of the
    security forces.

    Args:
        classes:
            The number of criminality classes.

        forces:
            The number of efficacy classes of the security forces.

        criminals:
            The initial fraction of citizens in the most criminal class, the others being honest.

        security:
            The initial density of the security forces, spread evenly over their efficacy classes.

        attraction:
            The probability that a citizen who meets a more criminal one moves one class up.

        redemption:
            The probability that a citizen who meets a more honest one moves one class down.

        deterrence:
            The rate at which the most effective agents deter a criminal citizen.

        recruitment:
            The recruitment rate of security agents per unit of mean criminality.

        retirement:
            The retirement rate of security agents.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.crowds import CriminalitySystem
        >>> [item.label for item in CriminalitySystem().prepare().system_.interactions]
        ['contagion', 'deterrence', 'recruitment']
    """

    field = 'social/crowds'
    topics = ('criminality', 'security', 'kinetic', 'distribution')
    references = CRIMINALITY_REFERENCES
    until = 60
    dt = 0.5

    def __init__(
        self,
        classes: int = 5,
        forces: int = 3,
        *,
        criminals: float = 0.2,
        security: float = 0.05,
        attraction: float = 0.3,
        redemption: float = 0.1,
        deterrence: float = 2.0,
        recruitment: float = 0.5,
        retirement: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.classes = classes
        self.forces = forces
        self.criminals = criminals
        self.security = security
        self.attraction = attraction
        self.redemption = redemption
        self.deterrence = deterrence
        self.recruitment = recruitment
        self.retirement = retirement

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        citizens = np.zeros(self.classes)
        citizens[0], citizens[-1] = 1.0 - self.criminals, self.criminals
        return System(
            self.label,
            elements=[
                GroupElement('citizens', element=DensityElement, size=self.classes, on='sites', values={'f': citizens}),
                GroupElement(
                    'security',
                    element=SecurityAgentsElement,
                    size=self.forces,
                    on='sites',
                    space=LatticeSpace(self.forces, boundary='fixed'),
                    values={
                        'f': np.full(self.forces, self.security / self.forces),
                        'share': np.full(self.forces, 1.0 / self.forces),
                    },
                ),
            ],
            interactions=[
                CriminalContagionInteraction(
                    'contagion',
                    candidate='citizens',
                    field='citizens',
                    classes=self.classes,
                    attraction=self.attraction,
                    redemption=self.redemption,
                ),
                DeterrenceInteraction(
                    'deterrence',
                    candidate='citizens',
                    field='security',
                    classes=self.classes,
                    forces=self.forces,
                    deterrence=self.deterrence,
                ),
                RecruitmentInteraction(
                    'recruitment',
                    target='security',
                    citizens='citizens',
                    classes=self.classes,
                    recruitment=self.recruitment,
                    retirement=self.retirement,
                ),
            ],
            space=LatticeSpace(self.classes, boundary='fixed'),
            time=ContinuousTime(),
        )

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        The observables of the parts are built from the model parameters.
        """
        settings = super().get_run_settings()
        system = self._pass_parameters()
        settings['observables'] = criminality_observables(*_sizes_of(system))
        return settings
