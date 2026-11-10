"""Neuroscience systems: exact neural masses under theta forcing.

The neural masses are those of quadratic integrate-and-fire populations.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

FIELDS = ('r', 's')
SETUPS = ('ping', 'ing')


class NeuralMassElement(Element):
    """Population of quadratic integrate-and-fire neurons in an exact mean field.

    The mean field is the one of Montbrió, Pazó, and Roxin.

    The excitabilities of the neurons follow a Lorentzian distribution of median `H` and half-width `delta`, so the
    firing rate `r` and the mean membrane potential `v` of the population follow `dr/dt = delta / (tau^2 pi) + 2 r v
    / tau` and `dv/dt = (v^2 + H) / tau - tau (pi r)^2` exactly, with the membrane time constant `tau`. The synaptic
    field `s` filters the rate with exponential synapses, `ds/dt = (r - s) / tau_d`. Couplings and inputs add to the
    rate of `v`.

    Args:
        label:
            The label of the population.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        r:
            The initial firing rate, in spikes per millisecond.

        v:
            The initial mean membrane potential.

        s:
            The initial synaptic field.

        tau:
            The membrane time constant, in milliseconds.

        delta:
            The half-width of the Lorentzian excitabilities.

        H:
            The median of the Lorentzian excitabilities.

        tau_d:
            The decay time of the synapses, in milliseconds.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import NeuralMassElement
        >>> NeuralMassElement('excitatory', tau=20.0).tau
        20.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        r: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        v: Annotated[float, StateSpec()] = -1.0,
        s: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        delta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        H: Annotated[float, ParameterSpec()] = -5.0,
        tau_d: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.r = r
        self.v = v
        self.s = s
        self.tau = tau
        self.delta = delta
        self.H = H
        self.tau_d = tau_d

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the mean-field rate, potential, and synaptic field.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.r, s.delta / (s.tau**2 * pi) + 2 * s.r * s.v / s.tau),
            RateLaw(s.v, (s.v**2 + s.H) / s.tau - s.tau * (pi * s.r) ** 2),
            RateLaw(s.s, (s.r - s.s) / s.tau_d),
        ]


class SynapticDriveInteraction(Interaction):
    """Synaptic drive of a population by another, `dv/dt += J x`.

    The drive is by the rate or the synaptic field of the other population.

    Instantaneous synapses pass the firing rate `r` of the source, and exponential synapses its synaptic field `s`. A
    negative strength `J` is inhibition.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the driven population.

        source:
            The path of the driving population, which may be the target.

        J:
            The strength of the synapses, negative for inhibition.

        field:
            `'r'` for instantaneous synapses or `'s'` for exponential synapses.

    Raises:
        ValidationError:
            The field is unknown.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import SynapticDriveInteraction
        >>> SynapticDriveInteraction('ie', target='e', source='i', J=-10.0).J
        -10.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        J: Annotated[float, ParameterSpec()] = 1.0,
        field: str = 'r',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.J = J
        self.field = field

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Check the field.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The field is unknown.
        """
        super()._prepare(context)
        if self.field not in FIELDS:
            error_msg = f'The field must be one of {", ".join(FIELDS)}, not {self.field!r}.'
            raise ValidationError(error_msg)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the drive of the potential of the target.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.v, s.J * getattr(s.source, self.field))]


class ThetaForcingInteraction(Interaction):
    """Theta forcing of a population by the current `I0 (1 - cos(2 pi nu t)) / 2`.

    The forcing is as in optogenetic stimulation.

    The current enters the potential of the population as its excitabilities do, `dv/dt += I(t) / tau`. Time is in
    milliseconds and the frequency `nu` in hertz.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the forced population.

        I0:
            The peak current.

        nu:
            The frequency of the forcing, in hertz.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import ThetaForcingInteraction
        >>> ThetaForcingInteraction('theta', target='e', nu=8.0).nu
        8.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        I0: Annotated[float, ParameterSpec()] = 10.0,
        nu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.I0 = I0
        self.nu = nu

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the forcing current of the potential.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        current = s.I0 / 2 * (1 - cos(2 * pi * s.nu * s.t / 1000))
        return [RateLaw(s.target.v, current / s.target.tau)]


class NextGenerationNeuralMassSystem(BaseSystem):
    """The exact neural mass of quadratic integrate-and-fire neurons.

    Its gamma rhythm nests in a theta drive.

    In the `'ping'` setup an excitatory and an inhibitory population, with membrane time constants of 20 and 10 ms,
    couple through instantaneous synapses, and the theta current drives the excitatory one. In the `'ing'` setup a
    single inhibitory population inhibits itself through exponential synapses of 10 ms, and the current drives it.
    Without forcing the populations rest or oscillate collectively in the gamma band, about 45 Hz for `H_e = 10`, as
    their excitabilities cross a Hopf bifurcation. A slow theta current sweeps them across it, so gamma bursts appear at
    every theta peak, theta-nested gamma oscillations. The defaults are those of Segneri et al. (2020, figure 3). The
    system has the rates `excitatory.r` and `inhibitory.r`, in spikes per millisecond, and the potentials `excitatory.v`
    and `inhibitory.v`, or the inhibitory population alone in `'ing'`.

    Args:
        setup:
            `'ping'` for excitatory and inhibitory populations or `'ing'` for one inhibitory population.

        H_e:
            The median excitability of the excitatory population.

        H_i:
            The median excitability of the inhibitory population, or `None` for -5 in `'ping'` and 2 in `'ing'`.

        J_ee:
            The excitatory self-coupling of `'ping'`.

        J_ie:
            The inhibition of the excitatory population by the inhibitory one in `'ping'`.

        J_ei:
            The excitation of the inhibitory population by the excitatory one in `'ping'`.

        J_ii:
            The inhibitory self-coupling, or `None` for 0 in `'ping'` and 21 in `'ing'`.

        delta:
            The half-width of the excitabilities, or `None` for 1 in `'ping'` and 0.3 in `'ing'`.

        I0:
            The peak theta current, or `None` for 10 in `'ping'` and 9 in `'ing'`.

        nu:
            The theta frequency, in hertz.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The setup is unknown.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import NextGenerationNeuralMassSystem
        >>> [element.label for element in NextGenerationNeuralMassSystem('ing').prepare().system_.elements]
        ['inhibitory']
    """

    field = 'biology/neuroscience'
    topics = ('neural mass', 'next generation', 'gamma oscillations', 'theta-gamma', 'ode')
    references = ('Montbrió, Pazó, and Roxin 2015', 'Segneri et al. 2020')
    until = 1000.0
    dt = 0.5

    def __init__(
        self,
        setup: str = 'ping',
        *,
        H_e: float = 1.3,
        H_i: float | None = None,
        J_ee: float = 8.0,
        J_ie: float = 10.0,
        J_ei: float = 10.0,
        J_ii: float | None = None,
        delta: float | None = None,
        I0: float | None = None,
        nu: float = 5.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.setup = setup
        self.H_e = H_e
        self.H_i = H_i
        self.J_ee = J_ee
        self.J_ie = J_ie
        self.J_ei = J_ei
        self.J_ii = J_ii
        self.delta = delta
        self.I0 = I0
        self.nu = nu

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the exact quadratic integrate-and-fire neural mass."""
        if self.setup not in SETUPS:
            error_msg = f'The setup must be one of {", ".join(SETUPS)}, not {self.setup!r}.'
            raise ValidationError(error_msg)
        if self.setup == 'ing':
            population = NeuralMassElement(
                'inhibitory',
                tau=10.0,
                delta=0.3 if self.delta is None else self.delta,
                H=2.0 if self.H_i is None else self.H_i,
            )
            return CompositeSystem(
                self.label,
                elements=[population],
                interactions=[
                    SynapticDriveInteraction(
                        'ii',
                        target='inhibitory',
                        source='inhibitory',
                        J=-(21.0 if self.J_ii is None else self.J_ii),
                        field='s',
                    ),
                    ThetaForcingInteraction(
                        'theta',
                        target='inhibitory',
                        I0=9.0 if self.I0 is None else self.I0,
                        nu=self.nu,
                    ),
                ],
            )
        width = 1.0 if self.delta is None else self.delta
        return CompositeSystem(
            self.label,
            elements=[
                NeuralMassElement('excitatory', tau=20.0, delta=width, H=self.H_e),
                NeuralMassElement('inhibitory', tau=10.0, delta=width, H=-5.0 if self.H_i is None else self.H_i),
            ],
            interactions=[
                SynapticDriveInteraction('ee', target='excitatory', source='excitatory', J=self.J_ee),
                SynapticDriveInteraction('ie', target='excitatory', source='inhibitory', J=-self.J_ie),
                SynapticDriveInteraction('ei', target='inhibitory', source='excitatory', J=self.J_ei),
                SynapticDriveInteraction(
                    'ii',
                    target='inhibitory',
                    source='inhibitory',
                    J=-(0.0 if self.J_ii is None else self.J_ii),
                ),
                ThetaForcingInteraction(
                    'theta',
                    target='excitatory',
                    I0=10.0 if self.I0 is None else self.I0,
                    nu=self.nu,
                ),
            ],
        )
