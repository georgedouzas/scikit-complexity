"""Neuroscience systems: exact neural masses of quadratic integrate-and-fire populations under theta forcing."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.components.system import System
from ....core.errors import ValidationError
from ..._catalog import register_system
from ._neural_mass import NeuralMassElement
from ._synaptic_drive import SynapticDriveInteraction
from ._theta_forcing import ThetaForcingInteraction

SETUPS = ('ping', 'ing')


@register_system(
    field='biology/neuroscience',
    tags=('neural mass', 'next generation', 'gamma oscillations', 'theta-gamma', 'ode'),
    references=('Montbrió, Pazó, and Roxin 2015', 'Segneri et al. 2020'),
    defaults={'until': 1000.0, 'dt': 0.5},
)
def next_generation_neural_mass(
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
) -> System:
    """Return the exact neural mass of quadratic integrate-and-fire neurons whose gamma rhythm nests in a theta drive.

    In the `'ping'` setup an excitatory and an inhibitory population, with membrane time constants of 20 and 10 ms,
    couple through instantaneous synapses, and the theta current drives the excitatory one. In the `'ing'` setup a
    single inhibitory population inhibits itself through exponential synapses of 10 ms, and the current drives it.
    Without forcing the populations rest or oscillate collectively in the gamma band, about 45 Hz for `H_e = 10`, as
    their excitabilities cross a Hopf bifurcation. A slow theta current sweeps them across it, so gamma bursts appear
    at every theta peak, theta-nested gamma oscillations. The defaults are those of Segneri et al. (2020, figure 3).

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

    Returns:
        system:
            The system, with the rates `excitatory.r` and `inhibitory.r`, in spikes per millisecond, and the
            potentials `excitatory.v` and `inhibitory.v`, or the inhibitory population alone in `'ing'`.

    Raises:
        ValidationError:
            The setup is unknown.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import next_generation_neural_mass
        >>> [element.label for element in next_generation_neural_mass('ing').elements]
        ['inhibitory']
    """
    if setup not in SETUPS:
        error_msg = f'The setup must be one of {", ".join(SETUPS)}, not {setup!r}.'
        raise ValidationError(error_msg)
    if setup == 'ing':
        population = NeuralMassElement(
            'inhibitory',
            tau=10.0,
            delta=0.3 if delta is None else delta,
            H=2.0 if H_i is None else H_i,
        )
        return System(
            elements=[population],
            interactions=[
                SynapticDriveInteraction(
                    'ii',
                    target='inhibitory',
                    source='inhibitory',
                    J=-(21.0 if J_ii is None else J_ii),
                    field='s',
                ),
                ThetaForcingInteraction('theta', target='inhibitory', I0=9.0 if I0 is None else I0, nu=nu),
            ],
        )
    width = 1.0 if delta is None else delta
    return System(
        elements=[
            NeuralMassElement('excitatory', tau=20.0, delta=width, H=H_e),
            NeuralMassElement('inhibitory', tau=10.0, delta=width, H=-5.0 if H_i is None else H_i),
        ],
        interactions=[
            SynapticDriveInteraction('ee', target='excitatory', source='excitatory', J=J_ee),
            SynapticDriveInteraction('ie', target='excitatory', source='inhibitory', J=-J_ie),
            SynapticDriveInteraction('ei', target='inhibitory', source='excitatory', J=J_ei),
            SynapticDriveInteraction(
                'ii',
                target='inhibitory',
                source='inhibitory',
                J=-(0.0 if J_ii is None else J_ii),
            ),
            ThetaForcingInteraction('theta', target='excitatory', I0=10.0 if I0 is None else I0, nu=nu),
        ],
    )
