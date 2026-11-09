"""
# Elman network

Elman's simple recurrent network reads a stream of bits one at a time and learns to predict the next bit. In the
temporal XOR stream every third bit is the exclusive or of the two bits before it, and the other bits are random, so
only every third bit can be predicted. Training by backpropagation through time, one epoch per frame, drives the error
on those bits toward 0, while the error on the random bits stays at 0.25, the error of the best guess of 0.5.
"""

# %%
# The network has 4 hidden units and trains with Adam on windows of 12 bits. The primary view shows the training loss
# with the two metrics, `xor_error` on the predictable bits and `random_error` on the others, and the heat maps of
# the input, recurrent, and readout weights next to it.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.specs import ObservableSpec
from skcomplex.systems.computer_science.neural_networks import ElmanNetworkSystem

SYSTEM = 'elman_network'
EPOCHS = 200

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
system = ElmanNetworkSystem()
trajectory = simulate(system, until=EPOCHS, observers=[observer], random_state=0)

# %%
# Error at each position of the bit triple
# ----------------------------------------
# The trained system is the nested network with the weights of a frame set through `set_params`. Run over a fresh
# stream with its inputs given by a schedule, the network before training should miss every bit by about 0.25, and
# after training it should predict the third bit of every triple almost exactly while the first two stay at 0.25.


def squared_error(member):
    """Return the squared error of the prediction of the output unit."""
    return (member.y - member.target) ** 2


def errors_by_position(frame, bits):
    """Return the mean squared error of the predictions at each position of the triple, with the weights of a frame."""
    net = ElmanNetworkSystem().prepare().system_.systems[0]
    for interaction, name in zip(net.interactions, ('D', 'A', 'W'), strict=True):
        interaction.set_params(**{name: trajectory.get_values(f'net/{interaction.label}.{name}')[frame]})
    net.elements[1].set_params(values={'b': trajectory.get_values('net/hidden.b')[frame]})
    net.elements[2].set_params(values={'c': trajectory.get_values('net/outputs.c')[frame]})
    schedule = {'inputs.x': lambda t: bits[round(t)], 'outputs.target': lambda t: bits[round(t)]}
    error = ObservableSpec('error', group='outputs', reduce='sum', expression=squared_error)
    run = simulate(net, until=len(bits) - 1, backend=NumpyBackend(), schedule=schedule, observables=[error])
    values = run.get_values('error')
    steps = np.arange(len(bits))
    return [values[(steps % 3 == position) & (steps > 3)].mean() for position in range(3)]


rng = np.random.default_rng(7)
stream = rng.integers(0, 2, 600).astype(float)
stream[2::3] = np.logical_xor(stream[0::3], stream[1::3])
fig, (losses, positions) = plt.subplots(1, 2, figsize=(10, 4))
for name in ('loss', 'xor_error', 'random_error'):
    losses.plot(trajectory.times, trajectory.get_values(f'training/{name}'), label=name)
losses.set_xlabel('epoch')
losses.set_ylabel('mean squared error')
losses.set_title('Training')
losses.legend()
width = 0.35
moments = ((-width / 2, 0, 'before training'), (width / 2, -1, 'after training'))
for shift, frame, label in moments:
    positions.bar(np.arange(3) + shift, errors_by_position(frame, stream), width, label=label)
positions.set_xticks(range(3), ['first bit', 'second bit', 'third bit (XOR)'])
positions.set_ylabel('mean squared error')
positions.set_title('Prediction error on a fresh stream')
positions.legend()
fig.tight_layout()
plt.show()
