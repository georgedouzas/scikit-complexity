"""
# Hopfield network

A Hopfield network stores a few patterns in its Hebbian weights and, started from a copy of the first pattern with
three neurons in ten flipped, repairs the copy within one or two sweeps of asynchronous updates: an associative
memory. Store too many patterns and recall fails.
"""

# %%
# The network has 100 neurons and stores 5 random patterns. After the run, the states of the neurons at every step are
# logged once as 10 by 10 images in two rows, the corrupted start at the top left and the stored pattern at the bottom
# right, and the blueprint puts these images first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'hopfield'
N = 100


def spins(trajectory, frame):
    """Return the states of the neurons at one frame, in the order of their ids."""
    snapshot = trajectory.agents('neurons', frame)
    return snapshot.values['spin'][np.argsort(snapshot.ids)]


patterns = np.random.default_rng(1).choice([-1.0, 1.0], size=(5, N))
blueprint = rrb.Blueprint(
    rrb.Spatial2DView(
        origin="/recall",
        name="recall, step by step, and the stored pattern",
        visual_bounds=rrb.VisualBounds2D(x_range=[0, 288], y_range=[0, 192]),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL, patterns=patterns, noise=0.3, random_state=0)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=4)
frames = [spins(simulation.trajectory_, frame).reshape(10, 10) for frame in range(5)] + [patterns[0].reshape(10, 10)]
tiles = [np.pad(frame, 1) for frame in frames]
strip = np.block([tiles[:3], tiles[3:]])
palette = np.array([[70, 70, 80], [68, 1, 84], [253, 231, 37]], dtype=np.uint8)
image = palette[np.sign(strip).astype(int) % 3].repeat(8, axis=0).repeat(8, axis=1)
observer.recording_.log("recall", rr.Image(image), static=True)
observer.recording_.flush()

# %%
# Overlap and capacity
# --------------------
# The overlap `m = Σ s_i ξ_i / n` with the first pattern should jump from its start, near 0.4 for a copy with three
# neurons in ten flipped, to 1 within a sweep. Recall should stay near perfect while the number of patterns is well
# below the capacity of about `0.14 n = 14`, and the final overlap should fall as the network is loaded beyond it.

overlap = [spins(simulation.trajectory_, frame) @ patterns[0] / N for frame in range(5)]
loads = [2, 5, 8, 11, 14, 17, 20, 25]
final = [
    np.mean(
        [
            spins(
                Simulation(
                    create(MODEL, n_neurons=N, n_patterns=p, noise=0.1, random_state=seed),
                    dt=1,
                    random_state=seed,
                )
                .run(until=4)
                .trajectory_,
                -1,
            )
            @ np.random.default_rng(seed).choice([-1.0, 1.0], size=(p, N))[0]
            / N
            for seed in range(6)
        ],
    )
    for p in loads
]
fig, (recall, capacity) = plt.subplots(1, 2, figsize=(10, 4))
recall.plot(range(5), overlap, 'o-')
recall.set_ylim(0, 1.05)
recall.set_xlabel('sweep')
recall.set_ylabel('overlap with the first pattern')
recall.set_title('Recall from a corrupted copy')
capacity.plot(loads, final, 'o-')
capacity.axvline(0.14 * N, color='black', linestyle=':', label='0.14 n')
capacity.set_xlabel('stored patterns')
capacity.set_ylabel('final overlap, mean of 6 runs')
capacity.set_title('Recall fails beyond the capacity')
capacity.legend()
fig.tight_layout()
plt.show()
