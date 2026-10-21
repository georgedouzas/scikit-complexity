"""
# Evolution of bit strings

A population of random 8-bit genomes evolves by fitness-proportional reproduction and rare mutation on an additive
landscape, where every bit contributes on its own. Where the two alleles differ much the better one spreads, where
they differ little drift can fix the worse one, and the mean fitness climbs while the population loses its diversity.
"""

# %%
# The population has 100 genomes; every step a tenth of them die and are replaced by copies of parents chosen in
# proportion to their fitness, with one bit flipped in one copy in twenty. The model only logs the size of the
# population, which stays 100. After the run, the share of genomes carrying the better allele at each of the 8 loci
# (rows, sorted by the gap between the two alleles, largest first) at every step (columns) is logged once as an
# image, dark for none and yellow for all, and the blueprint puts it first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'bit_string_evolution'

blueprint = rrb.Blueprint(
    rrb.Spatial2DView(origin='/alleles', name='share of the better allele, locus by step'),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, random_state=0), dt=1, observers=[observer], random_state=0).run(until=200)
landscape = np.random.default_rng(0)
zero_values, one_values = landscape.random(8), landscape.random(8)
trajectory = simulation.trajectory_
snapshots = [trajectory.agents('genomes', frame) for frame in range(len(trajectory.times))]
bits = np.array([(snapshot.values['genome'].astype(int)[:, None] >> np.arange(8)) & 1 for snapshot in snapshots])
better = np.where(one_values > zero_values, bits.mean(axis=1), 1 - bits.mean(axis=1)).T
order = np.argsort(np.abs(one_values - zero_values))[::-1]
image = plt.get_cmap('viridis')(better[order])[:, :, :3].repeat(12, axis=0)
observer.recording_.log('alleles', rr.Image((255 * image).astype(np.uint8)), static=True)
observer.recording_.flush()

# %%
# Climbing the landscape
# ----------------------
# The fitness of a genome is the mean over its loci of the value of its allele, so the best genome takes the better
# allele everywhere. The mean fitness should rise from the random start towards that optimum, though drift in a
# population of 100 can lose the best genome for a while, and the number of distinct genomes should fall from about 80
# as selection fixes the better alleles.

optimum = np.maximum(zero_values, one_values).mean()
fitness = np.array([snapshot.values['fitness'] for snapshot in snapshots])
distinct = [len(np.unique(snapshot.values['genome'])) for snapshot in snapshots]
fig, (climb, diversity) = plt.subplots(1, 2, figsize=(11, 4))
climb.plot(trajectory.times, fitness.mean(axis=1), label='mean fitness')
climb.plot(trajectory.times, fitness.max(axis=1), label='best in the population')
climb.axhline(optimum, color='black', linestyle=':', label='optimum')
climb.set_xlabel('step')
climb.set_ylabel('fitness')
climb.set_title('Selection climbs the additive landscape')
climb.legend()
diversity.plot(trajectory.times, distinct)
diversity.set_xlabel('step')
diversity.set_ylabel('distinct genomes')
diversity.set_title('Diversity falls as the better alleles spread')
fig.tight_layout()
plt.show()
