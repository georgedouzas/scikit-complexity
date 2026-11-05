"""
# A translated world population

Meadows describes a population with a reinforcing loop of births and a balancing loop of deaths. The world of 2007 holds
6.6 billion people, with 21 births and 9 deaths a year for every 1,000 people. In her scenario, fertility falls steadily
until it equals mortality in 2035, and then both stay constant, so the population levels off. This example writes that
scenario as a Vensim file, translates the file into a system module, runs the module, and plots the shift in the
dominance of the two loops.
"""

# %%
# The system file
# --------------
# The file states the scenario in Vensim's language. `SystemDynamicsTranslator` reads it and writes the source of a
# module, which the example saves and imports like any system.

import importlib
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import LoopAnalysis
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.translators import SystemDynamicsTranslator
from skcomplex.core.engine.viz import RerunObserver

FILE_TEXT = """
Population = INTEG(Births - Deaths, 6.6) ~ Billion People ~ |
Births = Population * Fertility ~ Billion People/Year ~ |
Deaths = Population * Mortality ~ Billion People/Year ~ |
Fertility = 0.021 + RAMP((Mortality - 0.021) / 28, 2007, 2035) ~ 1/Year ~ |
Mortality = 0.009 ~ 1/Year ~ |
INITIAL TIME = 2007 ~ Year ~ |
FINAL TIME = 2120 ~ Year ~ |
TIME STEP = 0.5 ~ Year ~ |
SAVEPER = 0.5 ~ Year ~ |
"""

folder = Path(tempfile.mkdtemp())
(folder / 'world.mdl').write_text(FILE_TEXT)
translator = SystemDynamicsTranslator(folder / 'world.mdl').prepare().run()
(folder / 'translated_world.py').write_text(translator.source_)
sys.path.insert(0, str(folder))
module = importlib.import_module('translated_world')

# %%
# The run
# -------
# The module holds the settings of the file. Time in the module starts at zero, so a year is `module.START` plus the
# time of a frame. The view shows the population of every frame.

system = module.create_system()
observer = RerunObserver('translated_world_population', save_path='translated_world_population.rrd', interval=1)
simulation = Simulation(system, dt=module.DT, observers=[observer], observables=module.observables(system))
trajectory = simulation.prepare().run(until=module.UNTIL).trajectory_

# %%
# Shifting dominance
# ------------------
# `LoopAnalysis` finds the reinforcing loop through births and the balancing loop through deaths. The gain of each loop
# is its fertility or its mortality. The reinforcing loop dominates while fertility is above mortality, and the two
# gains meet in 2035, when births balance deaths and the population stays near 7.8 billion.

dominance = LoopAnalysis('dominance').prepare(trajectory).run()
years = module.START + dominance.times_
fig, (stocks, gains) = plt.subplots(1, 2, figsize=(10, 4))
stocks.plot(years, trajectory.values('world.population'))
stocks.set_xlabel('year')
stocks.set_ylabel('billion people')
stocks.set_title('The population levels off')
for index, nodes in enumerate(dominance.loops_):
    gains.plot(years, np.abs(dominance.gains_[:, index]), label=f'{dominance.polarities_[index]}: {nodes[1]}')
gains.set_xlabel('year')
gains.set_ylabel('absolute gain per year')
gains.set_title('The gains of the two loops meet in 2035')
gains.legend()
fig.tight_layout()
plt.show()
