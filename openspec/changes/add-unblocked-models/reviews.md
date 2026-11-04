# Gallery review

The screenshots were not captured, since `nox -s gallery` was not run in this change. `Thumbnail shows` states what the
primary view logs, read from the recorded data. `Budget` is measured on a machine shared with three other agents.

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| szabo_cells | 200 cells in the periodic box, coloured by heading, moving together at the end | order 0.99 at density 1 for 200 cells; for 100 cells 0.23, 0.41, 0.99 at densities 0.05, 0.25, 1 | 207 s on the shared machine, 121 frames, 1.9 MB |
| active_passive_flocking | 45 active and 297 passive agents in the unit box of figure 10.7, flocks of active agents among still passive patches | with flocking on, active and passive mean distances 0.33 and 0.38 at the compactness 0.672; with flocking off within 0.007 of each other; polarization above 0.9 with every agent active | 20 s on the shared machine, 101 frames, 1.5 MB |
| koopman_model_predictive_control | output of the forced Duffing oscillator against `-sin(t)` | mean squared error, controlled against no input, at the 100 runs of 1000 samples of the review: Duffing 0.0020 and 0.82, Van der Pol 0.0018 and 0.95, motor 0.19 and 2.8e4 | 41 s, 201 frames, 69 KB |
| next_generation_reservoir | the three variables of the forecast of the Lorenz flow | forecast within 10% of the attractor size for 2.4 Lyapunov times in the example window, median 4.5 over 20 windows; it stays on the attractor afterwards | 9 s, 201 frames, 94 KB |
