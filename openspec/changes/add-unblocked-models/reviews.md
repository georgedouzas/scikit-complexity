# Gallery review

The screenshots were not captured, since `nox -s gallery` was not run in this change. `Thumbnail shows` states what the
primary view logs, read from the recorded data. `Budget` is measured on a machine shared with three other agents.

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| szabo_cells | 200 cells in the periodic box, coloured by heading, moving together at the end | order 0.99 at density 1 for 200 cells; for 100 cells 0.23, 0.41, 0.99 at densities 0.05, 0.25, 1 | 207 s on the shared machine, 121 frames, 1.9 MB |
| active_passive_flocking | 90 active and 210 passive agents in the dense box | polarization above 0.9 with every agent active at packing 0.3, near 0 without alignment; the mean distances stay within 0.3, so no segregation, as the caption says | 47 s on the shared machine, 201 frames, 1.4 MB |
| koopman_model_predictive_control | output of the forced Duffing oscillator against `-sin(t)` | mean squared error, controlled against no input: Duffing 0.214 and 0.635, Van der Pol 0.0161 and 1.08, motor 0.0757 and 2.82e4 | 16 s, 201 frames, 73 KB |
| next_generation_reservoir | the three variables of the forecast of the Lorenz flow | forecast within 10% of the attractor size for 5.6 Lyapunov times; it stays on the attractor afterwards | 11 s, 201 frames, 66 KB |
