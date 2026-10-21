# Mathematics review

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| lorenz | butterfly attractor in 3D | successive z maxima lie on a thin tent-shaped Lorenz map | 0.5 s, 251 frames, 149 KB |
| rossler | folded single-band attractor in 3D | x maxima on a single-humped return map; spectral peak at 1/6 | 1.2 s, 241 frames, 164 KB |
| henon_map | folded bands of the strange attractor | zoom shows bands within bands; correlation dimension 1.19 vs ~1.21 | 3.5 s, 251 frames, 83 KB |
| logistic_map | orbit diagram with period doubling and windows | Benettin exponent touches 0 at doublings, positive in bands, ln 2 at r = 4 | 30.8 s, 201 frames, 124 KB |
| tent_map | orbit diagram with band merging, no cascade | Lyapunov exponent equals ln mu at every slope | 20.0 s, 201 frames, 254 KB |
| circle_map | devil's staircase of the winding number at K = 1 | plateaus widen with K; locked share grows to ~0.5 on a 201-point grid | 17.5 s, 101 frames, 35 KB |
| standard_map | chaotic sea and regular islands at K = 1.2 | momentum variance bounded at K = 0.8, diffusive near K²/4 at K = 2.5 and 5 | 5.1 s, 251 frames, 179 KB |
| cat_map | square of points drawn into parallel stripes after 3 steps | exponents ±0.962 = ±2 ln φ; occupied cells reach 1 by step 4 | 6.9 s, 201 frames, 48 KB |
| baker_map | Cantor stack of horizontal lines | Kaplan-Yorke dimension 1.610 vs Farmer-Ott-Yorke 1.609 | 5.2 s, 251 frames, 249 KB |
| intermittency_map | long laminar phases near 0 broken by bursts | laminar-duration CCDF slopes -1 (z = 2) and -2 (z = 1.5) | 14.0 s, 291 frames, 39 KB |
| doubling_map | points spreading over the unit interval and staying spread | floating-point doubling collapses to 0 by step ~55; model stays uniform | 0.1 s, 101 frames, 87 KB |
| skew_tent_map | points fanning out to fill the unit interval | return map on the two tent branches; flat invariant density 1 | 0.1 s, 41 frames, 54 KB |
| cubic_chaotic_map | deterministic walkers fanning out diffusively | mean square position linear with the Green-Kubo slope | 0.3 s, 251 frames, 457 KB |
| cusp_form | x jumping between branches as beta1 is swept up and down | hysteresis loop on the outer branches, jumps just past the folds ±0.385 | 5.0 s, 201 frames, 59 KB |
| saddle_node_form | x lingering near 0 for ~40 time units, then escaping | passage time follows π / √abs(mu) over three decades | 3.8 s, 241 frames, 35 KB |
| hopf_form | orbits from inside and outside winding onto the cycle | final amplitude 0 below mu = 0 and √mu above | 5.2 s, 251 frames, 75 KB |
| subcritical_hopf_form | oscillation lost and regained at the same alpha | amplitudes on the cycle radius; bistability exactly in -1/4 < alpha < 0 | 3.6 s, 291 frames, 93 KB |
| takens_bogdanov_form | limit cycle inside the saddle separatrix | cycle grows from the Hopf point; period diverges near the homoclinic at ~-0.212 | 13.1 s, 251 frames, 90 KB |
| finite_time_blow_up | x curving up to 100 just before t* = 1 | log-log divergence on the exact solutions for p = 1.5, 2, 3 | 0.5 s, 248 frames, 43 KB |
| cantor_set | intervals losing their middle thirds | length falls as (2/3)^n; box dimension log 2 / log 3 | 0.4 s, 9 frames, 28 KB |
| chaos_game | Sierpiński triangle drawn by random points | box-counting dimension log 3 / log 2; Barnsley fern | 1.7 s, 31 frames, 5.0 MB |
| koch_curve | Koch curve from the L-system | length grows as (4/3)^n; box dimension log 4 / log 3 | 0.2 s, 6 frames, 40 KB |
| sierpinski_substitution | Sierpiński gasket and carpet | areas fall geometrically; box dimensions log 3 / log 2 and log 8 / log 3 | 0.8 s, 7 frames, 146 KB |
| julia | escape counts around the connected Julia set of c = -0.8 + 0.156i | counts symmetric under z → -z; long tail of slow escapes | 0.6 s, 21 frames, 778 KB |
| mandelbrot | escape-time bands around the Mandelbrot set | bounded area levels off near 1.506 | 0.2 s, 21 frames, 670 KB |
| network_structure_analysis | four planted communities of a block model | Poisson-like degrees; closed triads far above degree-preserving nulls | 2.5 s, 2 frames, 101 KB |
| birth_death_queue | three queue lengths at half load | geometric occupation (1 - rho) rho^n; mean rho / (1 - rho) | 2.6 s, 300 frames, 49 KB |
| poisson_process | six counting staircases | counts Poisson with mean 50 at t = 50; Fano factor 1 within a few percent | 0.7 s, 251 frames, 37 KB |
| polya_urn | shares of red in six urns freezing at different values | limits follow Beta(1, 1) and Beta(2, 1) over 2000 urns | 7.7 s, 201 frames, 64 KB |
| random_walk | six continuous-time walks spreading | variance t with or without bias; mean drift 0.4 at bias 0.7; Gaussian at t = 100 | 5.4 s, 201 frames, 42 KB |
| persistent_random_walk | walkers running in long straight stretches | MSD matches the correlated-step formula, 4 t at long times | 0.0 s, 201 frames, 311 KB |
| random_walk_with_memory | smooth paths of walkers with velocity memory | exact MSD from the moment recursion, long-time slope 4 | 0.0 s, 201 frames, 1.0 MB |
| markov_source | symbol 0 in long stretches, symbol 1 briefly | block-entropy differences drop to h ≈ 0.569 bits at m = 2 | 2.9 s, 300 frames, 25 KB |
| hidden_markov_source | symbols loosely following the hidden state | emission shares match E; share of symbol 0 is 0.6 | 3.7 s, 300 frames, 41 KB |
| on_off_intermittency | rests near 0 broken by bursts | laminar durations distributed as tau^(-3/2) | 3.1 s, 300 frames, 58 KB |
