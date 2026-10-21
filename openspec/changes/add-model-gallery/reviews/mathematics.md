# Mathematics review

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| lorenz | butterfly attractor in 3D, both wings and their fixed points visible | successive z maxima lie on a thin tent-shaped Lorenz map | 0.5 s, 251 frames, 149 KB |
| rossler | folded single-band attractor in 3D | x maxima on a single-humped return map; spectral peak at 1/6 | 1.2 s, 241 frames, 164 KB |
| henon_map | folded bands of the strange attractor | zoom shows bands within bands; correlation dimension 1.19 vs ~1.21 | 3.5 s, 251 frames, 83 KB |
| logistic_map | orbit diagram with period doubling and windows | Benettin exponent touches 0 at doublings, positive in bands, ln 2 at r = 4 | 30.8 s, 201 frames, 124 KB |
| tent_map | orbit diagram with band merging, no cascade | Lyapunov exponent equals ln mu at every slope | 20.0 s, 201 frames, 254 KB |
| circle_map | devil's staircase of the winding number at K = 1 | plateaus widen with K; locked share grows to ~0.5 on a 201-point grid | 17.5 s, 101 frames, 35 KB |
| standard_map | chaotic sea and regular islands at K = 1.2 | momentum variance bounded at K = 0.8, diffusive near K²/4 at K = 2.5 and 5 | 5.1 s, 251 frames, 179 KB |
| cat_map | square of points drawn into parallel stripes after 3 steps | exponents ±0.962 = ±2 ln φ; occupied cells reach 1 by step 4 | 6.9 s, 201 frames, 48 KB |
| baker_map | Cantor stack of horizontal lines | Kaplan-Yorke dimension 1.610 vs Farmer-Ott-Yorke 1.609 | 5.2 s, 251 frames, 249 KB |
| intermittency_map | long laminar phases near 0 broken by bursts | laminar-duration CCDF slopes -1 (z = 2) and -2 (z = 1.5) | 14.0 s, 291 frames, 39 KB |
| doubling_map | occupied bins per step: one cell at step 0 filling the whole column by step ~7 | floating-point doubling collapses to 0 by step ~55; model stays uniform | 0.2 s, 101 frames, 97 KB |
| skew_tent_map | occupied bins per step: one cell at step 0 filling the whole column by step ~8 | return map on the two tent branches; flat invariant density 1 | 0.0 s, 41 frames, 67 KB |
| cubic_chaotic_map | 200 deterministic walks fanning out from 0 like a diffusion cone | mean square position linear with the Green-Kubo slope | 0.3 s, 251 frames, 457 KB |
| cusp_form | x jumping between branches as beta1 is swept up and down | hysteresis loop on the outer branches, jumps just past the folds ±0.385 | 5.0 s, 201 frames, 59 KB |
| saddle_node_form | x lingering near 0 for ~40 time units, then escaping | passage time follows π / √abs(mu) over three decades | 3.8 s, 241 frames, 35 KB |
| hopf_form | orbits from inside and outside winding onto the cycle | final amplitude 0 below mu = 0 and √mu above | 5.2 s, 251 frames, 75 KB |
| subcritical_hopf_form | x and y oscillating, flat while the oscillation is lost, then oscillating again | amplitudes on the cycle radius; bistability exactly in -1/4 < alpha < 0 | 3.6 s, 291 frames, 93 KB |
| takens_bogdanov_form | limit cycle inside the saddle separatrix | cycle grows from the Hopf point; period diverges near the homoclinic at ~-0.212 | 13.1 s, 251 frames, 90 KB |
| finite_time_blow_up | x curving up to 100 just before t* = 1 | log-log divergence on the exact solutions for p = 1.5, 2, 3 | 0.5 s, 248 frames, 43 KB |
| cantor_set | generations 0 to 6 stacked as bars, middle thirds removed at every level | length falls as (2/3)^n; box dimension log 2 / log 3 | 0.2 s, 9 frames, 52 KB |
| chaos_game | upright Sierpiński triangle of random points | box-counting dimension log 3 / log 2; Barnsley fern | 0.3 s, 31 frames, 5.2 MB |
| koch_curve | bright Koch curve after five generations | length grows as (4/3)^n; box dimension log 4 / log 3 | 0.2 s, 6 frames, 55 KB |
| sierpinski_substitution | Sierpiński gasket after six substitutions (caption names the carpet as figure-only) | areas fall geometrically; box dimensions log 3 / log 2 and log 8 / log 3 | 0.8 s, 7 frames, 146 KB |
| julia | yellow filled Julia set with spirals on an escape-time background | counts symmetric under z → -z; long tail of slow escapes | 0.6 s, 21 frames, 778 KB |
| mandelbrot | yellow Mandelbrot set with bulbs and filaments on escape-time bands | bounded area levels off near 1.506 | 0.2 s, 21 frames, 670 KB |
| network_structure_analysis | four coloured communities, densely linked inside, sparsely between | Poisson-like degrees; closed triads far above degree-preserving nulls | 2.5 s, 2 frames, 101 KB |
| birth_death_queue | three queue lengths, mostly 0 to 2, peaking at 5 | geometric occupation (1 - rho) rho^n; mean rho / (1 - rho) | 2.6 s, 300 frames, 49 KB |
| poisson_process | six counting staircases | counts Poisson with mean 50 at t = 50; Fano factor 1 within a few percent | 0.7 s, 251 frames, 37 KB |
| polya_urn | shares of red in six urns freezing at different values | limits follow Beta(1, 1) and Beta(2, 1) over 2000 urns | 7.7 s, 201 frames, 64 KB |
| random_walk | six continuous-time walks spreading | variance t with or without bias; mean drift 0.4 at bias 0.7; Gaussian at t = 100 | 5.4 s, 201 frames, 42 KB |
| persistent_random_walk | 300 walks fanning out in long straight runs | MSD matches the correlated-step formula, 4 t at long times | 0.0 s, 201 frames, 311 KB |
| random_walk_with_memory | 300 smooth walks spreading from 0 | exact MSD from the moment recursion, long-time slope 4 | 0.0 s, 201 frames, 1.0 MB |
| markov_source | emitted symbol with long runs of 0 and brief 1s | block-entropy differences drop to h ≈ 0.569 bits at m = 2 | 2.9 s, 300 frames, 25 KB |
| hidden_markov_source | hidden state, mostly 0 with short visits to 1 (symbols in the second view) | emission shares match E; share of symbol 0 is 0.6 | 2.7 s, 300 frames, 41 KB |
| on_off_intermittency | two variables resting near 0 between bursts up to ~1.4 | laminar durations distributed as tau^(-3/2) | 3.1 s, 300 frames, 58 KB |
| newton_raphson_map | three basins of the cube roots meeting along a fractal boundary | orbits end on the roots of their basins; boundary box dimension 1.41 | 0.5 s, 31 frames, 37 KB |
| sprott_jerk | folded band of the attractor in (x, x', x'') | single-curve return map of x maxima; exponents 0.041, 0, -0.641 summing to -A | 6.6 s, 251 frames, 94 KB |
| towel_map | folded sheet of points (y stretched 4x) | exponents 0.43, 0.38, -3.30; Kaplan-Yorke dimension 2.25 | 1.5 s, 251 frames, 160 KB |
| network_diffusion | karate club coloured by content, bright near the source, dark on the far side | distance to 1/n decays at λ2 = 0.469; mass drift 1e-13; farther nodes fill later | 0.3 s, 61 frames, 56 KB |
| pagerank | karate club with the two hubs largest and brightest | L1 error to networkx.pagerank falls to 1e-7, faster than 0.85^t; rank grows with degree | 0.2 s, 41 frames, 53 KB |
| network_random_walk | walkers crowding on the hubs 0 and 33 | share of walker-steps equals k_i / 2m | 0.2 s, 201 frames, 584 KB |
| bernoulli_process | six success staircases around 0.5 t | final counts match B(100, 0.5) and B(100, 0.2) over 2000 sequences | 1.1 s, 101 frames, 40 KB |
