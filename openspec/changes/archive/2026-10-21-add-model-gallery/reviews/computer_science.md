# Computer science review

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| elementary_automaton | rule 30 triangle with irregular interior from one cell | rule 90 live cells equal 2^b(t) exactly; rule 30 light-cone density near 1/2 | 0.2 s, 101 frames, 236 KB |
| game_of_life | sparse still lifes, blinkers, and debris on an empty board after 200 generations | density falls from 0.3 and levels off | 0.2 s, 201 frames, 821 KB |
| highlife | sparse board of debris after the soup decays | soup density levels near 0.1, above Life; replicator grows only under B36/S23 | 0.2 s, 201 frames, 1.0 MB |
| langtons_ant | chaotic blob with the diagonal highway leaving it | distance grows 2√2 and black cells 12 per 104 steps after ~10,000 steps | 9.3 s, 121 frames, 366 KB |
| noisy_automata | rule 18 space-time texture of triangles with noise defects | adjacent live pairs decay as t^(-1/2) without noise, level off with noise | 0.3 s, 201 frames, 1.5 MB |
| diffusion_automaton | Gaussian bump spread from a square | variance grows as 99/12 + 2rt; profile matches the Gaussian; mass drift 6e-14 | 0.2 s, 201 frames, 267 KB |
| unbounded_alphabet_automaton | Pascal triangle on a log colour scale | rows equal C(t, k); row sums 2^t; largest value 2^t √(2/πt) | 0.3 s, 51 frames, 37 KB |
| free_group_automaton | Cayley tree of radius 6 on rings, frozen blocks of agreeing cells | agreement along edges rises from 1/2, changed cells fall to 0 | 0.1 s, 21 frames, 207 KB |
| lamplighter_automaton | Diestel-Leader ball on rings with a flickering parity pattern | number of ones fluctuates under the linear rule; invertibility on the level-3 quotient | 2.0 s, 21 frames, 137 KB |
| heisenberg_automaton | Heisenberg ball on rings with frozen clusters | ball sizes approach n^4; changed cells fall to 0 | 0.1 s, 21 frames, 57 KB |
| baumslag_solitar_automaton | BS(1, 2) ball on rings with frozen clusters | ratio of consecutive ball sizes settles below 3; density settles | 1.4 s, 21 frames, 68 KB |
| grigorchuk_automaton | Grigorchuk ball of 271 elements on rings, frozen clusters | ball sizes 1 to 643 for radii 0 to 10; local log-log slope rises, log B(n)/n falls | 4.2 s, 21 frames, 56 KB |
| linear_ring_automaton | Laplacian modulo 3 from one cell, a nested triangle of three colours | nonzero cells stay inside the light cone 2t + 1; not invertible on the ring, surjective on Z | 0.3 s, 41 frames, 34 KB |
| subshifts | space-time image of the golden mean chain, never two adjacent ones | every frame admissible; density near 0.276; entropy of words nears log2 of phi | 0.3 s, 51 frames, 98 KB |
| bit_string_evolution | share of the better allele per locus over time; strong loci fix, weak ones drift | mean fitness climbs towards the optimum; distinct genomes fall from ~80 to ~10 | 0.2 s, 201 frames, 41 KB |
| robby_ga | mean and best score rising from -220 to above 0, then stepping up | mean near -200 at start, near 0 after ~30 generations; best climbs to ~35 | 9.3 s, 291 frames, 166 KB |
| hopfield | corrupted pattern repaired in one sweep, next to the stored pattern | overlap 0.5 to 1 in one sweep; final overlap falls beyond 0.14 n patterns | 2.4 s, 5 frames, 56 KB |
| diluted_hopfield | overlap rising from 0.3 and levelling near 0.92 within ~5 updates | overlaps follow the annealed map of Bertin Eq. (6.55) | 2.1 s, 21 frames, 303 KB |
| echo_state_network | three units in two copies of the reservoir, started apart, whose pairs of lines merge within ~30 steps and then follow the sine input | distance between the copies falls exponentially for spectral radius 0.5 and 0.9, not for 2.0 | 1.4 s, 101 frames, 232 KB |
