# Social review

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| majority_rule | 100 x 100 lattice frozen into smooth domains | interface density falls from 0.5 and freezes near 0.1 within 10 steps; final share of opinion 1 is an S-curve of the initial share, amplifying the majority | 0.2 s, 21 frames, 0.13 MB |
| public_goods_game | cooperators (yellow) holding most of the lattice, defectors in patches | final share of cooperators is 0 below r ~ 3.74, rises between 3.74 and 5.49, and reaches 1 above, matching Szolnoki, Perc, and Szabo | 1.8 s, 201 frames, 0.41 MB |
| schelling | residents sorted into large single-kind patches | like-neighbour share climbs from 0.5 to 0.82 for threshold 0.4; final share exceeds every mild threshold, falling back for 0.8 | 7.0 s, 51 frames, 0.65 MB |
| spatial_prisoners_dilemma | chaotic patchwork of cooperator clusters among defectors at b = 1.85 | cooperators win most of the lattice below b = 1.8, fluctuate near 12 ln 2 - 8 ~ 0.32 between 1.8 and 2, and vanish above 2 | 0.9 s, 101 frames, 0.20 MB |
| stochastic_schelling | agents piled into a few full 5 x 5 blocks, the others empty, at beta = 4 | spread of block densities stays near the random 0.1 for small beta, rises steeply around beta = 2, saturates at 0.5; densities bimodal at 0 and 1 for beta = 4 | 27.8 s, 151 frames, 0.20 MB |
| threshold_adoption | linked groups of adopters (yellow) among unreached people (purple) on grey, radius 9 | by infection the final share jumps from the innovators to everyone just below the continuum-percolation degree k_c ~ 4.51 (finite size); by influence it stalls near 0.1 for every circle size | 8.3 s, 31 frames, 0.07 MB |
| social_force | two crowds sorted into lanes of single direction across the corridor | lane order rises to 1 (20 each way) and ~0.87 (40) within 20 s with speed reaching the desired 1.34 m/s; at 60 each way lanes keep breaking and speed stays ~1 m/s | 5.7 s, 151 frames, 0.80 MB |
