# Social review

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| majority_rule | 100 x 100 lattice frozen into smooth domains | interface density falls from 0.5 and freezes near 0.1 within 10 steps; final share of opinion 1 is an S-curve of the initial share, amplifying the majority | 0.2 s, 21 frames, 0.13 MB |
| public_goods_game | cooperators (yellow) holding most of the lattice, defectors in patches | final share of cooperators is 0 below r ~ 3.74, rises between 3.74 and 5.49, and reaches 1 above, matching Szolnoki, Perc, and Szabo | 1.8 s, 201 frames, 0.41 MB |
| schelling | residents sorted into large single-kind patches | like-neighbour share climbs from 0.5 to 0.82 for threshold 0.4; final share exceeds every mild threshold, falling back for 0.8 | 7.0 s, 51 frames, 0.65 MB |
| spatial_prisoners_dilemma | chaotic patchwork of cooperator clusters among defectors at b = 1.85 | cooperators win most of the lattice below b = 1.8, fluctuate near 12 ln 2 - 8 ~ 0.32 between 1.8 and 2, and vanish above 2 | 0.9 s, 101 frames, 0.20 MB |
| maki_thompson | ignorants falling, spreaders rising and dying out, stiflers taking over, about 200 ignorants left | final ignorant share clusters at x = exp(-2 (1 - x)) = 0.203, tighter for 5000 than 200 people | 7 s script, 41 frames, 0.03 MB |
| simon_yule | number of distinct words growing linearly to 1000 | rank-frequency slope ~ -1 / (1 - 0.1) beyond the first word; distinct words = 0.1 x length | 4 s script, 101 frames, 0.03 MB |
| watts_cascades | network with a global cascade (yellow adopters) | probability of a global cascade falls from 1 to 0 between thresholds 0.16 and 0.26; outcomes all or nothing | 12 s script, 31 frames, 0.08 MB |
