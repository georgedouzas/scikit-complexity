# Chemistry review

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| brusselator | hexagonal spots of `u` | structure factor peaks in the Turing band (q ~ 0.78 against fastest mode 0.6) | 16 s script, 101 frames, 1.77 MB |
| brusselator_ode | `u`, `v` relaxation oscillations | runs from four starts converge on one limit cycle around the unstable fixed point (1, 3) | 8 s script, 251 frames, 0.06 MB |
| catalyst_pellet | `c` decays from 1 to the inhibited state 0.68 | three roots of dc/dt; 21 runs split at the unstable root 0.2 | 18 s script, 201 frames, 0.05 MB |
| chemical_langevin_equation | `x` leaves 250 for the high state | 30 runs from the unstable state split between the stable roots 85 and 570 | 32 s script, 201 frames, 0.08 MB |
| cubic_autocatalysis | `a`, `b` settle in the reacting state | grid of starts splits between washout and reacting state across the saddle | 10 s script, 251 frames, 0.06 MB |
| cyclic_cellular_automaton | spiral and diagonal waves of 14 levels | advancing share rises to ~0.9; phase correlation reaches farther over time | 11 s script, 151 frames, 2.35 MB |
| gray_scott | spots of `v` filling the domain | radial power spectrum peaks at the spot spacing | not captured in this worktree |
| hodgepodge | waves and fronts of illness (g = 10) | mean cell autocorrelation peaks at the 14-step cycle | 11 s script, 151 frames, 2.01 MB |
| michaelis_menten | E, S, C, P: fast binding then steady conversion | 5 Gillespie runs scatter around the rate equations; E + C and S + C + P conserved | 9 s script, 101 frames, 0.05 MB |
| nanoparticle_corona | albumin corona replaced by lipoprotein (Vroman effect) | complexes tend to the mass-action equilibrium, lipoprotein holds nearly all | not captured in this worktree |
| oregonator | bromide spikes of a relaxation oscillation | log-scale limit cycle; spike intervals all 6.20 | 9 s script, 201 frames, 0.07 MB |
| pulsed_electrochemical_reactor | `eta` spikes and `c` dips locked to the pulses | folded periods collapse; stroboscopic `c` converges to 0.63 | 9 s script, 241 frames, 0.08 MB |
| schnakenberg | regular Turing spots of `u` | structure factor peaks in the unstable band (q ~ 0.48 against fastest mode 0.42) | 28 s script, 101 frames, 2.92 MB |
