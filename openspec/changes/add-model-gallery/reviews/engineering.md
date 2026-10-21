# Engineering review

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| chua | chaotic x, y, z of the double scroll | x-z projection winds around both scrolls; largest Lyapunov exponent 0.43 | 6 s script, 251 frames, 0.09 MB |
| pendulum_swing_up | swings growing until the pendulum flips over | energy rises in steps to omega^2 and stays; phase plane fills to the separatrix | 6 s script, 251 frames, 0.10 MB |
| pid_loop | output reaching the set point, dipping at the disturbance, recovering | PID returns to 1; P-only settles at 2/3, then lower after the disturbance | 5 s script, 201 frames, 0.09 MB |
| rlc_circuit | decaying ringing of q and its rate | matches the exact damped solution; resonance curve peaks at 1 / sqrt(L C) | 7 s script, 251 frames, 0.08 MB |
| thermostat | room warming to ~17.5 °C and swinging with the day | 13 leaks land on (setting + leak outside) / (1 + leak) | 5 s script, 193 frames, 0.06 MB |
| tunnel_diode_oscillator | voltage growing from 0.1 into a steady oscillation | four starts converge on one limit cycle of amplitude 1.16 (weak-nonlinearity estimate 1.15) | 5 s script, 251 frames, 0.07 MB |
| faraday_disk | disk spinning up while current and battery voltage settle | spin-up ends on a steady state; steady speed grows linearly with torque from 3 to 9 | 5 s script, 251 frames, 0.20 MB |
| impulsive_circuits | response kicked every 2 units settling into a periodic waveform | stroboscopic x and x' converge; last five periods coincide | 5 s script, 251 frames, 0.09 MB |
| memristor_oscillator | irregular oscillations of v, i, and w | v-i attractor; largest Lyapunov exponent 0.034 > 0 | 7 s script, 251 frames, 0.08 MB |
| mixed_mode_chua | small oscillations between large spikes | c = 0 small only, c = 0.00053 mixed, c = 0.00105 large only | 21 s script, 201 frames, 0.35 MB |
| power_grid | frequency deviations splitting to +10 and -10 with the added line | default ring synchronises (max deviation 0.00); added (0, 2) or doubled (0, 1) line does not (10.14) | 19 s script, 151 frames, 0.56 MB |
| regenerative_braking | speed falling as the capacitor voltage rises | stored energy reaches ~17% of the kinetic energy lost, peaking near the current reversal | 4 s script, 121 frames, 0.07 MB |
| vibration_energy_harvester | wheel and body oscillating with the road | voltage amplitude peaks near the two natural frequencies 7 and 74 rad/s | 18 s script, 251 frames, 0.16 MB |
| wheel_rail_contact | rail and sleeper dipping twice per bogie | two dips per bogie; peak deflection proportional to the load | 7 s script, 241 frames, 0.22 MB |
| delayed_car_following | speed swings growing from the leader to car 7 | amplitude grows along the chain for T = 1 and decays for T = 0.2, as omega < 2 lambda sin(omega T) predicts | 8 s script, 121 frames, 0.38 MB |
| intelligent_driver_model | 20 cars accelerating from rest to one common speed | final speeds for 10 to 60 cars match the IDM equilibrium | 7 s script, 201 frames, 0.16 MB |
| optimal_velocity | speeds fanning out into a stop-and-go wave | spread grows for headway 2 and decays for headway 4; late states loop around V(h) | 8 s script, 201 frames, 0.13 MB |
| lighthill_whitham_richards | density history with the shock edge moving upstream | shock position follows the Rankine-Hugoniot speed -0.1 | 8 s script, 51 frames, 0.03 MB |
| nagel_schreckenberg | space-time road with backward-leaning jam stripes | fundamental diagram: free flow (vmax - p) rho up to ~0.15, then falling flow | 7 s script, 201 frames, 7.95 MB |
