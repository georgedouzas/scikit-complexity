# Engineering review

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| chua | chaotic x, y, z of the double scroll | x-z projection winds around both scrolls; largest Lyapunov exponent 0.43 | 6 s script, 251 frames, 0.09 MB |
| pendulum_swing_up | swings growing until the pendulum flips over | energy rises in steps to omega^2 and stays; phase plane fills to the separatrix | 6 s script, 251 frames, 0.10 MB |
| pid_loop | output reaching the set point, dipping at the disturbance, recovering | PID returns to 1; P-only settles at 2/3, then lower after the disturbance | 5 s script, 201 frames, 0.09 MB |
| rlc_circuit | decaying ringing of q and its rate | matches the exact damped solution; resonance curve peaks at 1 / sqrt(L C) | 7 s script, 251 frames, 0.08 MB |
| thermostat | room warming to ~17.5 °C and swinging with the day | 13 leaks land on (setting + leak outside) / (1 + leak) | 5 s script, 193 frames, 0.06 MB |
| tunnel_diode_oscillator | voltage growing from 0.1 into a steady oscillation | four starts converge on one limit cycle of amplitude 1.16 (weak-nonlinearity estimate 1.15) | 5 s script, 251 frames, 0.07 MB |
