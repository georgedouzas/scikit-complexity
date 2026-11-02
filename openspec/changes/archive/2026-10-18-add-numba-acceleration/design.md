# Design

## Context

After `add-element-groups`, laws are lowered once per group with `lambdify` and NumPy broadcasting, pair interactions
use a KD-tree with `np.add.at`, and steppers loop in Python. A prototype compiled a SymPy `Piecewise` law with
`lambdify(modules='math')` and `numba.vectorize`: compiling took 0.5 seconds, and 100 steps over a million elements took
0.09 seconds against 0.51 seconds with NumPy.

## Goals / Non-Goals

**Goals:**

- One compiled function per form function that fuses every law of every group, with no NumPy temporaries.
- Compiled stepper loops, so a frame of many internal steps costs one call from Python.
- The NumPy lowering kept as the reference that tests compare against.

**Non-Goals:**

- GPU kernels. Numba CUDA or JAX can register as further backends later.
- Compiling `ScipyODE`. It keeps calling the compiled drift from Python, which removes most of its cost.

## Decisions

### Code generation from SymPy

Each law expression is printed with SymPy's `NumPyPrinter` for scalars into the body of a generated function that loops
over members with `numba.prange`. The generated source of all laws of a form function is joined into one function and
compiled with `numba.njit(cache=True, parallel=True, fastmath=False)`.

- Alternative considered: `numba.vectorize` per law, as in the prototype. Rejected as the final form because each law
  would still be a separate pass over memory. It stays the fallback for laws that the printer cannot place in a loop.
- `fastmath` stays off, so the backends agree to 1e-12 and runs are reproducible.

### Backends are registered

`compile_system(system, backend='auto')` resolves the backend from a registry in `compile`. `auto` picks `numba` when
the groups of the system hold at least 1000 members in total and `numpy` otherwise. A probe measured Numba's first
compilation of one small function at 0.6 to 1.3 seconds, so a two-variable model would start seconds later on Numba
for no gain, while the disk cache removes the cost for repeated large systems. `Simulation` gains a
`backend` parameter passed through. A backend implements `lower(form, kernels, changes, size)` and returns the numeric
functions of the form with optional compiled stepper loops. Both backends lower the same backend-neutral kernels, so
the compiler builds kernels once and the backends only generate code from them.

### Layout as runtime data

Generated kernels read the start and row count of every entry from a layout array passed at call time, so the
generated source depends on the structure of a system and not on the size of its groups. An ideal gas of 10 and of a
million molecules share one compiled module. Kinds of laws with a plain Python callable fall back to the NumPy
functions as a whole, and the problem then has no compiled stepper loops.

### Compilation before the clock

Numba compiles on the first call, so `Simulation` calls every numeric function and compiled loop once while it
prepares a run, before the clock starts, and real-time runs do not count compilation as lag.

### Callables

A law value that is a `numba` dispatcher is called inside the generated loop with the member values as arguments. A
plain Python callable forces its slot onto the Python path, which runs after the compiled kernel and adds its result,
as the current lowering does.

### Compiled steppers

`RK4`, `EulerMaruyama`, `Gillespie`, and `Iteration` gain compiled inner loops that take the compiled form functions as
first-class arguments. Random numbers come from a NumPy `Generator` passed in, which Numba supports, so seeded runs stay
reproducible across backends where the algorithm draws numbers in the same order.

### Velocity Verlet

`VelocityVerlet` reads which components are positions from the compiled equations: a component whose equation is
another state symbol, as `dx/dt = v`, is a position and that symbol is its velocity. Every other component raises a
solver error. Each step kicks the velocities by half a step of the drift, moves the positions, and kicks again with
the drift at the new positions, so a step costs one drift evaluation after the first. A probe measured one drift of
100000 soft spheres at 5.25 ms on 8 threads, so `RK4`, with four evaluations per step, cannot keep a real-time clock
at frames of 0.01 for that size on such a machine, while velocity Verlet can, and it conserves energy over long runs.

### Cell lists

Pair interactions use a compiled cell list: members are binned into cells of the cutoff size, and each member visits
the neighbouring cells. Forces are accumulated in a per-thread buffer and reduced, which avoids write races.

### Kernel cache

Generated sources are written to a cache directory under the user cache path, named by a hash of the source, and
imported from there, so Numba's `cache=True` finds them in later processes.

## Risks / Trade-offs

- [The first compilation of a system takes seconds] → The disk cache removes it for later runs, and the NumPy backend
  stays available for quick experiments.
- [Numba lags new NumPy and Python releases] → The NumPy backend keeps working when Numba cannot be imported, with a
  warning, and the dependency bounds follow Numba's releases.
- [Parallel loops change the order of floating point sums] → Pair forces reduce per member in a fixed order, so results
  do not depend on the thread count.
