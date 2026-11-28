## 1. Fields (#45, #46)

- [x] 1.1 Read the initial state of the tangent dynamics as a vector, and take the Jacobian from the backend or by
      finite differences above the symbolic member limit, tested in the mirrored test of the Lyapunov module
- [x] 1.2 Linearise `partial` derivatives in the dispersion relation, tested in `tests/core/analysis/test_stability.py`

## 2. Defects (#47)

- [x] 2.1 Make the wrapped differences antisymmetric and document it, tested in
      `tests/core/analysis/test_defects.py`

## 3. Reaction diffusion (#48)

- [ ] 3.1 Profile a 40 by 40 Schnakenberg run and remove the dominant cost, with a timing check kept out of the quick
      tests

## 4. Close

- [ ] 4.1 Run the fast checks, the tests of the changed modules, and the full CI on `development`
- [ ] 4.2 Close issues #45 to #48 with a comment naming the commit
