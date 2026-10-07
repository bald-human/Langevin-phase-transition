# Fluid–solid phase transition with Langevin dynamics

Agent-based simulation of Lennard-Jones particles in an implicit solvent. On slow cooling the system goes from a gas-like state through local order to a hexagonal crystal, including the formation and healing of crystal defects.

<!-- TODO: add the four snapshots (report Fig. 4.1) as an image, e.g. figures/crystallisation.png -->

## Model

- **Langevin dynamics** (m = k_B = 1): drag γ plus Gaussian white noise of strength √(2γTΔt), i.e. a stochastic thermostat
- **Lennard-Jones** pair forces with cutoff r_c = 2.5σ
- 2D periodic box with the **minimum-image convention**
- Vectorised NumPy pair computation; forces applied with Newton's third law
- Linear cooling schedule from T = 1.3 down to T_min = 0.05

## Validation

| Check | Expected | Measured |
|---|---|---|
| Mean velocity ⟨v⟩ | 0 (no preferred direction) | −1.76 × 10⁻³ |
| Equipartition ⟨v²⟩ = 2T at T_min = 0.05 | 0.100 | 0.108 (8.3%) |
| Radial distribution g(r) | peaks at triangular-lattice shells a·√(i²+ij+j²), a = 2^(1/6)σ | peaks sharpen and align with theory on cooling |

<!-- TODO: add the g(r) figure (report Fig. 5.2c) -->

The 8.3% excess in ⟨v²⟩ is attributed to the first-order integration scheme. A splitting integrator such as BAOAB would be the natural fix.

## Run

```bash
pip install numpy matplotlib
python phase_transition.py
```

Parameters used for the report: N = 360, box = 80σ, Δt = 0.05, γ = 1.3, ε = 2.4, σ = 3.0, T from 1.3 to 0.05 at a cooling rate of 0.0015 per step.

## Report

Full write-up (PDF): <!-- TODO: add report PDF -->

## Possible extensions

- BAOAB Langevin integrator for more accurate sampling
- Cell lists or neighbour lists to replace the O(N²) pair search, for larger N
- Systematic parameter sweep in (ε, γ, T)
