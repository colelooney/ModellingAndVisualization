# PDE Solvers: Cahn–Hilliard and Poisson Equation Simulation

This repository contains numerical implementations of two important partial differential equations:

- **Cahn–Hilliard Equation** (phase separation dynamics)
- **Poisson Equation** (electrostatics and magnetostatics)

Both solvers are written in Python and include:
- Finite difference discretisations
- Iterative solvers (Jacobi, Gauss–Seidel, SOR)
- Visualisation tools (contours, vector fields, animations)
- Quantitative analysis (scaling laws, convergence, energy minimisation)

---

# Model Definitions

## 1. Cahn–Hilliard Equation

The Cahn–Hilliard equation models **phase separation** in binary mixtures.

### Governing Equation

$\[
\frac{\partial \phi}{\partial t} = \nabla^2 \mu
\]$

where the chemical potential is:

\[
\mu = -\phi + \phi^3 - \nabla^2 \phi
\]

### Physical Meaning
- \(\phi(x, y)\): local composition field
- System evolves to minimise free energy
- Leads to **spinodal decomposition** and domain formation

### Free Energy Functional

\[
F = \int \left(-\frac{1}{2}\phi^2 + \frac{1}{4}\phi^4 + \frac{1}{2}|\nabla \phi|^2 \right) dV
\]

### Numerical Implementation

- 2D lattice: \(N \times N\)
- Periodic boundary conditions
- Finite difference Laplacian
- Explicit time stepping:
  - Compute \(\mu\)
  - Update: \(\phi \leftarrow \phi + dt \nabla^2 \mu\)

### Observables

- Total free energy vs time
- Domain formation patterns
- Convergence to equilibrium

---

## 2. Poisson Equation

The Poisson equation governs electrostatic and magnetostatic potentials.

### Governing Equation

\[
\nabla^2 \Phi = -\rho
\]

### Physical Interpretations

#### Electrostatics (Monopole)
- \(\Phi\): electric potential
- \(\mathbf{E} = -\nabla \Phi\)
- Expected scaling:
  - \(E \sim r^{-2}\)
  - \(\Phi \sim r^{-1}\)

#### Magnetostatics (Wire)
- Solve for vector potential \(A_z\)
- Magnetic field:
  \[
  \mathbf{B} = \nabla \times \mathbf{A}
  \]
- Expected scaling:
  - \(B \sim r^{-1}\)
  - \(A_z \sim \ln r\)

---

# Numerical Methods

## Discretisation

- 3D lattice: \(N \times N \times N\)
- Finite difference Laplacian (7-point stencil)
- Dirichlet boundary conditions (\(\Phi = 0\) at edges)

---

## Solvers Implemented

### 1. Jacobi Method
- Fully explicit
- Uses previous iteration values only
- Simple but slow convergence

### 2. Gauss–Seidel Method
- Updates in-place
- Faster convergence than Jacobi
- Uses **red-black ordering** for efficiency

### 3. Successive Over-Relaxation (SOR)

\[
\phi^{new} = (1 - \omega)\phi^{old} + \frac{\omega}{6}(\text{neighbors} + dx^2 \rho)
\]

- Accelerated Gauss–Seidel
- Optimal \(\omega \approx 1.9\)–\(1.95\)
- Includes:
  - Standard implementation
  - **Numba-optimised version** for speed

---

# Features

## Cahn–Hilliard
- Energy minimisation tracking
- Equilibrium detection via threshold
- Animation of phase separation
- Free energy plots

## Poisson Solver
- Multiple solver choices (Jacobi, GS, SOR)
- Electric and magnetic field computation
- Contour plots of potential
- Vector field visualisation
- Scaling law verification via log–log fits
- SOR parameter tuning

---

# Measured Observables

## Cahn–Hilliard
- Total free energy:
  \[
  F(t)
  \]
- Convergence based on energy fluctuations
- Pattern formation dynamics

## Poisson Equation

### Monopole Case
- Electric field magnitude vs distance:
  \[
  |E(r)| \sim r^{-2}
  \]
- Potential:
  \[
  \Phi(r) \sim r^{-1}
  \]

### Wire Case
- Magnetic field:
  \[
  |B(r)| \sim r^{-1}
  \]
- Vector potential:
  \[
  A_z \sim \ln r
  \]

---

# Usage

## Requirements

- Python ≥ 3.8
- NumPy
- Matplotlib
- SciPy
- argparse
- numba (for SOR acceleration)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Simulations
### Cahn Hilliard
```bash
python CahnHilliard.py [OPTIONS]
```

#### Arguments
| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--size` | `-N` | int | 100 | Grid size (N × N) |
| `--phi0` |  | float | 0 | Average initial composition |
| `-dx` |  | float | 1 | Spatial step size |
| `-dt` |  | float | 1e-4 | Time step size |
| `--num_iter` |  | int | 50000 | Maximum number of iterations |
| `--threshold` |  | float | 1e-6 | Energy change threshold for convergence |
| `--animate` |  | action |  | Animate phase separation evolution |

### Poisson
```bash
python Poisson.py [OPTIONS]
```
#### Arguments
| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--size` | `-N` | int | 100 | Grid size (N × N × N) |
| `--phi0` |  | float | 0 | Initial potential value (with noise) |
| `-dx` |  | float | 1 | Spatial step size |
| `-dt` |  | float | 1e-4 | Time step (not used in static solve, included for consistency) |
| `--num_iter` |  | int | 50000 | Maximum number of iterations |
| `--threshold` |  | float | 1e-6 | Convergence threshold |
| `--rho` | `-R` | str | monopole | Charge distribution (`monopole`, `wire`) |
| `--solver` |  | str | gauss_seidel | Solver algorithm (`jacobi`, `gauss_seidel`, `sor`) |
| `-w` |  | float | 1.94 | Relaxation parameter for SOR |
| `--animate` |  | action |  | Animate solver evolution (midplane slice) |
| `--sor_iter` |  | action |  | Sweep over relaxation parameter \(\omega\) and analyse convergence |

### Examples
- solve Poisson equation (monopole)
```bash
python Poisson.py --rho monopole --solver gauss_seidel
```

- solve with SOR
```bash
python Poisson.py --rho monopole --solver sor -w 1.94
```

- Tune SOR parameter
```bash
python Poisson.py --sor_iter
```

- Animate solver
```bash
python Poisson.py --solver sor -w 1.94 --animate
```

# Output Files

The simulations generate a range of outputs for both **visualisation** and **quantitative analysis**. These are automatically saved to the working directory.

## Plots

The following figures are produced depending on the simulation type:

### Cahn–Hilliard
- `free_energy_phi*.png`  
  - Free energy vs iteration number  
  - Used to verify convergence to equilibrium

### Poisson (Monopole)
- `Electric_vs_distance.png`  
  - Log–log plot of electric field magnitude vs distance  
  - Includes fitted slope and theoretical \(r^{-2}\) comparison  

- `potential_vs_distance.png`  
  - Log–log plot of electric potential vs distance  
  - Validates \(r^{-1}\) behaviour  

- `monopole_contour.png`  
  - 2D contour plot of potential in the midplane  

- `electric_field_vectors.png`  
  - Vector field visualisation of electric field  

---

### Poisson (Wire)
- `Magnetic_vs_distance.png`  
  - Log–log plot of magnetic field vs distance  
  - Validates \(r^{-1}\) scaling  

- `vector_potential_vs_distance.png`  
  - Semi-log plot of vector potential \(A_z\) vs distance  
  - Confirms logarithmic behaviour  

- `magnetic_potential_contour.png`  
  - Contour plot of vector potential  

- `magnetic_field_vectors.png`  
  - Vector field plot of magnetic field  

---

### SOR Analysis
- `sor_convergences_2.png`  
  - Number of iterations to convergence vs relaxation parameter \(\omega\)

---

## Data Files

Raw numerical outputs are saved for further analysis:

- `energy_phi*.dat`  
  - Free energy values over time (Cahn–Hilliard)

- `electric_potential_vs_distance.dat`  
  - Radial potential data  

- `vector_potential_vs_distance.dat`  
  - Radial vector potential data  

- `magnetic_strength_vs_distance.dat`  
  - Magnetic field magnitude vs radius  

- `vector_field_midplane.dat`  
  - Electric field components in midplane  

- `magnetic_field_midplane.dat`  
  - Magnetic field components in midplane  

- `potential_midplane.dat`  
  - Potential slice through system  

- `potential_midplane_wire.dat`  
  - Vector potential slice for wire case  

- `sor_convergences_2.dat`  
  - Convergence iterations for different \(\omega\)

---

# Key Insights

- The **Cahn–Hilliard solver** demonstrates:
  - Energy minimisation dynamics  
  - Emergence of phase-separated domains  
  - Sensitivity to timestep and grid resolution  

- The **Poisson solver** verifies fundamental physical laws:
  - Electric field: \(E \sim r^{-2}\)  
  - Electric potential: \(\Phi \sim r^{-1}\)  
  - Magnetic field (wire): \(B \sim r^{-1}\)  
  - Vector potential: \(A_z \sim \ln r\)

- **Successive Over-Relaxation (SOR)**:
  - Significantly accelerates convergence  
  - Requires careful tuning of \(\omega\)  
  - Optimal values typically lie near \(1.9\)–\(1.95\)

- Numerical considerations:
  - Boundary effects distort large-distance behaviour  
  - Singularities affect short-distance scaling  
  - Grid resolution impacts accuracy of gradients  

---

# Future Improvements

- Implement **multigrid solvers** for faster Poisson convergence  
- Add **implicit or semi-implicit schemes** for Cahn–Hilliard  
- Introduce **GPU acceleration** (e.g., CuPy or PyTorch)  
- Support **adaptive mesh refinement (AMR)**  
- Extend to **higher-order finite difference schemes**  
- Add automated benchmarking and error analysis tools  

---

# Author Notes

This project provides a practical framework for exploring:

- Numerical solutions to PDEs  
- Finite difference methods  
- Iterative linear solvers  
- Physical field theory simulations  

It is particularly useful for:
- Physics and applied mathematics students  
- Computational modelling projects  
- Validating analytical predictions through simulation  

The combination of **visual outputs** and **quantitative analysis** makes it well-suited for both learning and research applications.
