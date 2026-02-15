# Two-Dimensional Ising Model Simulation

This repository contains a Monte Carlo simulation of the **two-dimensional Ising model** with periodic boundary conditions, implemented in Python.  
The code supports both **Glauber** and **Kawasaki** dynamics and computes thermodynamic observables including energy, heat capacity, magnetisation, and magnetic susceptibility, with uncertainty estimation via **bootstrap** or **jackknife resampling**.

---

## Model Definition

We consider the standard 2D Ising model on an \(N \times N\) square lattice with periodic boundary conditions.

### Spins
- Spin variables:  
    \[
    s_i \in \{+1, -1\}
    \]

### Hamiltonian
\[
H = -J \sum_{\langle i j \rangle} s_i s_j
\]
where the sum runs over nearest-neighbour pairs.

### Units and Conventions
- Boltzmann constant: \(k_B = 1\)
- Temperature \(T\) is in units of \(k_b *T\)
- Energies returned by the code are **total energies**, unless explicitly normalised
- Magnetisation:
    \[
    M = \sum_i s_i
    \]

---

## Dynamics

The simulation supports two types of Monte Carlo dynamics:

### Glauber Dynamics
- Single-spin flip updates
- Magnetisation is **not conserved**
- Suitable for studying:
    - Energy
    - Heat Capacity
    - Magnetisation
    - Magnetic susceptibility
    - Critical behaviour

### Kawasaki Dynamics
- Spin-exchange updates
- Magnetisation is **conserved**
- Magnetisation and susceptibility are therefore **not defined**
- Suitable for studying:
    - Energy
    - Heat capacity

Updates are performed using the **Metropolis algorithm**.

---

## Measured Observables

At each temperature, the following quantities are computed:

### Energy
- Total energy \(E\)

### Heat Capacity
Computed from energy fluctuations:
\[
C = \frac{\langle E^2 \rangle - \langle E \rangle^2}{N^2 T^2}
\]

Uncertainty is estimated using:
- **Bootstrap resampling**, or
- **Jackknife resampling**

### Magnetisation (Glauber only)
\[
\langle |M| \rangle
\]

### Magnetic Susceptibility (Glauber only)
\[
\chi = \frac{\langle M^2 \rangle - \langle M \rangle^2}{N^2 T}
\]

---

## Simulation Protocol

For each temperature:
1. The system is equilibrated for a fixed number of Monte Carlo sweeps
2. Measurements are taken every fixed number of sweeps to reduce autocorrelation
3. Observables are averaged over samples
4. Heat capacity uncertainties are estimated from the full energy time series

The temperature is scanned **from high to low** to reduce critical slowing-down effects.

---

## Usage

### Requirements
- Python ≥ 3.8
- NumPy
- Matplotlib

Install dependencies via:
```bash
pip install numpy matplotlib
```

### Running the Simulation

Execute the script with command-line arguments to configure the simulation:

```bash
python ising_model.py [OPTIONS]
```

#### Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--lattice-size` | `-N` | int | 50 | Size of the square lattice (N × N) |
| `--temperature` | `-T` | float | 2.5 | Temperature of the system (Units of K_b T) |
| `--dynamic` | — | str | `glauber` | Dynamics type: `glauber` or `kawasaki` |
| `--uncertainty` | — | str | `bootstrap` | Uncertainty method: `bootstrap` or `jackknife` |
| `--interaction` | `-J` | float | 1.0 | Interaction energy between neighbouring spins |
| `--function` | — | str | `animate` | Function to execute: `run`, `animate`, or `plot` |
| `--save_fig` | — | flag | False | Save generated plots as PNG files |

#### Examples

Run a standard simulation with default parameters:
```bash
python ising_model.py
```

Simulate a 100×100 lattice at T=2.0 with Kawasaki dynamics:
```bash
python ising_model.py -N 100 -T 2.0 --dynamic kawasaki
```

Generate plots for default size and temperature and save figures:
```bash
python ising_model.py --function plot --save_fig
```

Run data collection for glauber dynamics with Jackknife uncertainty and save plots:
```bash
python ising_model.py --function run --uncertainty jackknife --dynamic glauber --save_fig
```

