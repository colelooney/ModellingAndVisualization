
# Game of Life and SIRS Model Simulation

This repository contains implementations of **Conway's Game of Life** and the **SIRS (Susceptible-Infected-Recovered-Susceptible) epidemiological model**, written in Python.  
The code supports visualization and analysis of emergent patterns and disease dynamics, with utilities for computing statistical observables

---

## Model Definitions

### Game of Life

A cellular automaton on an $\(N \times N\)$ grid with periodic boundary conditions.

#### States
- Cell states: $\(\sigma_i \in \{\text{alive}, \text{dead}\}\)$

#### Rules
- A live cell survives if it has 2–3 live neighbours
- A dead cell becomes alive if it has exactly 3 live neighbours
- All other cells die or remain dead

#### Update Rule
Synchronous updates applied to all cells each generation.

### SIRS Model

A stochastic epidemiological model on an $\(N \times N\)$ lattice.

#### States
- Individual states: $\(s_i \in \{\text{S}, \text{I}, \text{R}, {f_{im}}\}\)$  
    (Susceptible, Infected, Recovered, Permanently Immune (tunable parameter))

#### Transition Rates
- Infection: $\(\text{S} \to \text{I}\)$ with probability $\(P_{S}\)$ if at least one neighbor infected
- Recovery: $\(\text{I} \to \text{R}\)$ with probability $\(P_{I}\)$
- Loss of immunity: $\(\text{R} \to \text{S}\)$ with probability $\(P_{R}\)$

---

## Measured Observables

### Game of Life
- Time to Equilibrium
- Glider Speeds

### SIRS Model
- Infected fraction at equilibrium: $\(I(t) = \frac{1}{N^2}\sum_i \mathbb{1}[s_i = I]\)$
- Average Infected Fraction against permanent immunity
- Variance of infected cells at equilibrium

---

## Usage

### Requirements
- Python ≥ 3.8
- NumPy
- Matplotlib
- argparse
- collections
- scipy
- numba (if running SIRS model in it's script)

Install dependencies via:
```bash
pip install requirements.txt
```

### Running Simulations

```bash
python main.py [OPTIONS]
```

### Running individual simulations
```bash
python GameOfLife.py  [OPTIONS]
python SIRS.py [OPTIONS]
```
#### Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--model` | | str | None | SIRS or Game of Life Model|
| `--size` | `-N` | int | 50 | Grid size (N × N) |
| `--num_runs` | | int | 1000 | Number of simulation steps |
|`--initial_state`| | str | random | initial state for Game of Life  (GOL)|
|`--alive_fraction`| `F` | float | 0.5 | fraction of initially alive cells (GOL) | 
| `--infection_prob` | `S` | float | 0.5 | Infection probability per neighbour (SIRS) |
| `--recovery_prob` | `-I` | float | 0.5 | Recovery rate (SIRS) |
| `--resusceptibility_prob` | `-R` | float | 0.5 | Immunity loss rate (SIRS) |
| `--immune_fraction`| `-f` | float | 0.0 | percent of permanently immune cells (SIRS)|
| `--animate`|| action | None | Animate evolution |
| `--debug`|| action | None | Run additional debug functions |
| `--run_variance| | action | None | Run Variance Data Collection (SIRS) |
| `--run_immunity| | action | None | Run Immunity Data Collection (SIRS) |

#### Examples

Animate Game of Life with default parameters:
```bash
python main.py --model game_of_life --function animate
```

Run SIRS model with custom infection dynamics and determine average immunity:
```bash
python main.py --model sirs --run_variance -N 100 
```
