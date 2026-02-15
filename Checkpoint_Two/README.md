
# Game of Life and SIRS Model Simulation

This repository contains implementations of **Conway's Game of Life** and the **SIRS (Susceptible-Infected-Recovered-Susceptible) epidemiological model**, written in Python.  
The code supports visualization and analysis of emergent patterns and disease dynamics, with utilities for computing statistical observables including infection rates, recovery statistics, and spatial correlation analysis.

---

## Model Definitions

### Game of Life

A cellular automaton on an \(N \times N\) grid with periodic boundary conditions.

#### States
- Cell states: \(\sigma_i \in \{\text{alive}, \text{dead}\}\)

#### Rules
- A live cell survives if it has 2–3 live neighbours
- A dead cell becomes alive if it has exactly 3 live neighbours
- All other cells die or remain dead

#### Update Rule
Synchronous updates applied to all cells each generation.

### SIRS Model

A stochastic epidemiological model on an \(N \times N\) lattice.

#### States
- Individual states: \(s_i \in \{\text{S}, \text{I}, \text{R}\}\)  
    (Susceptible, Infected, Recovered)

#### Transition Rates
- Infection: \(\text{S} \to \text{I}\) with probability proportional to infected neighbours
- Recovery: \(\text{I} \to \text{R}\) with rate \(\gamma\)
- Loss of immunity: \(\text{R} \to \text{S}\) with rate \(\nu\)

---

## Measured Observables

### Game of Life
- Population density (fraction of alive cells)
- Still-life and oscillator detection
- Pattern classification

### SIRS Model
- Infected fraction: \(I(t) = \frac{1}{N^2}\sum_i \mathbb{1}[s_i = I]\)
- Susceptible fraction: \(S(t)\)
- Time-averaged infection rate during stationary state

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

### Running Simulations

```bash
python game_of_life.py [OPTIONS]
python sirs_model.py [OPTIONS]
```

#### Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--lattice-size` | `-N` | int | 50 | Grid size (N × N) |
| `--steps` | `-S` | int | 1000 | Number of simulation steps |
| `--infection-rate` | `-p` | float | 0.3 | Infection probability per neighbour (SIRS) |
| `--recovery-rate` | `-r` | float | 0.5 | Recovery rate (SIRS) |
| `--immunity-loss` | `-nu` | float | 0.1 | Immunity loss rate (SIRS) |
| `--function` | — | str | `animate` | `animate` or `plot` |
| `--save_fig` | — | flag | False | Save generated plots |

#### Examples

Animate Game of Life with default parameters:
```bash
python game_of_life.py --function animate
```

Run SIRS model with custom infection dynamics and save results:
```bash
python sirs_model.py -N 100 -S 5000 -p 0.2 -r 0.3 --save_fig
```
