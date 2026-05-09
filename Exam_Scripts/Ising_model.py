"""
Ising_model.py

Two-dimensional Ising model Monte Carlo simulation with
Glauber and Kawasaki dynamics.
"""

# ============================================================
# INDEX
# ============================================================
# CLASS: IsingModel
#
#   SECTION: Initialisation
#     __init__(...)                 — store lattice, temperature sweep, dynamics,
#                                     coupling, field, and output settings
#     initialize_grid()             — return a random +/-1 spin lattice
#     pair_energy(...)              — return the bond energy between two spins
#     find_nearest_neighbors(...)   — return periodic nearest-neighbour coordinates
#     flip_probability(...)         — return the Metropolis acceptance factor
#
#   SECTION: Core dynamics
#     glauber_energy()              — propose a single-spin flip and return its
#                                     energy change
#     glauber_update()              — apply one Glauber spin-flip attempt
#     kawasaki_energy()             — propose a spin swap and return its
#                                     energy change
#     kawasaki_update()             — apply one Kawasaki exchange attempt
#
#   SECTION: Observables & statistics
#     bootstrap(...)                — estimate the bootstrap error on a sample mean
#     jackknife(...)                — estimate the jackknife error on a sample mean
#     determine_magnetisation()     — return total magnetisation and its square
#     average_magnetisation()       — sample magnetisation over a long run
#     magnetic_susceptibility(...)  — compute susceptibility from magnetisation
#                                     moments
#     total_energy()                — compute the total lattice interaction energy
#     heat_capacity_from_energies(...) — compute heat capacity from sampled energies
#     heat_capacity(...)            — compute heat capacity and its uncertainty
#
#   SECTION: Visualisation
#     animate()                     — animate lattice evolution at fixed temperature
#     plot_data(...)                — plot energy, heat capacity, and magnetic data
#     plot_stored_data()            — load stored CSV data and reproduce plots
#
#   SECTION: Exam extensions
#     staggered_magnetisation()     — compute the antiferromagnetic order parameter
#                                     for J < 0 exam variants
#     potts_delta_energy(...)       — compute Delta E for a q-state Potts update
#                                     to adapt Glauber logic in an exam
#
#   SECTION: I/O & data storage
#     run_data_collection()         — collect observables over a temperature sweep
#     store_data(...)               — save sweep results to CSV
#     run()                         — execute collection, plotting, and storage
#
# EXAM TOOLKIT (standalone functions below the class)
#   autocorrelation(...)            — compute normalised autocorrelation and
#                                     decorrelation time for sampled observables
#   bootstrap_error(...)            — bootstrap the error bar of any scalar
#                                     statistic from sampled data
#   gaussian_noise(...)             — generate Box-Muller Gaussian noise for
#                                     alternate exam initial conditions
#   survival_probability(...)       — estimate active-run survival curves for
#                                     absorbing-state exam variants
#
# CLI QUICK-REFERENCE
#   python Ising_model.py -N 50 --dynamic glauber --mode run --field 0.0
#   e.g. python Ising_model.py -N 50 --dynamic glauber --mode run
# ============================================================

import argparse
import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class IsingModel:
    """Two-dimensional Ising model with periodic boundary conditions."""

    # ── Initialisation ───────────────────────────────────────────

    def __init__(
        self,
        N,
        T,
        start_temp,
        end_temp,
        dynamic,
        uncertainty,
        J,
        save_fig,
        h=0.0,
    ):
        """Store simulation parameters and allocate model state.

        Args:
            N (int): Linear lattice size for an ``N x N`` spin grid.
            T (float): Temperature in units where ``k_B = 1``.
            start_temp (float): Initial temperature for a sweep.
            end_temp (float): Final temperature for a sweep.
            dynamic (str): Update rule, either ``"glauber"`` or ``"kawasaki"``.
            uncertainty (str): Error estimator, ``"bootstrap"`` or ``"jackknife"``.
            J (float): Nearest-neighbour coupling constant.
            save_fig (bool): Whether plots should be saved to disk.
            h (float): External magnetic field strength. Default 0.0.

        Returns:
            None: Initialises attributes in place.

        Notes:
            Physics/formula used: spins satisfy ``s_ij in {+1, -1}`` and evolve
            under the Ising Hamiltonian with nearest-neighbour interactions.
            ASSUMPTION: the class stores one lattice configuration at a time.
        """
        self.N = N
        self.T = T
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.dynamic = dynamic
        self.uncertainty = uncertainty
        self.J = J
        self.h = h
        self.save_fig = save_fig
        self.grid = None

    def pair_energy(self, i, j):
        """Return the pair interaction energy between two spins.

        Args:
            i (float): Spin value of the first site.
            j (float): Spin value of the second site.

        Returns:
            float: Pair energy ``-J s_i s_j``.

        Notes:
            Physics/formula used: nearest-neighbour bond energy
            ``E_ij = -J s_i s_j``.
            ASSUMPTION: both inputs are valid spin states or equivalent numeric
            values.
        """
        return -self.J * i * j  # Bond energy favours aligned spins when J > 0.

    def initialize_grid(self):
        """Generate a random initial ``N x N`` spin lattice.

        Args:
            None

        Returns:
            np.ndarray: Spin lattice with entries ``+1`` or ``-1``.

        Notes:
            Physics/formula used: each site starts in a random up/down state.
            ASSUMPTION: the initial state is an unbiased 50/50 mixture.
        """
        grid = np.random.rand(self.N, self.N)  # Uniform noise seeds the spin lattice.
        grid[grid >= 0.5] = 1  # Spins above the threshold start in the up state.
        grid[grid < 0.5] = -1  # Spins below the threshold start in the down state.
        return grid

    def flip_probability(self, delta_energy):
        """Return the Metropolis acceptance factor for a positive energy change.

        Args:
            delta_energy (float): Proposed energy increase ``Delta E``.

        Returns:
            float: Acceptance factor ``exp(-Delta E / T)``.

        Notes:
            Physics/formula used: Metropolis acceptance probability
            ``P = exp(-Delta E / T)`` for ``Delta E > 0``.
            ASSUMPTION: ``T > 0`` so the Boltzmann weight is well-defined.
        """
        P = math.exp(-delta_energy / self.T)  # Boltzmann suppression for uphill moves.
        return P

    def find_nearest_neighbors(self, i, j):
        """Return periodic nearest neighbours of lattice site ``(i, j)``.

        Args:
            i (int): Row index of the target site.
            j (int): Column index of the target site.

        Returns:
            list[tuple[int, int]]: Four periodic nearest-neighbour coordinates.

        Notes:
            Physics/formula used: square lattice with periodic boundaries.
            ASSUMPTION: interaction range is nearest neighbours only.
        """
        nearest_neighbors = [
            ((i + 1) % self.N, j),
            ((i - 1) % self.N, j),
            (i, (j + 1) % self.N),
            (i, (j - 1) % self.N),
        ]
        return nearest_neighbors

    # ── Core dynamics ────────────────────────────────────────────

    def glauber_energy(self):
        """Propose a single-spin flip and return its energy change.

        Args:
            None

        Returns:
            tuple: ``(delta_energy, (i, j))`` for the proposed flip site.

        Notes:
            Physics/formula used: ``Delta E = 2 J s_ij sum_nn s_nn + 2 h s_ij``.
            ASSUMPTION: the external field is spatially uniform.
        """
        i = np.random.randint(0, self.N)  # Random site selection samples all spins equally.
        j = np.random.randint(0, self.N)  # Random site selection samples all spins equally.
        nearest_neighbors = self.find_nearest_neighbors(i, j)
        delta_energy = (
            2
            * self.J
            * self.grid[i, j]
            * sum(self.grid[n] for n in nearest_neighbors)
            + 2 * self.h * self.grid[i, j]
        )  # external field term
        # EXTEND: replace self.h with h_fn(i, j, sweep) for spatially varying field
        return delta_energy, (i, j)

    def glauber_update(self):
        """Attempt one Glauber spin flip using Metropolis acceptance.

        Args:
            None

        Returns:
            None: Updates ``self.grid`` in place.

        Notes:
            Physics/formula used: accept all energy-lowering moves and otherwise
            accept with Boltzmann weight.
            ASSUMPTION: one call performs one attempted spin update.
        """
        delta_energy, (i, j) = self.glauber_energy()
        if delta_energy <= 0:
            self.grid[(i, j)] = -self.grid[(i, j)]  # Downhill flips are always accepted.
        elif self.flip_probability(delta_energy) > np.random.rand():
            self.grid[(i, j)] = -self.grid[(i, j)]  # Uphill flips follow Metropolis sampling.

    def kawasaki_energy(self):
        """Propose a spin swap and return its energy change.

        Args:
            None

        Returns:
            tuple: ``(delta_energy, (i_1, j_1), (i_2, j_2))`` for the swap.

        Notes:
            Physics/formula used: compare pre-swap and post-swap bond energies
            around the two selected sites.
            ASSUMPTION: magnetisation is conserved because only swaps are allowed.
        """
        i_1 = np.random.randint(0, self.N)
        j_1 = np.random.randint(0, self.N)
        i_2 = np.random.randint(0, self.N)
        j_2 = np.random.randint(0, self.N)

        energy = 0
        swap_energy = 0

        if self.grid[i_1, j_1] == self.grid[i_2, j_2]:
            return 0, (i_1, j_1), (i_2, j_2)

        nearest_neighbors_one = self.find_nearest_neighbors(i_1, j_1)
        nearest_neighbors_two = self.find_nearest_neighbors(i_2, j_2)

        if (i_2, j_2) not in nearest_neighbors_one:
            for neighbor in nearest_neighbors_one:
                energy += self.pair_energy(
                    self.grid[i_1, j_1], self.grid[neighbor]
                )  # Energy of bonds attached to the first spin before swap.
                swap_energy += self.pair_energy(
                    self.grid[i_2, j_2], self.grid[neighbor]
                )  # Energy if the second spin occupies the first site.

            for neighbor in nearest_neighbors_two:
                energy += self.pair_energy(
                    self.grid[i_2, j_2], self.grid[neighbor]
                )  # Energy of bonds attached to the second spin before swap.
                swap_energy += self.pair_energy(
                    self.grid[i_1, j_1], self.grid[neighbor]
                )  # Energy if the first spin occupies the second site.
        else:
            for neighbor in nearest_neighbors_one:
                energy += self.pair_energy(
                    self.grid[i_1, j_1], self.grid[neighbor]
                )  # Bond energy around the first site before swapping neighbours.
                temp_neighbor = (i_1, j_1) if neighbor == (i_2, j_2) else neighbor
                swap_energy += self.pair_energy(
                    self.grid[i_2, j_2], self.grid[temp_neighbor]
                )  # Uses swapped occupancy when the two selected sites are adjacent.

            for neighbor in nearest_neighbors_two:
                energy += self.pair_energy(
                    self.grid[i_2, j_2], self.grid[neighbor]
                )  # Bond energy around the second site before swapping neighbours.
                temp_neighbor = (i_2, j_2) if neighbor == (i_1, j_1) else neighbor
                swap_energy += self.pair_energy(
                    self.grid[i_1, j_1], self.grid[temp_neighbor]
                )  # Uses swapped occupancy when the two selected sites are adjacent.

        delta_energy = swap_energy - energy  # Net cost of exchanging the two spins.
        return delta_energy, (i_1, j_1), (i_2, j_2)

    def kawasaki_update(self):
        """Attempt one Kawasaki spin exchange using Metropolis acceptance.

        Args:
            None

        Returns:
            None: Updates ``self.grid`` in place.

        Notes:
            Physics/formula used: exchange two spins while conserving total
            magnetisation.
            ASSUMPTION: one call performs one attempted swap.
        """
        delta_energy, (i_1, j_1), (i_2, j_2) = self.kawasaki_energy()
        if delta_energy <= 0:
            self.grid[(i_1, j_1)], self.grid[(i_2, j_2)] = (
                self.grid[(i_2, j_2)],
                self.grid[(i_1, j_1)],
            )  # Downhill exchanges are always accepted.
        elif self.flip_probability(delta_energy) > np.random.rand():
            self.grid[(i_1, j_1)], self.grid[(i_2, j_2)] = (
                self.grid[(i_2, j_2)],
                self.grid[(i_1, j_1)],
            )  # Uphill exchanges are Boltzmann weighted.

    # ── Observables & statistics ─────────────────────────────────

    def bootstrap(self, data, num_samples=1000):
        """Estimate the bootstrap error on the sample mean.

        Args:
            data (array-like): Measurements to resample.
            num_samples (int): Number of bootstrap replicas. Default 1000.

        Returns:
            float: Bootstrap standard error of the mean.

        Notes:
            Physics/formula used: resample with replacement to estimate the
            spread of mean values.
            ASSUMPTION: the supplied measurements are representative samples.
        """
        n = len(data)
        means = []
        for _ in range(num_samples):
            sample = np.random.choice(data, size=n, replace=True)  # Bootstrap resample.
            means.append(np.mean(sample))  # Mean of one resampled observable set.
        return np.std(means)  # Spread of bootstrap means estimates the error bar.

    def jackknife(self, data):
        """Estimate the jackknife error on the sample mean.

        Args:
            data (array-like): Measurements to resample.

        Returns:
            float: Jackknife standard error of the mean.

        Notes:
            Physics/formula used: leave-one-out resampling with the standard
            jackknife variance formula.
            ASSUMPTION: the statistic of interest is the sample mean.
        """
        n = len(data)
        means = []
        for i in range(n):
            sample = np.delete(data, i)  # Leave-one-out sample for jackknife resampling.
            means.append(np.mean(sample))  # Mean of the reduced sample.
        means = np.array(means)
        mean_of_means = np.mean(means)  # Average jackknife estimate over all omissions.
        variance = (n - 1) / n * np.sum(
            (means - mean_of_means) ** 2
        )  # Standard jackknife variance estimator.
        return np.sqrt(variance)

    def determine_magnetisation(self):
        """Return the total magnetisation and its square.

        Args:
            None

        Returns:
            tuple: ``(M, M_squared)`` for the current lattice.

        Notes:
            Physics/formula used: ``M = sum_ij s_ij``.
            ASSUMPTION: magnetisation is reported as a total, not per spin.
        """
        M = np.sum(self.grid)  # Net spin sum is the order parameter for J > 0.
        M_squared = M**2  # Second moment enters susceptibility estimates.
        return M, M_squared

    def average_magnetisation(self):
        """Sample the average magnetisation over a long run.

        Args:
            None

        Returns:
            tuple: Average magnetisation and average squared magnetisation.

        Notes:
            Physics/formula used: performs 100 warm-up sweeps and 10000
            production sweeps with sampling every 10 sweeps.
            ASSUMPTION: this legacy routine is retained exactly for compatibility.
        """
        mags = []
        mags_squared = []
        for _ in range(100):
            for _ in range(self.N * self.N):
                if self.dynamic == "glauber":
                    self.glauber_update()
                elif self.dynamic == "kawasaki":
                    self.kawasaki_update()

        for j in range(10000):
            for _ in range(self.N * self.N):
                if self.dynamic == "glauber":
                    self.glauber_update()
                elif self.dynamic == "kawasaki":
                    self.kawasaki_update()

            if j % 10 == 0:
                M, M_squared = self.determine_magnetisation()  # Sample magnetisation every 10 sweeps.
                mags.append(M)
                mags_squared.append(M_squared)

        avg_mag = sum(mags) / len(mags)  # Mean magnetisation over the sampled run.
        avg_mag_squared = sum(mags_squared) / len(mags_squared)  # Mean square for fluctuations.
        return avg_mag, avg_mag_squared

    def magnetic_susceptibility(self, avg_mag, avg_mag_squared):
        """Compute the magnetic susceptibility from magnetisation moments.

        Args:
            avg_mag (float): Mean magnetisation.
            avg_mag_squared (float): Mean squared magnetisation.

        Returns:
            float: Magnetic susceptibility ``chi``.

        Notes:
            Physics/formula used: ``chi = (<M^2> - <M>^2) / (N^2 T)``.
            ASSUMPTION: ``T`` is measured in units with ``k_B = 1``.
        """
        chi = (avg_mag_squared - avg_mag**2) / (
            self.N**2 * self.T
        )  # Fluctuation-dissipation estimate of susceptibility.
        return chi

    def total_energy(self):
        """Compute the total interaction energy of the current lattice.

        Args:
            None

        Returns:
            float: Total nearest-neighbour interaction energy.

        Notes:
            Physics/formula used: sums ``-J s_i s_j`` over all bonds and divides
            by two to remove double counting.
            ASSUMPTION: this observable preserves the original code's interaction-
            only energy convention.
        """
        E = 0
        for i in range(self.N):
            for j in range(self.N):
                for ni, nj in self.find_nearest_neighbors(i, j):
                    E += -self.J * self.grid[i, j] * self.grid[
                        ni, nj
                    ]  # Add one bond contribution for each neighbour pair.
        return E / 2  # Remove double counting because each bond was visited twice.

    def heat_capacity_from_energies(self, energies):
        """Compute the heat capacity from a list of sampled energies.

        Args:
            energies (array-like): Sampled total energies.

        Returns:
            float: Heat capacity per spin from energy fluctuations.

        Notes:
            Physics/formula used: ``C = (<E^2> - <E>^2) / (N^2 T^2)``.
            ASSUMPTION: energies are sampled from equilibrium configurations.
        """
        E = np.array(energies)
        E_mean = np.mean(E)  # First energy moment for fluctuation calculations.
        E2_mean = np.mean(E**2)  # Second energy moment for fluctuation calculations.
        return (E2_mean - E_mean**2) / (
            self.N**2 * self.T**2
        )  # Heat capacity from energy variance.

    def heat_capacity(self, energies):
        """Compute heat capacity and its uncertainty.

        Args:
            energies (array-like): Sampled total energies.

        Returns:
            tuple: ``(C, C_error)`` for the current temperature.

        Notes:
            Physics/formula used: uses energy fluctuations for ``C`` and either
            bootstrap or jackknife resampling for the uncertainty.
            ASSUMPTION: the energy samples are sufficiently decorrelated.
        """
        C = self.heat_capacity_from_energies(energies)

        if self.uncertainty == "bootstrap":
            C_samples = []
            n = len(energies)
            for _ in range(1000):
                resample = np.random.choice(
                    energies, size=n, replace=True
                )  # Bootstrap resample of equilibrium energies.
                C_samples.append(
                    self.heat_capacity_from_energies(resample)
                )  # Derived heat capacity for one replica.
            C_error = np.std(C_samples)  # Spread of bootstrap heat capacities.

        elif self.uncertainty == "jackknife":
            C_samples = []
            n = len(energies)
            for i in range(n):
                resample = np.delete(
                    energies, i
                )  # Leave-one-out sample for jackknife error estimation.
                C_samples.append(
                    self.heat_capacity_from_energies(resample)
                )  # Derived heat capacity for one jackknife replica.
            C_samples = np.array(C_samples)
            C_mean = np.mean(C_samples)  # Mean jackknife estimate.
            C_error = np.sqrt(
                (n - 1) / n * np.sum((C_samples - C_mean) ** 2)
            )  # Standard jackknife variance formula.
        else:
            C_error = 0.0  # ASSUMPTION: parser choices prevent unsupported modes.

        return C, C_error

    # ── Visualisation ────────────────────────────────────────────

    def animate(self):
        """Animate lattice evolution at fixed temperature.

        Args:
            None

        Returns:
            None: Displays a Matplotlib animation.

        Notes:
            Physics/formula used: one frame corresponds to ``N^2`` attempted
            spin updates.
            ASSUMPTION: animation is for qualitative inspection, not data taking.
        """
        self.grid = self.initialize_grid()
        fig = plt.figure()
        im = plt.imshow(self.grid, animated=True, cmap="binary")

        def update_frame(_):
            for _ in range(self.N * self.N):
                if self.dynamic == "glauber":
                    self.glauber_update()
                elif self.dynamic == "kawasaki":
                    self.kawasaki_update()
            im.set_array(self.grid)  # Display the current spin configuration.
            return [im]

        ani = animation.FuncAnimation(
            fig,
            update_frame,
            frames=1000,
            interval=20,
            blit=True,
            repeat_delay=1000,
        )
        plt.show()

    def plot_data(
        self,
        total_mags,
        susceptibilities,
        energies,
        heat_capacities,
        heat_capacity_errors,
        temperatures,
    ):
        """Plot thermodynamic observables collected across a temperature sweep.

        Args:
            total_mags (list[float]): Average absolute magnetisations.
            susceptibilities (list[float]): Magnetic susceptibilities.
            energies (list[float]): Average total energies.
            heat_capacities (list[float]): Heat capacities.
            heat_capacity_errors (list[float]): Uncertainties on heat capacities.
            temperatures (list[float]): Simulated temperatures.

        Returns:
            None: Displays and optionally saves the figure.

        Notes:
            Physics/formula used: plots fluctuation observables versus temperature
            and marks the 2D Ising critical temperature for Glauber runs.
            ASSUMPTION: the supplied lists already correspond to equilibrium data.
        """
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "lines.linewidth": 2,
                "figure.dpi": 300,
            }
        )

        if self.dynamic == "glauber":
            fig, axes = plt.subplots(2, 2, figsize=(11, 9))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes = axes.flatten()

        axes[0].plot(temperatures, energies, marker="o", markersize=4)
        axes[0].set_xlabel(r"Temperature $T$")
        axes[0].set_ylabel(r"Total Energy $E$")
        axes[0].grid(alpha=0.3)
        axes[0].set_title("(a) Energy")

        axes[1].errorbar(
            temperatures,
            heat_capacities,
            yerr=heat_capacity_errors,
            fmt="o-",
            markersize=4,
            capsize=3,
        )
        axes[1].set_xlabel(r"Temperature $T$")
        axes[1].set_ylabel(r"Heat Capacity per Spin $C$")
        axes[1].grid(alpha=0.3)
        if self.dynamic == "glauber":
            axes[1].axvline(
                x=2.27,
                color="r",
                linestyle="--",
                label="Critical Temperature $T_c$",
            )  # Reference line marks the Onsager critical point.
        axes[1].set_title("(b) Heat capacity")

        if self.dynamic == "glauber":
            axes[2].plot(temperatures, total_mags, marker="o", markersize=4)
            axes[2].set_xlabel(r"Temperature $T$")
            axes[2].set_ylabel(r"Average Magnetisation $|M|$")
            axes[2].grid(alpha=0.3)
            axes[2].set_title("(c) Magnetisation")

            axes[3].plot(temperatures, susceptibilities, marker="o", markersize=4)
            axes[3].set_xlabel(r"Temperature $T$")
            axes[3].set_ylabel(r"Susceptibility $\chi$")
            axes[3].axvline(
                x=2.27,
                color="r",
                linestyle="--",
                label="Critical Temperature $T_c$",
            )  # Reference line marks the Onsager critical point.
            axes[3].grid(alpha=0.3)
            axes[3].set_title("(d) Susceptibility")

        plt.tight_layout()
        if self.save_fig:
            plt.savefig(
                f"ising_plots_{self.dynamic}_{self.uncertainty}_N{self.N}_T{self.start_temp}.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def plot_stored_data(self):
        """Load stored CSV data and reproduce the standard plots.

        Args:
            None

        Returns:
            None: Displays the reconstructed figure.

        Notes:
            Physics/formula used: visualises previously computed observables
            without rerunning the Monte Carlo simulation.
            ASSUMPTION: the CSV file format matches ``store_data``.
        """
        try:
            data = np.loadtxt(
                f"ising_data_{self.dynamic}_N{self.N}_T{self.start_temp}.csv",
                delimiter=",",
                skiprows=1,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"No stored data found for dynamic={self.dynamic}, N={self.N}"
            ) from exc

        temperatures = data[:, 0]
        total_mags = data[:, 1]
        susceptibilities = data[:, 2]
        energies = data[:, 3]
        heat_capacities = data[:, 4]
        heat_capacity_errors = data[:, 5]

        self.plot_data(
            total_mags,
            susceptibilities,
            energies,
            heat_capacities,
            heat_capacity_errors,
            temperatures,
        )

    # ── Exam extensions ──────────────────────────────────────────

    def staggered_magnetisation(self):  # EXAM ADDITION
        """Compute the staggered magnetisation for antiferromagnetic order.

        Args:
            None

        Returns:
            float: ``M_s = sum_ij (-1)^(i+j) s_ij``.

        Notes:
            Physics/formula used: the checkerboard factor distinguishes the two
            sublattices and makes antiferromagnetic order visible when ``J < 0``.
            ASSUMPTION: the lattice is bipartite, as on the square grid used here.
            EXAM: If the paper switches to an antiferromagnet, set ``model.J = -1``
            and replace magnetisation measurements with:
            ``Ms = model.staggered_magnetisation()``. Plot ``<|Ms|>`` versus ``T``
            instead of ``<|M|>`` to identify the ordering transition.
        """
        i_idx, j_idx = np.indices((self.N, self.N))
        staggered = ((-1) ** (i_idx + j_idx)) * self.grid  # Sublattice sign pattern for AF order.
        return float(np.sum(staggered))  # Net checkerboard-weighted spin sum.

    def potts_delta_energy(self, i, j, new_state):  # EXAM ADDITION
        """Compute the energy change for a Potts spin update at ``(i, j)``.

        Args:
            i (int): Row index of the spin to update.
            j (int): Column index of the spin to update.
            new_state (int): Proposed new Potts state.

        Returns:
            float: Change in energy if the site changes to ``new_state``.

        Notes:
            Physics/formula used: Potts Hamiltonian
            ``H = -J sum_<ij> delta(s_i, s_j)``.
            ASSUMPTION: the current lattice entry is already an integer Potts state.
            EXAM: To adapt Glauber dynamics to a q-state Potts model, propose
            ``new_state = np.random.randint(0, q)``, skip if it equals the old
            state, compute ``dE = model.potts_delta_energy(i, j, new_state)``,
            then apply the same Metropolis acceptance rule already used here.
        """
        old_state = self.grid[i, j]
        nearest_neighbors = self.find_nearest_neighbors(i, j)
        old_matches = sum(
            1 for neighbor in nearest_neighbors if self.grid[neighbor] == old_state
        )  # Bonds currently satisfied by the old Potts state.
        new_matches = sum(
            1 for neighbor in nearest_neighbors if self.grid[neighbor] == new_state
        )  # Bonds that would be satisfied after the proposed state change.
        delta_energy = -self.J * new_matches + self.J * old_matches  # Energy gain/loss from changed match count.
        return float(delta_energy)

    # ── I/O & data storage ───────────────────────────────────────

    def run_data_collection(self):
        """Run the simulation over a temperature sweep and collect observables.

        Args:
            None

        Returns:
            tuple: Magnetisation, susceptibility, energy, heat capacity,
            heat-capacity error, and temperature lists.

        Notes:
            Physics/formula used: reuses the current configuration between nearby
            temperatures to reduce equilibration cost.
            ASSUMPTION: the sweep direction is from ``start_temp`` down to
            ``end_temp`` exactly as in the original script.
        """
        temp_range = self.start_temp - self.end_temp
        step_size = -1 * temp_range / 20  # Original sweep resolution: 21 temperature points.

        self.grid = self.initialize_grid()

        total_average_mags = []
        susceptibilities = []
        avg_energies = []
        heat_capacities = []
        heat_capacity_errors = []
        temperatures = []

        self.T = 3.0
        for _ in range(4900):
            for _ in range(self.N * self.N):
                if self.dynamic == "glauber":
                    self.glauber_update()
                elif self.dynamic == "kawasaki":
                    self.kawasaki_update()

        for T in np.arange(
            self.start_temp,
            self.end_temp - 0.1 * temp_range,
            step_size,
        ):
            self.T = T
            print(f"Simulating at Temperature: {self.T} with dynamic: {self.dynamic}")
            temp_mags = []
            temp_mags_squared = []
            temp_energies = []

            for _ in range(100):
                for _ in range(self.N * self.N):
                    if self.dynamic == "glauber":
                        self.glauber_update()
                    elif self.dynamic == "kawasaki":
                        self.kawasaki_update()

            for i in range(10000):
                for _ in range(self.N * self.N):
                    if self.dynamic == "glauber":
                        self.glauber_update()
                    elif self.dynamic == "kawasaki":
                        self.kawasaki_update()
                if i % 10 == 0:
                    if self.dynamic == "glauber":
                        temp_mag, temp_mag_squared = self.determine_magnetisation()
                        temp_mags.append(temp_mag)  # Sample magnetisation every 10 sweeps.
                        temp_mags_squared.append(
                            temp_mag_squared
                        )  # Track squared magnetisation for susceptibility.
                    temp_energies.append(self.total_energy())  # Sample total energy every 10 sweeps.

            if self.dynamic == "glauber":
                avg_mag = np.mean(np.abs(temp_mags))  # Absolute magnetisation is the finite-size order parameter.
                avg_mag_squared = np.mean(
                    temp_mags_squared
                )  # Second moment enters the fluctuation formula.
                chi = self.magnetic_susceptibility(avg_mag, avg_mag_squared)
                total_average_mags.append(avg_mag)
                susceptibilities.append(chi)
            else:
                total_average_mags.append(np.nan)
                susceptibilities.append(np.nan)

            avg_energy = np.mean(temp_energies)  # Mean sampled total energy at this temperature.
            C, C_error = self.heat_capacity(temp_energies)
            avg_energies.append(avg_energy)
            heat_capacities.append(C)
            heat_capacity_errors.append(C_error)
            temperatures.append(self.T)

        return (
            total_average_mags,
            susceptibilities,
            avg_energies,
            heat_capacities,
            heat_capacity_errors,
            temperatures,
        )

    def store_data(
        self,
        total_mags,
        susceptibilities,
        energies,
        heat_capacities,
        heat_capacity_uncertainty,
        temperatures,
    ):
        """Store sweep data in a CSV file.

        Args:
            total_mags (list[float]): Average absolute magnetisations.
            susceptibilities (list[float]): Magnetic susceptibilities.
            energies (list[float]): Average total energies.
            heat_capacities (list[float]): Heat capacities.
            heat_capacity_uncertainty (list[float]): Heat-capacity uncertainties.
            temperatures (list[float]): Simulated temperatures.

        Returns:
            None: Saves data to disk.

        Notes:
            Physics/formula used: stores one row per temperature for later
            thermodynamic analysis.
            ASSUMPTION: the output filename encodes the dynamic, size, and
            starting temperature only.
        """
        data = np.array(
            [
                temperatures,
                total_mags,
                susceptibilities,
                energies,
                heat_capacities,
                heat_capacity_uncertainty,
            ]
        )
        np.savetxt(
            f"ising_data_{self.dynamic}_N{self.N}_T{self.start_temp}.csv",
            data.T,
            delimiter=",",
            header=(
                "Temperature,Average Magnetisation,Magnetic Susceptibility,"
                "Average Energy,Heat Capacity"
            ),
            comments="",
        )

    def run(self):
        """Execute the full simulation workflow.

        Args:
            None

        Returns:
            None: Collects, plots, and stores data.

        Notes:
            Physics/formula used: chains the existing production workflow
            without altering the underlying Monte Carlo procedure.
            ASSUMPTION: ``mode=run`` is used for temperature sweeps.
        """
        (
            total_mags,
            susceptibilities,
            energies,
            heat_capacities,
            heat_capacity_errors,
            temperatures,
        ) = self.run_data_collection()
        self.plot_data(
            total_mags,
            susceptibilities,
            energies,
            heat_capacities,
            heat_capacity_errors,
            temperatures,
        )
        self.store_data(
            total_mags,
            susceptibilities,
            energies,
            heat_capacities,
            heat_capacity_errors,
            temperatures,
        )


def autocorrelation(data):
    """Compute the normalised autocorrelation function and decorrelation time.

    Args:
        data (array-like): Time series of a scalar observable, e.g. a list
            of magnetisation or energy values sampled at regular intervals.

    Returns:
        tuple:
            ac (np.ndarray): Normalised autocorrelation, length ``len(data)``.
                ``ac[0] == 1`` by construction; decays toward 0 for large lag.
            tau (float): Integrated decorrelation time. Measurements separated
                by more than ``2*tau`` sweeps can be treated as independent.

    Notes:
        Formula: ``ac(t) = C(t)/C(0)`` where
        ``C(t) = <(m-)(m(t)-)>``.
        Integrated tau sums ``ac`` while it remains positive
        (Madras & Sokal 1988).

        EXAM: After your warm-up loop, collect about 1000 measurements into a
        list such as ``temp_mags``. Then run:
        ``ac, tau = autocorrelation(temp_mags)`` and
        ``print(f"Decorrelation time tau = {tau:.1f} sweeps")``.
        If ``tau`` is much larger than your current sampling gap, increase the
        gap to roughly ``int(2*tau)`` sweeps before quoting error bars.
    """
    data = np.array(data, dtype=float) - np.mean(data)  # Remove the mean before correlating fluctuations.
    ac = np.correlate(data, data, mode="full")[len(data) - 1 :]  # Positive-lag autocovariance sequence.
    ac = ac / ac[0]  # normalise so ac[0] = 1
    tau = 0.5 + np.sum(ac[1:][ac[1:] > 0])  # integrated autocorrelation time
    return ac, tau


def bootstrap_error(data, stat_fn, n_samples=1000):
    """Estimate the standard error of a scalar statistic by bootstrap resampling.

    Args:
        data (array-like): Raw measurements, e.g. a list of energies.
        stat_fn (callable): Function that takes an array and returns a scalar.
        n_samples (int): Number of bootstrap resamples. Default 1000.

    Returns:
        float: Bootstrap estimate of the standard error of ``stat_fn(data)``.

    Notes:
        EXAM: Use this whenever the question asks for an uncertainty on a
        derived quantity. For example, for a heat-capacity error use
        ``bootstrap_error(temp_energies, lambda e: (np.mean(e**2) - np.mean(e)**2) / (N**2 * T**2))``.
        Then pass the returned value to ``yerr`` in ``plt.errorbar`` and state
        clearly that the bar came from bootstrap resampling.
    """
    n = len(data)
    return np.std(
        [
            stat_fn(np.random.choice(data, size=n, replace=True))  # Bootstrap replica of the raw measurements.
            for _ in range(n_samples)
        ]
    )


def gaussian_noise(shape, sigma=1.0):
    """Generate Gaussian noise with the Box-Muller transform.

    Args:
        shape (tuple): Shape of the output array, e.g. ``(N, N)``.
        sigma (float): Standard deviation of the Gaussian. Default 1.0.

    Returns:
        np.ndarray: Gaussian random array with variance ``sigma^2``.

    Notes:
        Converts two uniform random arrays into Gaussian noise using
        ``z = sigma * sqrt(-2 ln(u1)) * cos(2 pi u2)``.

        EXAM: If the paper explicitly asks for Box-Muller noise, replace the
        random-uniform initialisation line with
        ``noise = gaussian_noise((self.N, self.N), sigma=0.01)`` and add it to
        the homogeneous starting field or spin seed as required.
    """
    u1 = np.random.rand(*shape)  # Uniform variates provide the Box-Muller radius term.
    u2 = np.random.rand(*shape)  # Uniform variates provide the Box-Muller angle term.
    return sigma * np.sqrt(-2.0 * np.log(u1)) * np.cos(
        2.0 * np.pi * u2
    )  # Box-Muller


def survival_probability(step_fn, is_active_fn, n_runs=200, max_steps=500):
    """Estimate the fraction of runs still active at each time step.

    Args:
        step_fn (callable): Advances the model one sweep in-place. No args.
        is_active_fn (callable): Returns ``True`` if any active sites remain.
        n_runs (int): Number of independent runs from a single active seed.
        max_steps (int): Maximum number of sweeps per run.

    Returns:
        np.ndarray: Survival fraction at each sweep.

    Notes:
        Standard method for characterising absorbing-state phase transitions.

        EXAM: If the question asks for survival curves, create a fresh model for
        each parameter set, define or add a function that detects whether the
        active phase still exists, then call
        ``P = survival_probability(model.sweep, model.has_infected)``.
        Plot several ``P(t)`` curves together and identify the parameter where
        decay is closest to a power law.
    """
    alive = np.ones(max_steps, dtype=float)
    for _ in range(n_runs):
        for t in range(max_steps):
            step_fn()
            if not is_active_fn():
                alive[t:] -= 1.0 / n_runs  # this run died at step t
                break
    return alive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo simulation of the 2D Ising model with periodic boundary "
            "conditions.\n\nSupports Glauber (non-conserved magnetisation) and "
            "Kawasaki (conserved magnetisation) dynamics, with thermodynamic "
            "observables computed via Metropolis sampling."
        ),
        add_help=False,
    )
    parser.add_argument(
        "--help",
        action="help",
        help="Show this help message and exit.",
    )

    parser.add_argument(
        "-N",
        "--size",
        type=int,
        default=50,
        metavar="N",
        help="Linear lattice size. The system contains N×N spins.",
    )
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=2.5,
        metavar="T",
        help=(
            "Temperature of the system (Units of K_B T). "
            "For temperature sweeps, this value is used as the initial temperature."
        ),
    )
    parser.add_argument(
        "--start_temp",
        "-tmax",
        type=float,
        default=3.0,
        metavar="T_MAX",
        help="Starting temperature for temperature sweeps (default: 3.0), greater than end temp",
    )
    parser.add_argument(
        "--end_temp",
        "-tmin",
        type=float,
        default=1.0,
        metavar="T_MIN",
        help="Ending temperature for temperature sweeps (default: 1.0), less than start temp",
    )
    parser.add_argument(
        "-J",
        "--coupling",
        type=float,
        default=1.0,
        metavar="J",
        help=(
            "Nearest-neighbour coupling constant. "
            "Positive J corresponds to the ferromagnetic Ising model."
        ),
    )
    parser.add_argument(
        "-h",
        "--field",
        type=float,
        default=0.0,
        help="External magnetic field",
    )
    parser.add_argument(
        "--dynamic",
        choices=["glauber", "kawasaki"],
        default="glauber",
        help=(
            "Choice of Monte Carlo dynamics:\n"
            "  glauber  – single-spin flips (magnetisation not conserved)\n"
            "  kawasaki – spin exchanges (magnetisation conserved)"
        ),
    )
    parser.add_argument(
        "--uncertainty",
        choices=["bootstrap", "jackknife"],
        default="bootstrap",
        help=(
            "Statistical method used to estimate uncertainties "
            "in the heat capacity."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["run", "animate", "plot"],
        default="animate",
        help=(
            "Execution mode:\n"
            "  run     – run simulation, collect data, plot and store results\n"
            "  animate – animate lattice evolution at fixed temperature\n"
            "  plot    – plot previously stored data from CSV files"
        ),
    )
    parser.add_argument(
        "--save_fig",
        action="store_true",
        help="Save generated plots as high-resolution PNG files.",
    )

    args = parser.parse_args()

    model = IsingModel(
        N=args.size,
        T=args.temperature,
        start_temp=args.start_temp,
        end_temp=args.end_temp,
        dynamic=args.dynamic,
        uncertainty=args.uncertainty,
        J=args.coupling,
        save_fig=args.save_fig,
        h=args.field,
    )

    if args.mode == "run":
        model.run()
    elif args.mode == "animate":
        model.animate()
    elif args.mode == "plot":
        model.plot_stored_data()
