"""
simulate the SIRS Cellular Automaton model on a 2D grid with periodic boundary
conditions.
"""

# ============================================================
# INDEX
# ============================================================
# CLASS: SIRS
#
#   SECTION: Initialisation
#     __init__(...)                 — store epidemiological probabilities and
#                                     initialise the lattice
#     initialize_grid()             — return a random susceptible/infected/immune grid
#     find_nearest_neighbors(...)   — return the set of neighbour states around one site
#
#   SECTION: Core dynamics
#     update_cell(...)              — apply the SIRS rule to one randomly chosen cell
#     sweep()                       — perform one Monte Carlo sweep of random updates
#
#   SECTION: Observables & statistics
#     count_infected()              — count infected sites in the current lattice
#     majority_fraction()           — return the dominant state and its lattice share
#
#   SECTION: Visualisation
#     animate()                     — animate the lattice evolution
#     plot_single_frame()           — display one initial configuration
#
#   SECTION: Exam extensions
#     mutation_update(...)          — inject spontaneous random-state mutations
#                                     before the standard epidemiological sweep
#     majority_fraction()           — monitor which state dominates the lattice
#                                     in extended three-state CA exam questions
#
#   SECTION: I/O & data storage
#     run()                         — scan infection/resusceptibility space and
#                                     save the equilibrium heatmap
#     run_variance()                — measure infected-fraction variance versus S
#     run_immunity()                — measure mean infected fraction versus immune fraction
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
#   python SIRS.py -N 50 -S 0.5 -I 0.5 -R 0.5
#   e.g. python SIRS.py --run_variance --p_mut 0.01
# ============================================================

import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numba import njit


@njit
def find_nearest_neighbors(grid, i, j):
    """Return the set of nearest-neighbour states around ``(i, j)``."""
    nearest_neighbors_coordinates = set(
        [
            ((i + 1) % grid.shape[0], j),
            ((i - 1) % grid.shape[0], j),
            (i, (j + 1) % grid.shape[1]),
            (i, (j - 1) % grid.shape[1]),
        ]
    )
    nearest_neighbors = set()
    for coord in nearest_neighbors_coordinates:
        nearest_neighbors.add(grid[coord])
    return nearest_neighbors


@njit
def update_cell(grid, x, y, S, I, R):
    """Apply one local SIRS update to site ``(x, y)``."""
    current_state = grid[x, y]
    if current_state == 0:
        neighbor_states = find_nearest_neighbors(grid, x, y)
        if 1 in neighbor_states:
            grid[x, y] = 1 if np.random.rand() <= S else 0
    elif current_state == 1:
        grid[x, y] = 2 if np.random.rand() <= I else 1
    elif current_state == 2:
        grid[x, y] = 0 if np.random.rand() <= R else 2


@njit
def sweep(grid, S, I, R):
    """Perform one accelerated SIRS sweep with random sequential updates."""
    for _ in range(grid.shape[0] ** 2):
        i = np.random.randint(0, grid.shape[0])  # Random sequential update samples one lattice site.
        j = np.random.randint(0, grid.shape[1])  # Random sequential update samples one lattice site.
        update_cell(grid, i, j, S, I, R)


class SIRS:
    """SIRS cellular automaton with optional permanent immunity."""

    # ── Initialisation ───────────────────────────────────────────

    def __init__(self, N, debug, num_runs, S, I, R, f, resolution, p_mut=0.0):
        """Store SIRS parameters and initialise the lattice.

        Args:
            N (int): Linear lattice size for an ``N x N`` system.
            debug (bool): Whether to print diagnostic information.
            num_runs (int): Number of measurement sweeps or runs.
            S (float): Probability that a susceptible site becomes infected.
            I (float): Probability that an infected site recovers.
            R (float): Probability that a recovered site becomes susceptible.
            f (float): Fraction of permanently immune sites in the initial state.
            resolution (float): Step size used in parameter scans.
            p_mut (float): Spontaneous mutation probability. Default 0.0.

        Returns:
            None: Initialises attributes in place.

        Notes:
            Physics/formula used: states are 0 = susceptible, 1 = infected,
            2 = recovered, 3 = permanently immune.
            ASSUMPTION: permanent immunity enters only through the initial state.
        """
        self.N = N
        self.grid = None
        self.debug = debug
        self.num_runs = num_runs
        self.S = S
        self.I = I
        self.R = R
        self.f = f
        self.resolution = resolution
        self.p_mut = p_mut
        if self.debug:
            print(
                f"Initialized SIRS Model with N={self.N}, S={self.S}, "
                f"I={self.I}, R={self.R}, f={self.f}, p_mut={self.p_mut}"
            )

        self.grid = self.initialize_grid()

    def initialize_grid(self):
        """Generate the initial epidemic state of the lattice.

        Args:
            None

        Returns:
            np.ndarray: ``N x N`` array with states 0, 1, or 3 initially.

        Notes:
            Physics/formula used: recovered sites are not present initially; the
            lattice is seeded with susceptible, infected, and permanently immune
            sites only.
            ASSUMPTION: susceptible and infected shares split the non-immune
            fraction equally.
        """
        grid = np.random.choice(
            [0, 1, 3],
            size=(self.N, self.N),
            p=[0.5 - self.f / 2, 0.5 - self.f / 2, self.f],
        )  # Initial occupancy seeds susceptible, infected, and immune populations.
        return grid

    def find_nearest_neighbors(self, i, j):
        """Return the set of neighbour states around lattice site ``(i, j)``.

        Args:
            i (int): Row index of the target site.
            j (int): Column index of the target site.

        Returns:
            set[int]: Distinct states present among the four nearest neighbours.

        Notes:
            Physics/formula used: infection depends only on whether at least one
            infected neighbour exists, so a set is sufficient.
            ASSUMPTION: only nearest-neighbour contacts transmit infection.
        """
        nearest_neighbors_coordinates = set(
            [
                ((i + 1) % self.N, j),
                ((i - 1) % self.N, j),
                (i, (j + 1) % self.N),
                (i, (j - 1) % self.N),
            ]
        )
        nearest_neighbors = set()
        for coord in nearest_neighbors_coordinates:
            nearest_neighbors.add(self.grid[coord])  # Distinct neighbour states determine local transition options.
        if self.debug:
            print(f"nearest neighbors states for cell ({i},{j}): {nearest_neighbors}")
        return nearest_neighbors

    # ── Core dynamics ────────────────────────────────────────────

    def update_cell(self, x, y):
        """Apply the SIRS rule to one lattice site.

        Args:
            x (int): Row index of the selected cell.
            y (int): Column index of the selected cell.

        Returns:
            None: Modifies ``self.grid`` in place.

        Notes:
            Physics/formula used: susceptible sites infect if they touch at least
            one infected neighbour, infected sites recover, recovered sites lose
            immunity, all probabilistically.
            ASSUMPTION: permanently immune state 3 never changes.
        """
        current_state = self.grid[x, y]
        if current_state == 0:
            neighbor_states = self.find_nearest_neighbors(x, y)
            if 1 in neighbor_states:
                self.grid[x, y] = (
                    1 if np.random.rand() <= self.S else 0
                )  # Infection occurs only when an infected neighbour is present.
        elif current_state == 1:
            self.grid[x, y] = (
                2 if np.random.rand() <= self.I else 1
            )  # Infected sites recover with probability I.
        elif current_state == 2:
            self.grid[x, y] = (
                0 if np.random.rand() <= self.R else 2
            )  # Recovered sites become susceptible again with probability R.

    def sweep(self):
        """Perform one Monte Carlo sweep of random sequential SIRS updates.

        Args:
            None

        Returns:
            None: Updates ``self.grid`` in place.

        Notes:
            Physics/formula used: one sweep performs ``N^2`` random site updates.
            ASSUMPTION: random sequential updating approximates asynchronous
            epidemic dynamics.
        """
        for _ in range(self.N**2):
            i = np.random.randint(0, self.N)  # Random site selection avoids directional bias.
            j = np.random.randint(0, self.N)  # Random site selection avoids directional bias.
            self.update_cell(i, j)

    # ── Observables & statistics ─────────────────────────────────

    def count_infected(self):
        """Count infected cells in the current lattice.

        Args:
            None

        Returns:
            int: Number of sites currently in state 1.

        Notes:
            Physics/formula used: the infected fraction is the main order
            parameter for sustained epidemic activity.
            ASSUMPTION: only state 1 counts as infected.
        """
        return int(np.sum(self.grid == 1))  # Infected-site count measures current epidemic activity.

    # ── Visualisation ────────────────────────────────────────────

    def animate(self):
        """Animate the SIRS lattice evolution.

        Args:
            None

        Returns:
            None: Displays a Matplotlib animation.

        Notes:
            Physics/formula used: colours represent susceptible, infected,
            recovered, and permanently immune states.
            ASSUMPTION: if ``p_mut > 0`` the mutation step precedes each sweep.
        """
        self.grid = self.initialize_grid()
        fig = plt.figure()
        cmap = ListedColormap(["white", "red", "blue", "black"])
        im = plt.imshow(self.grid, animated=True, cmap=cmap, vmin=0, vmax=3)

        def update_frame(_):
            if self.p_mut > 0:
                self.mutation_update(p_mut=self.p_mut)  # EXAM ADDITION
            self.sweep()
            im.set_array(self.grid)
            return [im]

        ani=animation.FuncAnimation(
            fig,
            update_frame,
            frames=1000,
            interval=20,
            blit=True,
            repeat_delay=1000,
        )
        plt.show()

    def plot_single_frame(self):
        """Display one initial lattice configuration.

        Args:
            None

        Returns:
            None: Shows the grid as an image.

        Notes:
            Physics/formula used: visualises the starting epidemiological state.
            ASSUMPTION: used mainly for debugging.
        """
        self.grid = self.initialize_grid()
        plt.imshow(self.grid, cmap="binary")
        plt.title("Initial State of Game of Life")
        plt.axis("off")
        plt.show()

    # ── Exam extensions ──────────────────────────────────────────

    def mutation_update(self, p_mut=0.01):  # EXAM ADDITION
        """Apply spontaneous random state mutation before the normal sweep.

        Args:
            p_mut (float): Probability that a cell jumps to a random state.

        Returns:
            None: Modifies ``self.grid`` in place.

        Notes:
            Physics/formula used: mutation adds external noise by replacing
            chosen sites with a uniformly random state from ``{0, 1, 2}``.
            ASSUMPTION: permanently immune sites are not exempt from mutation in
            this simple extension unless you add a mask.
            EXAM: If the paper adds a random-reset probability, call
            ``self.mutation_update(p_mut=args.p_mut)`` immediately before the
            usual ``self.sweep()`` inside your loop. Then compare the infected
            fraction with and without mutation to show how noise suppresses the
            absorbing phase.
        """
        mutation_mask = np.random.rand(self.N, self.N) < p_mut  # Bernoulli mask selects spontaneously mutating sites.
        n_mut = np.count_nonzero(mutation_mask)
        if n_mut > 0:
            self.grid[mutation_mask] = np.random.randint(
                0, 3, size=n_mut
            )  # Randomly reset selected sites to susceptible, infected, or recovered.

    def majority_fraction(self):  # EXAM ADDITION
        """Return the dominant state and its fraction of the lattice.

        Args:
            None

        Returns:
            tuple: ``(dominant, fraction)`` for the most common lattice state.

        Notes:
            Physics/formula used: counts occupancy of states 0, 1, 2, and 3 and
            reports the largest share as a coarse ordering measure.
            ASSUMPTION: permanently immune state 3 is included in the tally.
            EXAM: If you are asked which state dominates, call
            ``dominant, fraction = self.majority_fraction()`` each sweep and plot
            ``fraction`` versus time. A value near 1 means one epidemiological
            state has almost completely taken over the lattice.
        """
        counts = np.bincount(self.grid.astype(int).ravel(), minlength=4)  # State histogram measures phase occupancy.
        dominant = int(np.argmax(counts))  # Most populated state defines the majority phase.
        fraction = float(counts[dominant] / self.grid.size)  # Largest occupancy fraction normalises by lattice size.
        return dominant, fraction

    # ── I/O & data storage ───────────────────────────────────────

    def run(self):
        """Scan ``S`` and ``R`` values and save the equilibrium infection heatmap.

        Args:
            None

        Returns:
            None: Displays and stores the heatmap data.

        Notes:
            Physics/formula used: the equilibrium infected fraction acts as the
            order parameter across the parameter plane.
            ASSUMPTION: ``I`` is fixed to 0.5 exactly as in the original script.
        """
        self.I = 0.5
        res_steps = len(np.arange(0, 1 + self.resolution, self.resolution))
        heatmap_data = np.zeros((res_steps, res_steps))

        for i, p_s in enumerate(np.arange(0, 1 + self.resolution, self.resolution)):
            for j, p_r in enumerate(np.arange(0, 1 + self.resolution, self.resolution)):
                self.S = p_s
                self.R = p_r
                self.grid = self.initialize_grid()
                infected_count = []
                for _ in range(100):
                    if self.p_mut > 0:
                        self.mutation_update(p_mut=self.p_mut)  # EXAM ADDITION
                    sweep(
                        self.grid, self.S, self.I, self.R
                    )  # Accelerated equilibration sweep for the chosen parameters.
                for _ in range(self.num_runs):
                    if self.p_mut > 0:
                        self.mutation_update(p_mut=self.p_mut)  # EXAM ADDITION
                    sweep(
                        self.grid, self.S, self.I, self.R
                    )  # Accelerated measurement sweep for the chosen parameters.
                    infected_count.append(self.count_infected())
                heatmap_data[j, i] = np.mean(infected_count) / (
                    self.N**2
                )  # Mean infected fraction defines the phase-diagram colour scale.

        plt.figure(figsize=(10, 6))
        plt.heatmap = plt.imshow(
            heatmap_data, extent=(0, 1, 0, 1), origin="lower", cmap="viridis"
        )
        plt.colorbar(
            plt.heatmap, label="Average Fraction of Infected Cells at Equilibrium"
        )
        plt.xlabel("Probability of Infection (S)")
        plt.ylabel("Probability of Resusceptibility (R)")
        plt.title("SIRS Model: Average Fraction of Infected Cells at Equilibrium")
        plt.savefig(f"{self.resolution}_sirs_equilibrium_heatmap.png")
        plt.show()

        np.savez(
            f"{self.resolution}_sirs_infection_heatmap_data.npz",
            heatmap_data=heatmap_data,
            S_vales=np.arange(0, 1 + self.resolution, self.resolution),
            R_values=np.arange(0, 1 + self.resolution, self.resolution),
        )

    def run_variance(self):
        """Measure infected-fraction variance versus infection probability.

        Args:
            None

        Returns:
            None: Displays and stores the variance curve.

        Notes:
            Physics/formula used: variance
            ``(<I^2> - <I>^2) / N^2`` peaks near a phase transition.
            ASSUMPTION: ``I = R = 0.5`` are fixed exactly as in the original code.
        """
        self.I = 0.5
        self.R = 0.5

        variance = []
        variance_errors = []

        for p_s in np.arange(0.2, 0.5 + self.resolution, self.resolution):
            self.S = p_s
            self.grid = self.initialize_grid()
            infected_count = []
            infected_squared_count = []
            for _ in range(100):
                if self.p_mut > 0:
                    self.mutation_update(p_mut=self.p_mut)  # EXAM ADDITION
                sweep(self.grid, self.S, self.I, self.R)

            for run in range(self.num_runs * 10):
                if self.p_mut > 0:
                    self.mutation_update(p_mut=self.p_mut)  # EXAM ADDITION
                sweep(self.grid, self.S, self.I, self.R)
                if run % 10 == 0:
                    infected = self.count_infected()
                    infected_count.append(infected)  # Sample infected count every 10 sweeps after equilibration.
                    infected_squared_count.append(
                        infected**2
                    )  # Second moment supports the variance estimate.

            mean_infected = np.mean(infected_count)  # First moment of the infected count.
            mean_infected_squared = np.mean(
                infected_squared_count
            )  # Second moment of the infected count.
            variance.append(
                (mean_infected_squared - mean_infected**2) / self.N**2
            )  # Susceptibility-like fluctuation measure for the infected fraction.

            bootstrap_variances = []
            for _ in range(1000):
                bootstrap_sample = np.random.choice(
                    infected_count, size=len(infected_count), replace=True
                )  # Bootstrap replica of the infected-count time series.
                bootstrap_mean = np.mean(bootstrap_sample)  # First moment of one bootstrap replica.
                bootstrap_mean_squared = np.mean(
                    bootstrap_sample**2
                )  # Second moment of one bootstrap replica.
                bootstrap_variances.append(
                    (bootstrap_mean_squared - bootstrap_mean**2) / self.N**2
                )  # Bootstrap variance estimate for this replica.
            variance_errors.append(
                np.std(bootstrap_variances)
            )  # Spread of bootstrap variances estimates the error bar.

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            np.arange(0.2, 0.5 + self.resolution, self.resolution),
            variance,
            yerr=variance_errors,
            label="Variance of Fraction of Infected Cells at Equilibrium",
        )
        plt.xlabel("Probability of Infection (S)")
        plt.ylabel("Variance of Fraction of Infected Cells at Equilibrium")
        plt.title("SIRS Model: Variance of Fraction of Infected Cells at Equilibrium")
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.resolution}_sirs_equilibrium_variance_plot.png")
        plt.show()

        np.savez(
            f"{self.resolution}_sirs_infection_variance_plot_data.npz",
            variance_data=variance,
            S_vales=np.arange(0.2, 0.5 + self.resolution, self.resolution),
            variance_errors=variance_errors,
        )

    def run_immunity(self):
        """Measure the mean infected fraction versus permanent immunity fraction.

        Args:
            None

        Returns:
            None: Displays and stores the immunity-response curve.

        Notes:
            Physics/formula used: increasing permanent immunity removes
            susceptible sites and suppresses sustained infection.
            ASSUMPTION: ``S = I = R = 0.5`` are fixed exactly as in the original code.
        """
        self.S = 0.5
        self.I = 0.5
        self.R = 0.5

        mean_infected = []

        for f in np.arange(0, 1 + self.resolution, self.resolution):
            self.f = f
            self.grid = self.initialize_grid()
            infected_count = []
            for _ in range(100):
                if self.p_mut > 0:
                    self.mutation_update(p_mut=self.p_mut)  # EXAM ADDITION
                sweep(self.grid, self.S, self.I, self.R)
            for _ in range(self.num_runs):
                if self.p_mut > 0:
                    self.mutation_update(p_mut=self.p_mut)  # EXAM ADDITION
                sweep(self.grid, self.S, self.I, self.R)
                infected_count.append(self.count_infected())
            mean_infected.append(
                np.mean(infected_count) / self.N**2
            )  # Mean infected fraction quantifies the epidemic level at this immune fraction.

        plt.figure(figsize=(10, 6))
        plt.plot(
            np.arange(0, 1 + self.resolution, self.resolution),
            mean_infected,
            label="Average Fraction of Infected Cells at Equilibrium",
        )
        plt.xlabel("Fraction of Recovered Cells that are Permanently Immune (f)")
        plt.ylabel("Average Fraction of Infected Cells at Equilibrium")
        plt.title("SIRS Model: Effect of Permanent Immunity on Equilibrium Infection Levels")
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.resolution}_sirs_equilibrium_immunity_plot.png")
        plt.show()

        np.savez(
            f"{self.resolution}_sirs_equilibrium_immunity_plot_data.npz",
            infected_count=mean_infected,
            f_values=np.arange(0, 1 + self.resolution, self.resolution),
        )


def autocorrelation(data):
    """Compute the normalised autocorrelation function and decorrelation time.

    Args:
        data (array-like): Time series of a scalar observable, such as infected count.

    Returns:
        tuple:
            ac (np.ndarray): Normalised autocorrelation.
            tau (float): Integrated decorrelation time.

    Notes:
        EXAM: After equilibration, store infected counts every few sweeps, call
        ``ac, tau = autocorrelation(infected_count)``, print ``tau``, and use it
        to justify your sampling interval before quoting any variance or error bar.
    """
    data = np.array(data, dtype=float) - np.mean(data)  # Remove the mean before correlating fluctuations.
    ac = np.correlate(data, data, mode="full")[len(data) - 1 :]  # Positive-lag autocovariance sequence.
    ac = ac / ac[0]  # normalise so ac[0] = 1
    tau = 0.5 + np.sum(ac[1:][ac[1:] > 0])  # integrated autocorrelation time
    return ac, tau


def bootstrap_error(data, stat_fn, n_samples=1000):
    """Estimate the standard error of a scalar statistic by bootstrap resampling.

    Args:
        data (array-like): Raw measurements, such as infected counts.
        stat_fn (callable): Function returning a scalar statistic.
        n_samples (int): Number of bootstrap resamples.

    Returns:
        float: Bootstrap estimate of the standard error.

    Notes:
        EXAM: If the question asks for an uncertainty on mean infected fraction
        or on a derived variance-like quantity, pass the corresponding statistic
        function to ``bootstrap_error`` and use the return value as ``yerr`` in
        your plot or as the quoted numerical uncertainty.
    """
    n = len(data)
    return np.std(
        [
            stat_fn(np.random.choice(data, size=n, replace=True))  # Bootstrap replica of the measurement list.
            for _ in range(n_samples)
        ]
    )


def gaussian_noise(shape, sigma=1.0):
    """Generate Gaussian noise with the Box-Muller transform.

    Args:
        shape (tuple): Output array shape.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: Gaussian-distributed random array.

    Notes:
        EXAM: If the paper asks for Gaussian disorder in an initial field or
        threshold variable, call ``gaussian_noise(shape, sigma)`` rather than
        ``np.random.randn`` so you can explicitly state that you used Box-Muller.
    """
    u1 = np.random.rand(*shape)  # Uniform source for the radial Box-Muller factor.
    u2 = np.random.rand(*shape)  # Uniform source for the angular Box-Muller factor.
    return sigma * np.sqrt(-2.0 * np.log(u1)) * np.cos(
        2.0 * np.pi * u2
    )  # Box-Muller


def survival_probability(step_fn, is_active_fn, n_runs=200, max_steps=500):
    """Estimate the fraction of runs still active at each time step.

    Args:
        step_fn (callable): Advances the model one sweep in-place.
        is_active_fn (callable): Returns ``True`` while infection survives.
        n_runs (int): Number of independent runs.
        max_steps (int): Maximum number of sweeps per run.

    Returns:
        np.ndarray: Survival fraction versus time.

    Notes:
        EXAM: Define ``is_active_fn`` as something like
        ``lambda: np.any(model.grid == 1)`` after seeding a single infected cell.
        Then call ``P = survival_probability(model.sweep, is_active_fn)`` for
        several values of ``S`` and compare the decay curves to locate the
        critical region between active and absorbing behaviour.
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
    parser = argparse.ArgumentParser(description="SIRS Model Simulation")

    parser.add_argument("-N", "--size", type=int, default=50, help="Size of the lattice (N x N)")
    parser.add_argument("--num_runs", type=int, default=1000, help="Number of simulation runs to perform")
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="Update resolution for probability updates in run method (default: 0.05)",
    )
    parser.add_argument(
        "-S",
        "--infected_prob",
        type=float,
        default=0.5,
        help="Probability of a susceptible cell becoming infected",
    )
    parser.add_argument(
        "-I",
        "--recovery_prob",
        type=float,
        default=0.5,
        help="Probability of an infected cell recovering",
    )
    parser.add_argument(
        "-R",
        "--resusceptibility_prob",
        type=float,
        default=0.5,
        help="Probability of a recovered cell becoming susceptible again",
    )
    parser.add_argument(
        "-f",
        "--immune_fraction",
        type=float,
        default=0.0,
        help="Fraction of recovered cells that become permanently immune",
    )
    parser.add_argument(
        "--p_mut",
        type=float,
        default=0.0,
        help="Spontaneous mutation probability",
    )
    parser.add_argument("--animate", action="store_true", help="Animate the evolution of the grid over time")
    parser.add_argument(
        "--run_variance",
        action="store_true",
        help="Run the simulation to calculate the variance of the number of infected cells at equilibrium",
    )
    parser.add_argument(
        "--run_immunity",
        action="store_true",
        help="Run the simulation to analyze the effect of permanent immunity on equilibrium infection levels",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to print additional information during simulation",
    )

    args = parser.parse_args()

    model = SIRS(
        N=args.size,
        debug=args.debug,
        num_runs=args.num_runs,
        S=args.infected_prob,
        I=args.recovery_prob,
        R=args.resusceptibility_prob,
        f=args.immune_fraction,
        resolution=args.resolution,
        p_mut=args.p_mut,
    )

    if args.debug:
        model.plot_single_frame()
    if args.animate:
        if args.debug:
            print("Starting animation of SIRS model evolution...")
        model.animate()
    elif args.run_variance:
        if args.debug:
            print("Starting variance analysis of SIRS model at equilibrium...")
        model.run_variance()
    elif args.run_immunity:
        if args.debug:
            print("Starting analysis of effect of permanent immunity on SIRS model equilibrium infection levels...")
        model.run_immunity()
    else:
        if args.debug:
            print("Starting main simulation run to analyze equilibrium infection levels across S and R parameter space...")
        model.run()
