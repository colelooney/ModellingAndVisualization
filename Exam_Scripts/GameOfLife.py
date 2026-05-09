"""
simulate the Game of life
"""

# ============================================================
# INDEX
# ============================================================
# CLASS: GameOfLife
#
#   SECTION: Initialisation
#     __init__(...)                 — store lattice, pattern, debugging, and
#                                     trajectory bookkeeping state
#     initialize_grid()             — return the requested initial configuration
#     determine_equilibriation()    — detect repeated states from recent history
#
#   SECTION: Core dynamics
#     sweep()                       — apply one Conway update sweep using
#                                     wrapped neighbour counts
#
#   SECTION: Observables & statistics
#     get_centre_of_mass()          — return the periodic centre of mass of live cells
#     calculate_speed()             — fit centre-of-mass travel to estimate speed
#
#   SECTION: Visualisation
#     animate()                     — animate the lattice evolution
#     plot_single_frame()           — display one initial configuration
#     plot_equilibrium_times()      — plot a histogram of equilibration times
#     plot_centres_of_mass()        — plot centre-of-mass displacement versus time
#
#   SECTION: Exam extensions
#     mutation_update(...)          — randomly flip cells before a normal sweep
#                                     to model spontaneous birth/death noise
#
#   SECTION: I/O & data storage
#     run()                         — measure equilibration times across runs
#     glider_run()                  — track the motion of a spaceship pattern
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
#   python GameOfLife.py -N 50 -S random -F 0.5
#   e.g. python GameOfLife.py -N 50 -S glider --animate
# ============================================================

import argparse
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

PATTERNS = {
    "glider": np.array([[0, 2], [1, 0], [2, 1], [2, 2], [1, 2]]),
    "LWSS": np.array(
        [[0, 0], [0, 2], [1, 3], [2, 3], [3, 0], [3, 3], [4, 1], [4, 2], [4, 3]]
    ),
    "MWSS": np.array(
        [
            [0, 2],
            [1, 0],
            [1, 4],
            [2, 5],
            [3, 0],
            [3, 5],
            [4, 1],
            [4, 2],
            [4, 3],
            [4, 4],
            [4, 5],
        ]
    ),
    "HWSS": np.array(
        [
            [0, 2],
            [0, 3],
            [1, 0],
            [1, 5],
            [2, 6],
            [3, 0],
            [3, 6],
            [4, 1],
            [4, 2],
            [4, 3],
            [4, 4],
            [4, 5],
            [4, 6],
        ]
    ),
    "block": np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
    "beehive": np.array([[1, 0], [2, 0], [0, 1], [3, 1], [1, 2], [2, 2]]),
    "loaf": np.array([[1, 0], [2, 0], [0, 1], [3, 1], [1, 2], [3, 2], [2, 3]]),
    "boat": np.array([[0, 0], [1, 0], [0, 1], [2, 1], [1, 2]]),
    "tub": np.array([[1, 0], [0, 1], [2, 1], [1, 2]]),
    "blinker": np.array([[0, 0], [1, 0], [2, 0]]),
    "toad": np.array([[1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1]]),
    "beacon": np.array([[0, 0], [1, 0], [0, 1], [3, 2], [2, 3], [3, 3]]),
    "pulsar": np.array(
        [
            [2, 0],
            [3, 0],
            [4, 0],
            [8, 0],
            [9, 0],
            [10, 0],
            [0, 2],
            [5, 2],
            [7, 2],
            [12, 2],
            [0, 3],
            [5, 3],
            [7, 3],
            [12, 3],
            [0, 4],
            [5, 4],
            [7, 4],
            [12, 4],
            [2, 5],
            [3, 5],
            [4, 5],
            [8, 5],
            [9, 5],
            [10, 5],
            [2, 7],
            [3, 7],
            [4, 7],
            [8, 7],
            [9, 7],
            [10, 7],
            [0, 8],
            [5, 8],
            [7, 8],
            [12, 8],
            [0, 9],
            [5, 9],
            [7, 9],
            [12, 9],
            [0, 10],
            [5, 10],
            [7, 10],
            [12, 10],
            [2, 12],
            [3, 12],
            [4, 12],
            [8, 12],
            [9, 12],
            [10, 12],
        ]
    ),
}


class GameOfLife:
    """Conway's Game of Life on a periodic square lattice."""

    # ── Initialisation ───────────────────────────────────────────

    def __init__(self, N, initial_state, debug, alive_fraction, num_runs):
        """Store simulation settings and initialise the lattice.

        Args:
            N (int): Linear lattice size for an ``N x N`` grid.
            initial_state (str): Named starting pattern or ``"random"``.
            debug (bool): Whether to print diagnostic output.
            alive_fraction (float): Initial live-cell fraction for random starts.
            num_runs (int): Number of runs used for averaged statistics.

        Returns:
            None: Initialises the model state in place.

        Notes:
            Physics/formula used: cells are binary, with 1 for alive and 0 for dead.
            ASSUMPTION: periodic boundaries are used everywhere in the model.
        """
        self.N = N
        self.initial_state = initial_state
        self.current_grid = None
        self.future_grid = None
        self.debug = debug
        self.alive_fraction = alive_fraction
        self.num_runs = num_runs
        self.history = deque(maxlen=2)
        if self.debug:
            print(
                f"Initialized GameOfLife with N={self.N}, "
                f"initial_state={self.initial_state}, "
                f"alive_fraction={self.alive_fraction}"
            )

        self.equilibrium_times = []
        self.centres_of_mass = []
        self.initial_state_patterns = PATTERNS

        self.current_grid = self.initialize_grid()  # EXAM ADDITION
        self.future_grid = np.copy(self.current_grid)  # EXAM ADDITION

    def initialize_grid(self):
        """Generate the requested initial lattice configuration.

        Args:
            None

        Returns:
            np.ndarray: ``N x N`` array containing 1 for alive and 0 for dead.

        Notes:
            Physics/formula used: random starts use Bernoulli occupancy, while
            named patterns are stamped into a zero background.
            ASSUMPTION: pattern placement wraps periodically across boundaries.
        """
        grid = np.zeros((self.N, self.N), dtype=np.int8)
        if self.initial_state == "random":
            grid = np.random.choice(
                [0, 1],
                size=(self.N, self.N),
                p=[1 - self.alive_fraction, self.alive_fraction],
            )  # Bernoulli occupation sets the initial live-cell density.
        else:
            grid = np.zeros((self.N, self.N))
            random_x, random_y = np.random.randint(0, self.N, size=2)
            pattern = self.initial_state_patterns[self.initial_state]
            for cell in pattern:
                x = (random_x + cell[0]) % self.N  # Wrapped x coordinate places the motif periodically.
                y = (random_y + cell[1]) % self.N  # Wrapped y coordinate places the motif periodically.
                grid[x, y] = 1  # Mark the selected site as alive.
        return grid

    def determine_equilibriation(self):
        """Detect whether the system has repeated a recent state.

        Args:
            None

        Returns:
            bool: ``True`` if the current configuration repeats recent history.

        Notes:
            Physics/formula used: a repeated lattice indicates a fixed point or a
            short-period oscillator under the current history window.
            ASSUMPTION: two previous states are enough for the intended analysis.
        """
        current_hash = self.current_grid.tobytes()  # Byte hash uniquely labels the present lattice state.
        if current_hash in self.history:
            return True
        self.history.append(current_hash)  # Store recent states to detect repeats.
        return False

    # ── Core dynamics ────────────────────────────────────────────

    def sweep(self):
        """Apply one Conway update sweep to the full lattice.

        Args:
            None

        Returns:
            None: Updates ``self.future_grid`` in place.

        Notes:
            Physics/formula used: live cells survive with 2 neighbours, any cell
            is born with 3 neighbours, and all other cells die or stay dead.
            ASSUMPTION: all updates are synchronous through ``future_grid``.
        """
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = convolve2d(
            self.current_grid,
            kernel,
            mode="same",
            boundary="wrap",
        )  # Wrapped convolution counts the eight neighbours around each cell.
        self.future_grid = (
            (neighbor_count == 3)
            | ((self.current_grid == 1) & (neighbor_count == 2))
        ).astype(int)  # Synchronous Conway rule produces the next generation.

    # ── Observables & statistics ─────────────────────────────────

    def get_centre_of_mass(self):
        """Compute the periodic centre of mass of all live cells.

        Args:
            None

        Returns:
            np.ndarray | None: Wrapped centre of mass ``(x, y)``, or ``None`` if
            no cells are alive.

        Notes:
            Physics/formula used: maps coordinates to angles so the centre of
            mass remains meaningful on a periodic domain.
            ASSUMPTION: all live cells have equal weight.
        """
        indices = np.argwhere(self.current_grid == 1)  # Positions of live cells define the active cluster.
        if len(indices) == 0:
            return None

        theta = (indices / self.N) * 2 * np.pi  # Convert lattice coordinates to angles on the torus.
        mean_cos = np.mean(np.cos(theta), axis=0)  # Circular mean of x/y positions via cosine projection.
        mean_sin = np.mean(np.sin(theta), axis=0)  # Circular mean of x/y positions via sine projection.
        mean_theta = np.arctan2(mean_sin, mean_cos)  # Recover the wrapped mean angle.
        com = ((mean_theta + 2 * np.pi) % (2 * np.pi)) * self.N / (
            2 * np.pi
        )  # Map the wrapped angular mean back to lattice coordinates.
        return com

    def calculate_speed(self):
        """Estimate centre-of-mass speed from cumulative displacement.

        Args:
            None

        Returns:
            tuple: ``(slope, cum_dist, time, intercept)`` from a linear fit.

        Notes:
            Physics/formula used: unwraps periodic jumps, then fits cumulative
            distance versus time to obtain an average speed.
            ASSUMPTION: the motion is approximately linear over the sampled window.
        """
        centres_of_mass_array = np.array(self.centres_of_mass)
        diffs = np.diff(
            centres_of_mass_array, axis=0
        )  # Consecutive centre-of-mass displacements measure pattern drift.
        diffs = (
            diffs + self.N / 2
        ) % self.N - self.N / 2  # Wrapped displacement removes artificial jumps across periodic edges.
        distances = np.linalg.norm(
            diffs, axis=1
        )  # Euclidean displacement per step gives the pattern travel distance.
        cum_dist = np.cumsum(distances)  # Cumulative path length is used for the speed fit.
        time = np.arange(len(cum_dist))  # Time index for the displacement record.
        slope, intercept = np.polyfit(
            time, cum_dist, 1
        )  # Linear fit estimates average speed from distance-time data.

        if self.debug:
            print(
                f"Centres of Mass: {centres_of_mass_array}"
                f"\nDiffs: {diffs}"
                f"\nDistances: {distances}"
                f"\ntime:steps: {np.arange(0, distances.size)}"
            )

        return slope, cum_dist, time, intercept

    # ── Visualisation ────────────────────────────────────────────

    def animate(self):
        """Animate the lattice evolution in real time.

        Args:
            None

        Returns:
            None: Displays a Matplotlib animation.

        Notes:
            Physics/formula used: each animation frame advances the automaton by
            one synchronous Conway update.
            ASSUMPTION: animation is for qualitative inspection only.
        """
        self.current_grid = self.initialize_grid()
        self.future_grid = np.copy(self.current_grid)

        fig = plt.figure()
        im = plt.imshow(self.current_grid, animated=True, cmap="binary")

        def update_frame(_):
            self.sweep()
            self.current_grid = self.future_grid.copy()  # Promote the next generation to the current state.
            im.set_array(self.current_grid)
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

    def plot_single_frame(self):
        """Display one initial configuration of the automaton.

        Args:
            None

        Returns:
            None: Shows the lattice as an image.

        Notes:
            Physics/formula used: visualises the initial occupancy field only.
            ASSUMPTION: this is mainly a debugging or setup check.
        """
        self.current_grid = self.initialize_grid()
        plt.imshow(self.current_grid, cmap="binary")
        plt.title("Initial State of Game of Life")
        plt.axis("off")
        plt.show()

    def plot_equilibrium_times(self):
        """Plot a histogram of equilibration times over repeated runs.

        Args:
            None

        Returns:
            None: Saves data and displays the histogram.

        Notes:
            Physics/formula used: repeated-state detection is used as the
            practical definition of equilibrium in this script.
            ASSUMPTION: the recorded runs are statistically comparable.
        """
        np.savez(
            f"{self.initial_state}_equilibrium_times",
            equilibrium_times=self.equilibrium_times,
        )
        plt.hist(
            self.equilibrium_times, bins=50, color="orange", alpha=0.7
        )  # Histogram shows the distribution of repeat-state times.
        plt.title("Distribution of Equilibrium Times")
        plt.xlabel("Time to Equilibrium (steps)")
        plt.ylabel("Frequency")
        plt.savefig(f"{self.initial_state}_equilibrium_times_histogram.png")
        plt.show()

    def plot_centres_of_mass(self):
        """Plot cumulative centre-of-mass displacement and fitted speed.

        Args:
            None

        Returns:
            None: Saves data and displays the trajectory plot.

        Notes:
            Physics/formula used: the fitted slope of cumulative displacement is
            interpreted as the spaceship speed.
            ASSUMPTION: enough centre-of-mass samples were stored beforehand.
        """
        centres_of_mass_array = np.array(self.centres_of_mass)
        slope, cum_dist, time, intercept = self.calculate_speed()

        np.savez(
            f"{self.initial_state}",
            centres_of_mass=centres_of_mass_array,
            times=time,
            speed=slope,
            distance=cum_dist,
            y_intercept=intercept,
        )

        plt.plot(
            time,
            cum_dist,
            "bo",
            markersize=2,
            color="blue",
            label="Centre of Mass Trajectory",
        )
        plt.plot(
            time,
            slope * time + intercept,
            "r-",
            label=f"Fit: v={slope:.3f}",
        )  # Linear trend estimates the spaceship speed.
        plt.legend()
        plt.title("Trajectory of Centre of Mass of Alive Cells")
        plt.xlabel("Time iteration")
        plt.ylabel("Speed of Centre of Mass")
        plt.savefig(f"{self.initial_state}_centre_of_mass_trajectory.png")
        plt.show()

    # ── Exam extensions ──────────────────────────────────────────

    def mutation_update(self, p_mut=0.01):  # EXAM ADDITION
        """Randomly flip cell states with probability ``p_mut``.

        Args:
            p_mut (float): Per-cell probability of a spontaneous flip.

        Returns:
            None: Modifies ``self.current_grid`` in place.

        Notes:
            Physics/formula used: independent Bernoulli flips model spontaneous
            birth/death noise outside the standard neighbour rules.
            ASSUMPTION: flips are applied before the normal synchronous sweep.
            EXAM: If the question adds a spontaneous mutation probability, call
            ``self.mutation_update(p_mut=0.01)`` immediately before
            ``self.sweep()``. Then state that the base Conway rule is unchanged
            and the extra term acts as external stochastic noise.
        """
        mutation_mask = np.random.rand(self.N, self.N) < p_mut  # Independent per-site noise trigger.
        self.current_grid[mutation_mask] = (
            1 - self.current_grid[mutation_mask]
        )  # Flip alive to dead and dead to alive at mutated sites.

    # ── I/O & data storage ───────────────────────────────────────

    def run(self):
        """Run repeated simulations and record equilibration times.

        Args:
            None

        Returns:
            None: Prints the average equilibration time and may plot a histogram.

        Notes:
            Physics/formula used: equilibrium means the lattice repeats a recent
            state under the current history criterion.
            ASSUMPTION: random starts use a longer cutoff because they can take
            much longer to settle.
        """
        for run in range(self.num_runs):
            self.current_grid = self.initialize_grid()
            self.future_grid = np.copy(self.current_grid)
            self.history.clear()

            if self.initial_state in ["random"]:
                max_time_steps = 15000
            else:
                max_time_steps = 100

            self.time_steps = 0
            while self.time_steps < max_time_steps:
                self.sweep()
                if self.determine_equilibriation():
                    break
                self.current_grid = self.future_grid.copy()  # Advance to the next synchronous configuration.
                self.time_steps += 1

            self.equilibrium_times.append(self.time_steps)
            if self.debug:
                print(
                    f"Run {run + 1}/{self.num_runs}: "
                    f"Time to equilibrium = {self.time_steps} steps"
                )

        average_time = np.mean(
            self.equilibrium_times
        )  # Mean repeat time summarises the run ensemble.
        print(
            f"Average time to reach equilibrium over {self.num_runs} runs: "
            f"{average_time:.2f} steps"
        )

        if self.initial_state in ["random"]:
            self.plot_equilibrium_times()

    def glider_run(self):
        """Track the motion of a spaceship pattern over time.

        Args:
            None

        Returns:
            None: Stores centre-of-mass data and plots the fitted trajectory.

        Notes:
            Physics/formula used: repeated synchronous sweeps move the spaceship
            across the periodic lattice.
            ASSUMPTION: the chosen pattern is a moving spaceship.
        """
        self.current_grid = self.initialize_grid()
        self.future_grid = np.copy(self.current_grid)
        self.centres_of_mass = []

        max_time_steps = 100
        self.time_steps = 0
        while self.time_steps < max_time_steps:
            self.sweep()
            self.current_grid = self.future_grid.copy()  # Promote the new pattern position.
            self.time_steps += 1
            self.centres_of_mass.append(
                self.get_centre_of_mass()
            )  # Record wrapped centre of mass to estimate motion.

        self.plot_centres_of_mass()


def autocorrelation(data):
    """Compute the normalised autocorrelation function and decorrelation time.

    Args:
        data (array-like): Time series of a scalar observable, e.g. a list
            of live-cell counts sampled at regular intervals.

    Returns:
        tuple:
            ac (np.ndarray): Normalised autocorrelation of the input series.
            tau (float): Integrated decorrelation time.

    Notes:
        Formula: ``ac(t) = C(t)/C(0)`` where ``C(t)`` is the autocovariance.

        EXAM: After equilibration, store a time series such as the live-cell
        count every sweep, then call ``ac, tau = autocorrelation(counts)``.
        Quote ``tau`` to justify how far apart your samples must be before you
        treat them as effectively independent.
    """
    data = np.array(data, dtype=float) - np.mean(data)  # Remove the mean before correlating fluctuations.
    ac = np.correlate(data, data, mode="full")[len(data) - 1 :]  # Positive-lag autocovariance sequence.
    ac = ac / ac[0]  # normalise so ac[0] = 1
    tau = 0.5 + np.sum(ac[1:][ac[1:] > 0])  # integrated autocorrelation time
    return ac, tau


def bootstrap_error(data, stat_fn, n_samples=1000):
    """Estimate the standard error of a scalar statistic by bootstrap resampling.

    Args:
        data (array-like): Raw measurements, e.g. a list of lifetimes.
        stat_fn (callable): Function returning a scalar statistic.
        n_samples (int): Number of bootstrap resamples.

    Returns:
        float: Bootstrap estimate of the standard error.

    Notes:
        EXAM: If you need an uncertainty on an average lifetime or speed, call
        ``bootstrap_error(data, np.mean)`` or pass a custom statistic function.
        Report the returned value as the error bar on the quantity you plot.
    """
    n = len(data)
    return np.std(
        [
            stat_fn(np.random.choice(data, size=n, replace=True))  # Bootstrap replica of the measured sample.
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
        EXAM: If the question asks for Gaussian perturbations around a seed
        density, generate ``noise = gaussian_noise((N, N), sigma=0.01)`` and
        threshold or add it to your initial field exactly as requested.
    """
    u1 = np.random.rand(*shape)  # Uniform source for the radial Box-Muller term.
    u2 = np.random.rand(*shape)  # Uniform source for the angular Box-Muller term.
    return sigma * np.sqrt(-2.0 * np.log(u1)) * np.cos(
        2.0 * np.pi * u2
    )  # Box-Muller


def survival_probability(step_fn, is_active_fn, n_runs=200, max_steps=500):
    """Estimate the fraction of runs still active at each time step.

    Args:
        step_fn (callable): Advances the model one sweep in-place.
        is_active_fn (callable): Returns ``True`` while activity remains.
        n_runs (int): Number of independent runs.
        max_steps (int): Maximum number of sweeps per run.

    Returns:
        np.ndarray: Survival fraction versus time.

    Notes:
        EXAM: For noisy Game of Life or contact-process variants, define an
        ``is_active_fn`` that checks whether any live cells remain, then call
        ``P = survival_probability(model.sweep, has_live_cells)``. Plot ``P(t)``
        and use the shape of the decay to discuss active versus absorbing phases.
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
    parser = argparse.ArgumentParser(description="Conway's Game of Life Simulation")
    choices = ["random"] + list(PATTERNS.keys())

    parser.add_argument(
        "-N", "--size", type=int, default=50, help="Size of the lattice (N x N)"
    )
    parser.add_argument(
        "-S",
        "--initial_state",
        type=str,
        default="random",
        choices=choices,
        help="Initial state of the lattice (random or ordered)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to print additional information during simulation",
    )
    parser.add_argument(
        "-F",
        "--alive_fraction",
        type=float,
        default=0.5,
        help="Fraction of cells that are initially alive (only used if initial_state is random)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate the evolution of the grid over time",
    )
    parser.add_argument(
        "-R",
        "--num_runs",
        type=int,
        default=1000,
        help="Number of simulation runs to perform for averaging equilibrium times",
    )

    args = parser.parse_args()

    model = GameOfLife(
        N=args.size,
        initial_state=args.initial_state,
        debug=args.debug,
        alive_fraction=args.alive_fraction,
        num_runs=args.num_runs,
    )
    if args.debug:
        model.plot_single_frame()
    if args.animate:
        model.animate()
    elif args.initial_state in ["glider", "LWSS", "MWSS", "HWSS"]:
        model.glider_run()
    else:
        model.run()
