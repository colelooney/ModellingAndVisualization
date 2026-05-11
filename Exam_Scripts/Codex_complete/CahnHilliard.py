"""
Numerical solve partial differential equation for Cahn-Hilliard equation
"""

# ============================================================
# INDEX
# ============================================================
# CLASS: cahn_hilliard
#
#   SECTION: Initialisation
#     __init__(...)                 — store grid spacing, timestep, composition,
#                                     and convergence settings
#     initialize_grid()             — return the initial composition field with noise
#     laplacian(...)                — compute the finite-difference Laplacian
#
#   SECTION: Core dynamics
#     chemical_potential()          — compute the Cahn-Hilliard chemical potential
#     sweep_phi()                   — advance the composition by one explicit time step
#
#   SECTION: Observables & statistics
#     free_energy()                 — compute the total free energy functional
#
#   SECTION: Visualisation
#     animate()                     — animate the phase-separating field
#     run_energy()                  — evolve until convergence and plot free energy
#
#   SECTION: Exam extensions
#     bilaplacian()                 — compute nabla^4 phi for higher-order PDE variants
#                                     needed in modified 2025-style exam equations
#     fisher_reaction(...)          — compute a Fisher-KPP reaction source term
#                                     for reaction-diffusion exam variants
#     lax_advection(...)            — compute one stable Lax advection step
#                                     for advection-augmented PDE exam variants
#
#   SECTION: I/O & data storage
#     None in the base class; plotting and saving occur in run_energy()
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
#   python CahnHilliard.py -N 100 --phi0 0 --num_iter 50000
#   e.g. python CahnHilliard.py -N 100 --phi0 0 --animate
# ============================================================

import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class cahn_hilliard:
    """Explicit finite-difference solver for the Cahn-Hilliard equation."""

    # ── Initialisation ───────────────────────────────────────────

    def __init__(self, N, phi_0, dx, dt, num_iter, threshold):
        """Store simulation parameters and initialise the composition field.

        Args:
            N (int): Linear lattice size for an ``N x N`` field.
            phi_0 (float): Mean composition of the system.
            dx (float): Spatial lattice spacing.
            dt (float): Time step for explicit updates.
            num_iter (int): Maximum number of iterations to run.
            threshold (float): Energy-fluctuation threshold used for convergence.

        Returns:
            None: Initialises attributes in place.

        Notes:
            Physics/formula used: the model evolves a conserved scalar order
            parameter ``phi`` on a periodic square lattice.
            ASSUMPTION: all coefficients are scaled to the original dimensionless
            form used in the script.
        """
        self.N = N
        self.dt = dt
        self.dx = dx
        self.phi_0 = phi_0
        self.num_iter = num_iter
        self.threshold = threshold
        self.phi = self.initialize_grid()

    def initialize_grid(self):
        """Generate the initial composition field with small random noise.

        Args:
            None

        Returns:
            np.ndarray: ``N x N`` field fluctuating weakly around ``phi_0``.

        Notes:
            Physics/formula used: phase separation is seeded by small-amplitude
            random composition fluctuations.
            ASSUMPTION: the initial noise amplitude is fixed at 0.01.
        """
        noise = 0.01 * (
            np.random.rand(self.N, self.N) - 0.5
        )  # Small random fluctuations seed spinodal decomposition.
        return self.phi_0 + noise

    def laplacian(self, f):
        """Compute the 2D Laplacian with periodic finite differences.

        Args:
            f (np.ndarray): Scalar field on the lattice.

        Returns:
            np.ndarray: Discrete Laplacian of ``f``.

        Notes:
            Physics/formula used: central-difference Laplacian with wrapped
            neighbours in both directions.
            ASSUMPTION: the grid is periodic.
        """
        f_left = np.roll(f, -1, axis=1)  # Field at j+1 enters the x-direction second derivative.
        f_right = np.roll(f, 1, axis=1)  # Field at j-1 enters the x-direction second derivative.
        f_up = np.roll(f, -1, axis=0)  # Field at i+1 enters the y-direction second derivative.
        f_down = np.roll(f, 1, axis=0)  # Field at i-1 enters the y-direction second derivative.
        return (f_left + f_right + f_up + f_down - 4 * f) / (
            self.dx**2
        )  # Standard five-point stencil approximates nabla^2 f.

    # ── Core dynamics ────────────────────────────────────────────

    def chemical_potential(self):
        """Compute the current chemical potential field.

        Args:
            None

        Returns:
            np.ndarray: Chemical potential ``mu`` at each lattice site.

        Notes:
            Physics/formula used: ``mu = -phi + phi^3 - nabla^2 phi`` in the
            dimensionless form used by the original code.
            ASSUMPTION: mobility and gradient-energy coefficients are 1.
        """
        mu = -self.phi + self.phi**3 - self.laplacian(
            self.phi
        )  # Local double-well term plus gradient penalty define the chemical potential.
        return mu

    def sweep_phi(self):
        """Advance the composition field by one explicit Cahn-Hilliard step.

        Args:
            None

        Returns:
            None: Updates ``self.phi`` in place.

        Notes:
            Physics/formula used: ``phi_t = nabla^2 mu`` with an explicit Euler
            time step, followed by mean subtraction to enforce mass conservation.
            ASSUMPTION: the explicit timestep is small enough to remain stable.
        """
        mu = self.chemical_potential()
        self.phi += self.dt * self.laplacian(
            mu
        )  # Conservative diffusive flux evolves phi according to nabla^2 mu.
        self.phi -= np.mean(self.phi) - self.phi_0  # Enforce conservation of the spatial average composition.

    # ── Observables & statistics ─────────────────────────────────

    def free_energy(self):
        """Compute the total free energy of the current composition field.

        Args:
            None

        Returns:
            float: Total free energy summed over the lattice.

        Notes:
            Physics/formula used: integrates the double-well bulk term and
            gradient penalty ``0.5 |grad phi|^2``.
            ASSUMPTION: central differences are used for both gradient components.
        """
        f = -0.5 * self.phi**2 + 0.25 * self.phi**4  # Local double-well free-energy density.
        grad_x = (
            np.roll(self.phi, -1, axis=0) - np.roll(self.phi, 1, axis=0)
        ) / (2 * self.dx)  # Central difference approximates dphi/dx.
        grad_y = (
            np.roll(self.phi, -1, axis=1) - np.roll(self.phi, 1, axis=1)
        ) / (2 * self.dx)  # Central difference approximates dphi/dy.
        f += 0.5 * (
            grad_x**2 + grad_y**2
        )  # Gradient penalty discourages sharp interfaces and defines interfacial energy.
        return float(np.sum(f))  # Summed density gives the total free energy functional.

    # ── Visualisation ────────────────────────────────────────────

    def animate(self):
        """Animate the evolution of the composition field.

        Args:
            None

        Returns:
            None: Displays a Matplotlib animation.

        Notes:
            Physics/formula used: each frame applies 10 explicit CH steps to make
            domain growth visible on-screen.
            ASSUMPTION: animation is for qualitative inspection rather than data taking.
        """
        self.phi = self.initialize_grid()
        fig = plt.figure()
        im = plt.imshow(self.phi, animated=True, cmap="coolwarm")

        def update_frame(_):
            for _ in range(10):
                self.sweep_phi()
            im.set_array(self.phi)
            return [im]

        ani = animation.FuncAnimation(
            fig,
            update_frame,
            frames=1000,
            interval=1,
            blit=True,
            repeat_delay=1000,
        )
        plt.show()

    def run_energy(self):
        """Evolve the field and plot the free-energy history until convergence.

        Args:
            None

        Returns:
            None: Saves the energy history and displays the plot.

        Notes:
            Physics/formula used: equilibrium is diagnosed from the standard
            deviation of the recent free-energy window.
            ASSUMPTION: the original sliding-window convergence test is retained.
        """
        energy_history = [self.free_energy()]
        iters = [0]
        window = 50

        for i in range(1, self.num_iter):
            self.sweep_phi()
            energy_history.append(self.free_energy())  # Free energy should relax downward during phase separation.
            iters.append(i)

            if i > window:
                recent = energy_history[-window:]  # Recent energy window is used for convergence diagnostics.
                if np.std(recent) < self.threshold:
                    print(
                        f"Equilibrium reached at iteration {i} "
                        f"(Threshold: {self.threshold})"
                    )
                    break

        plt.figure(figsize=(8, 5))
        plt.plot(iters, energy_history)
        plt.title(f"Free Energy Minimisation (phi_0={self.phi_0})")
        plt.xlabel("Iterations")
        plt.ylabel("Total Free Energy")
        plt.savefig(f"free_energy_phi{self.phi_0}.png")
        plt.show()

        np.savetxt(f"energy_phi{self.phi_0}.dat", energy_history)

    # ── Exam extensions ──────────────────────────────────────────

    def bilaplacian(self):  # EXAM ADDITION
        """Compute the bi-Laplacian ``nabla^4 phi``.

        Args:
            None

        Returns:
            np.ndarray: Fourth-order spatial derivative of ``phi``.

        Notes:
            Physics/formula used: ``nabla^4 phi = nabla^2(nabla^2 phi)``.
            ASSUMPTION: the same periodic five-point stencil is used twice.
            EXAM: If the paper adds a ``kappa * nabla^4 phi`` correction, call
            ``self.bilaplacian()`` inside ``sweep_phi()`` and add
            ``self.phi -= self.dt * kappa * self.bilaplacian()`` after the
            existing update line. Keep the mass-conservation line unchanged.
        """
        return self.laplacian(
            self.laplacian(self.phi)
        )  # Applying the Laplacian twice produces the fourth-order diffusion operator.

    def fisher_reaction(self, alpha=1.0):  # EXAM ADDITION
        """Compute the Fisher-KPP reaction term ``alpha * phi * (1 - phi)``.

        Args:
            alpha (float): Reaction rate. Default 1.0.

        Returns:
            np.ndarray: Reaction contribution at each lattice point.

        Notes:
            Physics/formula used: logistic growth drives ``phi`` toward 0 or 1.
            ASSUMPTION: the source term is local and does not conserve total mass.
            EXAM: If the paper adds a reaction term, evaluate
            ``self.fisher_reaction(alpha=1.0)`` and add
            ``self.phi += self.dt * self.fisher_reaction(alpha=1.0)`` after the
            existing conservative update. Then note explicitly that free energy
            need not decrease monotonically once reaction is present.
        """
        return alpha * self.phi * (
            1 - self.phi
        )  # Logistic reaction source grows intermediate values toward the stable states.

    def lax_advection(self, v, axis=0):  # EXAM ADDITION
        """Compute one stable Lax advection step.

        Args:
            v (float): Advection velocity.
            axis (int): Direction of advection, 0 for rows and 1 for columns.

        Returns:
            np.ndarray: Advected field after one Lax step.

        Notes:
            Physics/formula used: the Lax method replaces the local value by the
            neighbour average before applying the central advection difference.
            ASSUMPTION: CFL stability requires ``v * dt / dx <= 1``.
            EXAM: If the paper adds ``v dphi/dx`` or ``v dphi/dy``, replace the
            raw explicit update with ``self.phi = self.lax_advection(v, axis)``
            instead of using a plain central-difference advection term. Check
            ``self.dt * v / self.dx`` and reduce ``dt`` if it exceeds 1.
        """
        # ASSUMPTION: CFL condition v*dt/dx <= 1 must hold — reduce dt if needed.
        fwd = np.roll(self.phi, -1, axis=axis)  # Forward neighbour supplies the advective gradient.
        bwd = np.roll(self.phi, 1, axis=axis)  # Backward neighbour supplies the advective gradient.
        avg = 0.5 * (fwd + bwd)  # Lax averaging adds stabilising numerical diffusion.
        return avg - (v * self.dt / (2 * self.dx)) * (
            fwd - bwd
        )  # Stable Lax update advects the field along the chosen axis.


def autocorrelation(data):
    """Compute the normalised autocorrelation function and decorrelation time.

    Args:
        data (array-like): Time series such as free energy sampled during a run.

    Returns:
        tuple:
            ac (np.ndarray): Normalised autocorrelation.
            tau (float): Integrated decorrelation time.

    Notes:
        EXAM: Record an observable every few sweeps, call
        ``ac, tau = autocorrelation(samples)``, and report ``tau`` to justify the
        sampling interval you used when estimating averages or uncertainties.
    """
    data = np.array(data, dtype=float) - np.mean(data)  # Remove the mean before correlating fluctuations.
    ac = np.correlate(data, data, mode="full")[len(data) - 1 :]  # Positive-lag autocovariance sequence.
    ac = ac / ac[0]  # normalise so ac[0] = 1
    tau = 0.5 + np.sum(ac[1:][ac[1:] > 0])  # integrated autocorrelation time
    return ac, tau


def bootstrap_error(data, stat_fn, n_samples=1000):
    """Estimate the standard error of a scalar statistic by bootstrap resampling.

    Args:
        data (array-like): Raw measurements.
        stat_fn (callable): Function returning a scalar statistic.
        n_samples (int): Number of bootstrap resamples.

    Returns:
        float: Bootstrap estimate of the standard error.

    Notes:
        EXAM: If you need an error bar on mean free energy, domain size, or any
        other scalar derived from a sample list, pass the corresponding
        statistic into ``bootstrap_error`` and use the result as the uncertainty.
    """
    n = len(data)
    return np.std(
        [
            stat_fn(np.random.choice(data, size=n, replace=True))  # Bootstrap replica of the measured observable list.
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
        EXAM: If the initial field must be Gaussian rather than uniform, replace
        the current noise line with
        ``noise = gaussian_noise((self.N, self.N), sigma=0.01)`` and add it to
        ``phi_0`` before the evolution starts.
    """
    u1 = np.random.rand(*shape)  # Uniform source for the radial Box-Muller factor.
    u2 = np.random.rand(*shape)  # Uniform source for the angular Box-Muller factor.
    return sigma * np.sqrt(-2.0 * np.log(u1)) * np.cos(
        2.0 * np.pi * u2
    )  # Box-Muller


def survival_probability(step_fn, is_active_fn, n_runs=200, max_steps=500):
    """Estimate the fraction of runs still active at each time step.

    Args:
        step_fn (callable): Advances the model one step in-place.
        is_active_fn (callable): Returns ``True`` while activity survives.
        n_runs (int): Number of independent runs.
        max_steps (int): Maximum number of steps per run.

    Returns:
        np.ndarray: Survival fraction versus time.

    Notes:
        EXAM: This is mainly for absorbing-state variants of PDE or lattice
        models. If your modified equation has an active/inactive threshold,
        define ``is_active_fn`` to detect whether activity remains and compare
        survival curves across parameter values using this helper.
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
    parser = argparse.ArgumentParser(description="Cahn Hilliard Equation")
    parser.add_argument("-N", "--size", type=int, default=100, help="Size of the lattice (N x N)")
    parser.add_argument("--phi0", type=float, default=0, help="Average composition of grid")
    parser.add_argument("-dx", type=float, default=1, help="Length step")
    parser.add_argument("-dt", type=float, default=1e-4, help="Time step")
    parser.add_argument("--num_iter", type=int, default=50000, help="Number of Iteration to Run")
    parser.add_argument("--threshold", type=float, default=1e-6, help="Change thershold to determine equilibrium")
    parser.add_argument("--animate", action="store_true", help="argument to animate grid")

    args = parser.parse_args()

    model = cahn_hilliard(
        N=args.size,
        phi_0=args.phi0,
        dx=args.dx,
        dt=args.dt,
        num_iter=args.num_iter,
        threshold=args.threshold,
    )
    if args.animate:
        model.animate()
    else:
        model.run_energy()
