"""
Numerical solve partial differential equation for and poisson equation
"""

# ============================================================
# INDEX
# ============================================================
# CLASS: poisson
#
#   SECTION: Initialisation
#     __init__(...)                 — store solver parameters and initialise
#                                     potential and source grids
#     initialize_grid()             — return the initial potential and charge distribution
#     laplacian(...)                — compute the finite-difference 3D Laplacian
#
#   SECTION: Core dynamics
#     electric_field()              — compute E = -grad(phi)
#     jacobi(...)                   — compute one Jacobi update of the potential
#     jacobi_sweep()                — iterate Jacobi updates to convergence
#     gauss_seidel_step(...)        — perform one red-black Gauss-Seidel step
#     gauss_seidel_sweep()          — iterate Gauss-Seidel updates to convergence
#     sor_step(...)                 — perform one red-black SOR step
#     sor_sweep(...)                — iterate SOR updates to convergence
#     magnetic_field()              — compute B = curl(A) for the wire case
#     solve()                       — run the selected solver to convergence
#
#   SECTION: Observables & statistics
#     field_strength_vs_distance()  — bin and fit electric-field magnitude versus radius
#     potential_vs_distance()       — bin and fit electric potential versus radius
#     magnetic_strength_vs_distance() — bin and fit magnetic-field magnitude versus radius
#     vector_potential_vs_distance() — bin and fit vector potential versus radius
#
#   SECTION: Visualisation
#     contour_plot()                — contour plot of the monopole midplane potential
#     B_field_contour_plot()        — contour plot of the wire midplane potential
#     B_field_vector_plot()         — quiver plot of the wire magnetic field
#     vector_plot()                 — quiver plot of the monopole electric field
#     analyse()                     — run the relevant analysis/plotting suite
#     run()                         — solve and then analyse
#     w_tune_run()                  — scan SOR relaxation weights versus iteration count
#     animate()                     — animate solver convergence on a midplane slice
#
#   SECTION: Exam extensions
#     None in the base class — use the standalone exam toolkit below for quick
#                              analysis support during a written exam
#
#   SECTION: I/O & data storage
#     plotting and analysis methods above save their own output data files
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
#   python Poisson.py -N 100 -R monopole --solver gauss_seidel
#   e.g. python Poisson.py -N 100 -R wire --solver sor -w 1.94
# ============================================================

import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numba import njit


@njit
def numba_sor_sweep(w, N, phi, rho, dx, threshold):
    """Accelerated SOR sweep used for relaxation-parameter tuning."""
    dx2 = dx**2
    inv_6 = 1.0 / 6.0
    for it in range(1, 50001):
        diff = 0.0
        for pass_type in range(2):
            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    for k in range(1, N - 1):
                        if (i + j + k) % 2 == pass_type:
                            old_val = phi[i, j, k]
                            neighbor_sum = (
                                phi[i + 1, j, k]
                                + phi[i - 1, j, k]
                                + phi[i, j + 1, k]
                                + phi[i, j - 1, k]
                                + phi[i, j, k + 1]
                                + phi[i, j, k - 1]
                            )
                            new_val = (1.0 - w) * old_val + (w * inv_6) * (
                                neighbor_sum + dx2 * rho[i, j, k]
                            )
                            phi[i, j, k] = new_val
                            diff += (new_val - old_val) ** 2
        final_diff = np.sqrt(diff)
        if final_diff <= threshold:
            return it
    return 50000


class poisson:
    """Finite-difference Poisson solver for monopole and wire source terms."""

    # ── Initialisation ───────────────────────────────────────────

    def __init__(self, N, phi_0, dx, dt, num_iter, threshold, rho, solver, w):
        """Store solver settings and initialise the potential and source arrays.

        Args:
            N (int): Linear grid size for an ``N x N x N`` box.
            phi_0 (float): Baseline initial potential value.
            dx (float): Spatial lattice spacing.
            dt (float): Unused legacy timestep parameter retained for CLI compatibility.
            num_iter (int): Maximum number of iterations to run.
            threshold (float): Convergence threshold for iterative solvers.
            rho (str): Source configuration, ``"monopole"`` or ``"wire"``.
            solver (str): Chosen relaxation scheme.
            w (float): Over-relaxation weight for SOR.

        Returns:
            None: Initialises attributes in place.

        Notes:
            Physics/formula used: solves ``nabla^2 phi = -rho`` in the discrete
            convention encoded by the original update rules.
            ASSUMPTION: Dirichlet boundary conditions are fixed to zero on the box faces.
        """
        self.N = N
        self.dt = dt
        self.dx = dx
        self.phi_0 = phi_0
        self.num_iter = num_iter
        self.threshold = threshold
        self.rho = rho
        self.rho_arg = rho
        self.solver = solver
        self.w = w
        self.phi, self.rho = self.initialize_grid()

    def initialize_grid(self):
        """Generate the initial potential guess and source distribution.

        Args:
            None

        Returns:
            tuple: ``(phi, rho)`` arrays for the simulation volume.

        Notes:
            Physics/formula used: the initial potential is a noisy field with
            zero-valued boundaries, while ``rho`` encodes either a point charge
            or a straight wire source.
            ASSUMPTION: boundary conditions remain fixed throughout the solve.
        """
        noise = 0.01 * (
            np.random.rand(self.N, self.N, self.N) - 0.5
        )  # Small random perturbation seeds the initial relaxation field.
        grid = self.phi_0 + noise
        grid[0, :, :] = 0  # Dirichlet boundary fixes the x-min face potential.
        grid[self.N - 1, :, :] = 0  # Dirichlet boundary fixes the x-max face potential.
        grid[:, 0, :] = 0  # Dirichlet boundary fixes the y-min face potential.
        grid[:, self.N - 1, :] = 0  # Dirichlet boundary fixes the y-max face potential.
        grid[:, :, 0] = 0  # Dirichlet boundary fixes the z-min face potential.
        grid[:, :, self.N - 1] = 0  # Dirichlet boundary fixes the z-max face potential.

        rho = np.zeros_like(grid)
        if self.rho_arg == "monopole":
            rho[self.N // 2, self.N // 2, self.N // 2] = 1  # Central point charge creates a monopole source.
        elif self.rho_arg == "wire":
            rho[self.N // 2, self.N // 2, :] = 1  # Central line source along z creates the wire configuration.
        return grid, rho

    def laplacian(self, f):
        """Compute the 3D Laplacian using finite differences.

        Args:
            f (np.ndarray): Scalar field on the cubic lattice.

        Returns:
            np.ndarray: Discrete Laplacian of ``f``.

        Notes:
            Physics/formula used: six-point central-difference stencil on the
            interior only, with fixed boundaries left untouched.
            ASSUMPTION: boundary values are prescribed and not wrapped.
        """
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1, 1:-1] = (
            f[2:, 1:-1, 1:-1]
            + f[:-2, 1:-1, 1:-1]
            + f[1:-1, 2:, 1:-1]
            + f[1:-1, :-2, 1:-1]
            + f[1:-1, 1:-1, 2:]
            + f[1:-1, 1:-1, :-2]
            - 6 * f[1:-1, 1:-1, 1:-1]
        ) / (
            self.dx**2
        )  # Six-point stencil approximates nabla^2 f on the interior grid.
        return lap

    # ── Core dynamics ────────────────────────────────────────────

    def electric_field(self):
        """Compute the electric field ``E = -grad(phi)``.

        Args:
            None

        Returns:
            np.ndarray: Vector field with components ``(Ex, Ey, Ez)``.

        Notes:
            Physics/formula used: central differences approximate the gradient.
            ASSUMPTION: the field is evaluated on the lattice sites.
        """
        Ex = np.zeros_like(self.phi)
        Ey = np.zeros_like(self.phi)
        Ez = np.zeros_like(self.phi)
        Ex[1:-1, :, :] = -(
            self.phi[2:, :, :] - self.phi[:-2, :, :]
        ) / (2 * self.dx)  # Negative x-gradient gives the electric-field x component.
        Ey[:, 1:-1, :] = -(
            self.phi[:, 2:, :] - self.phi[:, :-2, :]
        ) / (2 * self.dx)  # Negative y-gradient gives the electric-field y component.
        Ez[:, :, 1:-1] = -(
            self.phi[:, :, 2:] - self.phi[:, :, :-2]
        ) / (2 * self.dx)  # Negative z-gradient gives the electric-field z component.
        return np.stack((Ex, Ey, Ez), axis=-1)

    def jacobi(self, f, rho):
        """Compute one Jacobi relaxation update.

        Args:
            f (np.ndarray): Current potential field.
            rho (np.ndarray): Source distribution.

        Returns:
            np.ndarray: Updated potential field after one Jacobi step.

        Notes:
            Physics/formula used: each interior point is replaced by the average
            of its six neighbours plus the source contribution.
            ASSUMPTION: boundaries remain fixed.
        """
        f_new = np.copy(f)
        f_new[1:-1, 1:-1, 1:-1] = (1 / 6.0) * (
            f[2:, 1:-1, 1:-1]
            + f[:-2, 1:-1, 1:-1]
            + f[1:-1, 2:, 1:-1]
            + f[1:-1, :-2, 1:-1]
            + f[1:-1, 1:-1, 2:]
            + f[1:-1, 1:-1, :-2]
            + (self.dx**2) * rho[1:-1, 1:-1, 1:-1]
        )  # Jacobi averages the six neighbours and adds the discrete source term.
        return f_new

    def jacobi_sweep(self):
        """Iterate Jacobi updates until convergence.

        Args:
            None

        Returns:
            None: Updates ``self.phi`` in place.

        Notes:
            Physics/formula used: convergence is monitored with the Euclidean norm
            between successive iterates.
            ASSUMPTION: the original unbounded loop is retained.
        """
        while True:
            self.phi_new = self.jacobi(self.phi, self.rho)
            dist = np.linalg.norm(
                self.phi_new - self.phi
            )  # Norm between iterates measures solver convergence.
            if dist <= self.threshold:
                break
            self.phi = np.copy(self.phi_new)

    def gauss_seidel_step(self, phi_interior, rho_interior, mask_red, mask_black):
        """Perform one red-black Gauss-Seidel step.

        Args:
            phi_interior (np.ndarray): Interior view of the potential field.
            rho_interior (np.ndarray): Interior view of the source field.
            mask_red (np.ndarray): Boolean mask for one checkerboard sublattice.
            mask_black (np.ndarray): Boolean mask for the other checkerboard sublattice.

        Returns:
            None: Updates ``phi_interior`` in place.

        Notes:
            Physics/formula used: red-black ordering enables vectorised
            Gauss-Seidel updates while using newly updated neighbour values.
            ASSUMPTION: the source term and boundaries are fixed during the step.
        """
        neighbor_sum = (
            self.phi[2:, 1:-1, 1:-1]
            + self.phi[:-2, 1:-1, 1:-1]
            + self.phi[1:-1, 2:, 1:-1]
            + self.phi[1:-1, :-2, 1:-1]
            + self.phi[1:-1, 1:-1, 2:]
            + self.phi[1:-1, 1:-1, :-2]
        )
        phi_interior[mask_red] = (
            (1 / 6) * neighbor_sum[mask_red] + self.dx**2 * rho_interior[mask_red]
        )  # Red sites are updated from the latest available neighbours.
        neighbor_sum = (
            self.phi[2:, 1:-1, 1:-1]
            + self.phi[:-2, 1:-1, 1:-1]
            + self.phi[1:-1, 2:, 1:-1]
            + self.phi[1:-1, :-2, 1:-1]
            + self.phi[1:-1, 1:-1, 2:]
            + self.phi[1:-1, 1:-1, :-2]
        )
        phi_interior[mask_black] = (
            (1 / 6) * neighbor_sum[mask_black] + self.dx**2 * rho_interior[mask_black]
        )  # Black sites then use the newly updated red neighbours.

    def gauss_seidel_sweep(self):
        """Iterate red-black Gauss-Seidel updates until convergence.

        Args:
            None

        Returns:
            None: Updates ``self.phi`` in place.

        Notes:
            Physics/formula used: convergence is monitored by the maximum change
            in the potential field after each full red-black pass.
            ASSUMPTION: boundaries are excluded from updates.
        """
        z, y, x = np.indices((self.N - 2, self.N - 2, self.N - 2))
        mask_red = (x + y + z) % 2 == 0
        mask_black = (x + y + z) % 2 != 0
        phi_interior = self.phi[1:-1, 1:-1, 1:-1]
        rho_interior = self.rho[1:-1, 1:-1, 1:-1]

        while True:
            phi_old = np.copy(self.phi)
            self.gauss_seidel_step(phi_interior, rho_interior, mask_red, mask_black)
            diff = np.max(
                np.abs(phi_old - self.phi)
            )  # Max norm between passes measures relaxation convergence.
            if diff <= self.threshold:
                break

    def sor_step(self, w, phi_interior, rho_interior, mask_red, mask_black):
        """Perform one red-black SOR step.

        Args:
            w (float): Relaxation weight.
            phi_interior (np.ndarray): Interior view of the potential field.
            rho_interior (np.ndarray): Interior view of the source field.
            mask_red (np.ndarray): Boolean mask for one checkerboard sublattice.
            mask_black (np.ndarray): Boolean mask for the other checkerboard sublattice.

        Returns:
            None: Updates ``phi_interior`` in place.

        Notes:
            Physics/formula used: SOR mixes the old value with the Gauss-Seidel
            update to accelerate convergence.
            ASSUMPTION: ``1 < w < 2`` is the intended regime for over-relaxation.
        """
        neighbor_sum = (
            self.phi[2:, 1:-1, 1:-1]
            + self.phi[:-2, 1:-1, 1:-1]
            + self.phi[1:-1, 2:, 1:-1]
            + self.phi[1:-1, :-2, 1:-1]
            + self.phi[1:-1, 1:-1, 2:]
            + self.phi[1:-1, 1:-1, :-2]
        )
        phi_interior[mask_red] = (1 - w) * phi_interior[mask_red] + (w / 6) * (
            neighbor_sum[mask_red] + self.dx**2 * rho_interior[mask_red]
        )  # SOR extrapolates beyond the Gauss-Seidel update on red sites.
        neighbor_sum = (
            self.phi[2:, 1:-1, 1:-1]
            + self.phi[:-2, 1:-1, 1:-1]
            + self.phi[1:-1, 2:, 1:-1]
            + self.phi[1:-1, :-2, 1:-1]
            + self.phi[1:-1, 1:-1, 2:]
            + self.phi[1:-1, 1:-1, :-2]
        )
        phi_interior[mask_black] = (1 - w) * phi_interior[mask_black] + (w / 6) * (
            neighbor_sum[mask_black] + self.dx**2 * rho_interior[mask_black]
        )  # SOR extrapolates beyond the Gauss-Seidel update on black sites.

    def sor_sweep(self, w):
        """Iterate SOR updates until convergence.

        Args:
            w (float): Relaxation weight.

        Returns:
            int: Number of iterations required before convergence or cutoff.

        Notes:
            Physics/formula used: convergence is monitored by the maximum interior
            update magnitude.
            ASSUMPTION: the solver is stopped after 50000 iterations if it stalls.
        """
        z, y, x = np.indices((self.N - 2, self.N - 2, self.N - 2))
        mask_red = (x + y + z) % 2 == 0
        mask_black = (x + y + z) % 2 != 0
        phi_interior = self.phi[1:-1, 1:-1, 1:-1]
        rho_interior = self.rho[1:-1, 1:-1, 1:-1]
        iteration = 0

        while True:
            iteration += 1
            phi_old = np.copy(self.phi)
            self.sor_step(w, phi_interior, rho_interior, mask_red, mask_black)
            diff = np.max(
                np.abs(phi_old[1:-1, 1:-1, 1:-1] - self.phi[1:-1, 1:-1, 1:-1])
            )  # Max interior change measures SOR convergence.
            if diff <= self.threshold:
                print(f"w: {w} converged at {iteration} iterations")
                break
            if iteration == 50000:
                print(f"SOR did not converge for w: {w}")
                break
        return iteration

    def magnetic_field(self):
        """Compute the magnetic field from a vector potential ``A = (0, 0, A_z)``.

        Args:
            None

        Returns:
            np.ndarray: Vector field with components ``(Bx, By, 0)``.

        Notes:
            Physics/formula used: ``B = curl(A)``, so only derivatives of ``A_z``
            contribute in the wire geometry.
            ASSUMPTION: ``self.phi`` stores ``A_z`` in the wire case.
        """
        Bx = np.zeros_like(self.phi)
        By = np.zeros_like(self.phi)
        Bx[:, 1:-1, :] = (
            self.phi[:, 2:, :] - self.phi[:, :-2, :]
        ) / (2 * self.dx)  # dA_z/dy gives the x component of the magnetic field.
        By[1:-1, :, :] = -(
            self.phi[2:, :, :] - self.phi[:-2, :, :]
        ) / (2 * self.dx)  # -dA_z/dx gives the y component of the magnetic field.
        return np.stack((Bx, By, np.zeros_like(Bx)), axis=-1)

    def solve(self):
        """Run the selected iterative solver to convergence.

        Args:
            None

        Returns:
            None: Updates ``self.phi`` in place.

        Notes:
            Physics/formula used: dispatches between Jacobi, Gauss-Seidel, and
            SOR without altering the underlying finite-difference equation.
            ASSUMPTION: ``self.solver`` is one of the parser-supported choices.
        """
        solver_dict = {
            "sor": self.sor_sweep,
            "gauss_seidel": self.gauss_seidel_sweep,
            "jacobi": self.jacobi_sweep,
        }
        solver = solver_dict[self.solver]
        if self.solver == "sor":
            solver(w=float(self.w))
        else:
            solver()

    # ── Observables & statistics ─────────────────────────────────

    def magnetic_strength_vs_distance(self):
        """Bin and fit magnetic-field magnitude versus cylindrical radius.

        Args:
            None

        Returns:
            None: Saves data and displays the fitted log-log plot.

        Notes:
            Physics/formula used: for an infinite wire the magnetic field should
            scale as ``|B| ~ r^-1`` away from the core and boundaries.
            ASSUMPTION: radial distance ignores z because the source is a wire.
        """
        field = self.magnetic_field()
        field_mag = np.linalg.norm(
            field, axis=-1
        )  # Magnetic-field magnitude is compared with the theoretical radial scaling.
        x, y, z = np.indices((self.N, self.N, self.N))
        r = np.sqrt((x - self.N // 2) ** 2 + (y - self.N // 2) ** 2) * self.dx
        r_flat = r.flatten()
        B_flat = field_mag.flatten()

        bins = np.linspace(0.5 * self.dx, np.max(r_flat), 50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        B_avg = np.array(
            [
                np.mean(B_flat[(r_flat >= bins[i]) & (r_flat < bins[i + 1])])
                for i in range(len(bins) - 1)
            ]
        )  # Radial binning averages the field on cylindrical shells.

        r_min_fit = 2 * self.dx
        r_max_fit = (self.N // 2) * 0.7 * self.dx
        fit_mask = (
            (bin_centers > 2 * self.dx)
            & (bin_centers < (self.N // 2) * 0.7 * self.dx)
            & (~np.isnan(B_avg))
        )
        coeffs = np.polyfit(
            np.log(bin_centers[fit_mask]), np.log(B_avg[fit_mask]), 1
        )  # Log-log slope estimates the magnetic-field power law.
        print(f"Magnetic Field Slope (Target -1.0): {coeffs[0]:.4f}")

        plt.figure(figsize=(8, 6))
        plt.loglog(bin_centers, B_avg, "o", label="Simulation Data", alpha=0.6)
        plt.loglog(
            bin_centers,
            np.exp(coeffs[1]) * bin_centers ** coeffs[0],
            "r-",
            label=f"Fit (Slope: {coeffs[0]:.2f})",
        )
        ref = bin_centers**-1
        plt.loglog(
            bin_centers,
            ref * B_avg[fit_mask][0] / (bin_centers[fit_mask][0] ** -1),
            "k--",
            label=r"Theoretical $r^{-1}$",
            alpha=0.5,
        )  # Reference curve visualises the ideal wire scaling.
        plt.axvspan(0, r_min_fit, color="gray", alpha=0.1, label="Singularity Region")
        plt.axvspan(r_max_fit, np.max(r_flat), color="red", alpha=0.1, label="Boundary Effects")
        np.savetxt(
            "magnetic_strength_vs_distance.dat",
            np.column_stack((bin_centers, B_avg)),
            header="r |B|",
        )
        plt.xlabel("Distance r")
        plt.ylabel("|B|")
        plt.title("Magnetic Field Strength vs Distance (Wire)")
        plt.legend()
        plt.savefig("Magnetic_vs_distance.png")
        plt.show()

    def field_strength_vs_distance(self):
        """Bin and fit electric-field magnitude versus spherical radius.

        Args:
            None

        Returns:
            None: Saves data and displays the fitted log-log plot.

        Notes:
            Physics/formula used: for a monopole the electric field should scale
            as ``|E| ~ r^-2`` away from the source and boundaries.
            ASSUMPTION: radial distance is measured from the box centre.
        """
        E = self.electric_field()
        field_mag = np.linalg.norm(
            E, axis=-1
        )  # Electric-field magnitude is compared with the theoretical Coulomb scaling.
        x, y, z = np.indices((self.N, self.N, self.N))
        cx = cy = cz = self.N // 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) * self.dx
        r_flat = r.flatten()
        E_flat = field_mag.flatten()

        bins = np.linspace(0.5 * self.dx, np.max(r_flat), 50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        E_avg = []
        for i in range(len(bins) - 1):
            mask = (r_flat >= bins[i]) & (r_flat < bins[i + 1])
            if np.any(mask):
                E_avg.append(np.mean(E_flat[mask]))  # Shell-average field magnitude estimates radial scaling.
            else:
                E_avg.append(np.nan)
        E_avg = np.array(E_avg)

        r_min_fit = 2 * self.dx
        r_max_fit = (self.N // 2) * 0.7 * self.dx
        fit_mask = (bin_centers > r_min_fit) & (bin_centers < r_max_fit) & (~np.isnan(E_avg))
        r_to_fit = bin_centers[fit_mask]
        E_to_fit = E_avg[fit_mask]
        coeffs = np.polyfit(
            np.log(r_to_fit), np.log(E_to_fit), 1
        )  # Log-log slope estimates the electric-field power law.
        print(f"Electric Field Strength Slope (Target -2.0): {coeffs[0]:.4f}")

        plt.figure(figsize=(8, 6))
        plt.loglog(bin_centers, E_avg, "o", label="Binned Simulation Data", alpha=0.6)
        fit_line = np.exp(coeffs[1]) * bin_centers ** coeffs[0]
        plt.loglog(bin_centers, fit_line, "r-", label=f"Fit (Slope: {coeffs[0]:.2f})")
        ref = bin_centers**-2
        plt.loglog(
            bin_centers,
            ref * E_to_fit[0] / (r_to_fit[0] ** -2),
            "k--",
            label=r"Theoretical $r^{-2}$",
            alpha=0.5,
        )  # Reference curve visualises the ideal Coulomb scaling.
        plt.axvspan(0, r_min_fit, color="gray", alpha=0.1, label="Singularity Region")
        plt.axvspan(r_max_fit, np.max(r_flat), color="red", alpha=0.1, label="Boundary Effects")
        np.savetxt(
            "vector_potential_vs_distance.dat",
            np.column_stack((bin_centers, E_avg)),
            header="r E",
        )
        plt.xlabel("Log(Distance r)")
        plt.ylabel("Log(|E|)")
        plt.legend()
        plt.title("Electric Field Strength vs Distance (Monopole)")
        plt.savefig("Electric_vs_distance.png")
        plt.show()

    def vector_potential_vs_distance(self):
        """Bin and fit the wire vector potential versus radius.

        Args:
            None

        Returns:
            None: Saves data and displays the semi-log plot.

        Notes:
            Physics/formula used: for an infinite wire the potential behaves as
            ``A_z ~ m ln(r) + c`` away from the core and boundaries.
            ASSUMPTION: ``self.phi`` stores ``A_z`` in the wire geometry.
        """
        phi_flat = self.phi.flatten()
        x, y, z = np.indices((self.N, self.N, self.N))
        r = np.sqrt((x - self.N // 2) ** 2 + (y - self.N // 2) ** 2) * self.dx
        r_flat = r.flatten()

        bins = np.linspace(0.5 * self.dx, np.max(r_flat), 50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        phi_avg = np.array(
            [
                np.mean(phi_flat[(r_flat >= bins[i]) & (r_flat < bins[i + 1])])
                for i in range(len(bins) - 1)
            ]
        )  # Cylindrical shell averaging isolates the radial dependence of A_z.

        valid_mask = (~np.isnan(phi_avg)) & (bin_centers > 0)
        r_clean = bin_centers[valid_mask]
        phi_clean = phi_avg[valid_mask]
        r_min_fit = 2 * self.dx
        r_max_fit = (self.N // 2) * 0.7 * self.dx
        fit_mask = (r_clean > 2 * self.dx) & (r_clean < (self.N // 2) * 0.7 * self.dx)
        coeffs = np.polyfit(
            np.log(r_clean[fit_mask]), phi_clean[fit_mask], 1
        )  # Semi-log fit extracts the coefficient of ln(r).
        print(f"Vector Potential Semi-log Slope (m in m*ln(r)+c): {coeffs[0]:.4f}")

        plt.figure(figsize=(8, 6))
        plt.semilogx(r_clean, phi_clean, "o", label="Simulation $A_z$", alpha=0.6)
        fit_line = coeffs[0] * np.log(r_clean) + coeffs[1]
        plt.semilogx(
            r_clean,
            fit_line,
            "r-",
            label=f"Fit: {coeffs[0]:.2f} ln(r) + {coeffs[1]:.2f}",
        )
        plt.axvspan(0, r_min_fit, color="gray", alpha=0.1, label="Singularity Region")
        plt.axvspan(r_max_fit, np.max(r_flat), color="red", alpha=0.1, label="Boundary Effects")
        plt.xlabel("Distance r (log scale)")
        plt.ylabel("Vector Potential $A_z$")
        plt.title("Semi-log plot of Vector Potential vs Distance (Wire)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.savefig("vector_potential_vs_distance.png")
        plt.show()

        np.savetxt(
            "vector_potential_vs_distance.dat",
            np.column_stack((r_clean, phi_clean)),
            header="r Az",
        )

    def potential_vs_distance(self):
        """Bin and fit the electric potential versus radius.

        Args:
            None

        Returns:
            None: Saves data and displays the fitted log-log plot.

        Notes:
            Physics/formula used: for a monopole the potential should scale as
            ``phi ~ r^-1`` away from the source and boundaries.
            ASSUMPTION: only positive fitted values are used in the log-log fit.
        """
        phi_flat = self.phi.flatten()
        x, y, z = np.indices((self.N, self.N, self.N))
        r = np.sqrt(
            (x - self.N // 2) ** 2 + (y - self.N // 2) ** 2 + (z - self.N // 2) ** 2
        ) * self.dx
        r_flat = r.flatten()

        bins = np.linspace(0.5 * self.dx, np.max(r_flat), 50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        phi_avg = np.array(
            [
                np.mean(phi_flat[(r_flat >= bins[i]) & (r_flat < bins[i + 1])])
                for i in range(len(bins) - 1)
            ]
        )  # Spherical shell averaging isolates the radial dependence of the potential.
        fit_mask = (
            (bin_centers > 2 * self.dx)
            & (bin_centers < (self.N // 2) * 0.5 * self.dx)
            & (phi_avg > 0)
        )
        coeffs = np.polyfit(
            np.log(bin_centers[fit_mask]), np.log(phi_avg[fit_mask]), 1
        )  # Log-log slope estimates the monopole potential power law.
        print(f"Electric Potential Slope (Target -1.0): {coeffs[0]:.4f}")

        r_min_fit = 2 * self.dx
        r_max_fit = (self.N // 2) * 0.7 * self.dx
        np.savetxt(
            "electric_potential_vs_distance.dat",
            np.column_stack((bin_centers, phi_avg)),
            header="r |V|",
        )
        plt.figure(figsize=(8, 6))
        plt.loglog(bin_centers, phi_avg, "o", label="Simulation Data")
        plt.loglog(
            bin_centers,
            np.exp(coeffs[1]) * bin_centers ** coeffs[0],
            "r-",
            label=f"Fit (Slope: {coeffs[0]:.2f})",
        )
        plt.axvspan(0, r_min_fit, color="gray", alpha=0.1, label="Singularity Region")
        plt.axvspan(r_max_fit, np.max(r_flat), color="red", alpha=0.1, label="Boundary Effects")
        plt.xlabel("Distance r")
        plt.ylabel(r"Potential $\Phi$")
        plt.title("Potential vs Distance (Monopole)")
        plt.legend()
        plt.savefig("potential_vs_distance.png")
        plt.show()

    # ── Visualisation ────────────────────────────────────────────

    def contour_plot(self):
        """Create a contour plot of the monopole midplane potential.

        Args:
            None

        Returns:
            None: Saves data and displays the contour plot.

        Notes:
            Physics/formula used: slices the cubic box through its central x-plane.
            ASSUMPTION: intended for the monopole case.
        """
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(self.phi[self.N // 2, :, :], cmap="viridis")
        plt.colorbar(cp, label=r"Potential $\Phi$")
        plt.xlabel("y-axis index")
        plt.ylabel("z-axis index")
        plt.title(r"Midplane Potential ($\Phi$) for a Monopole at Center")
        plt.savefig("monopole_contour.png")
        np.savetxt("potential_midplane.dat", self.phi[self.N // 2, :, :])
        plt.show()

    def B_field_contour_plot(self):
        """Create a contour plot of the wire midplane potential.

        Args:
            None

        Returns:
            None: Saves data and displays the contour plot.

        Notes:
            Physics/formula used: slices the cubic box through its central z-plane.
            ASSUMPTION: intended for the wire case.
        """
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(self.phi[:, :, self.N // 2], cmap="viridis")
        plt.colorbar(cp, label=r"Vector Potential $A_z$")
        plt.xlabel("x-axis index")
        plt.ylabel("y-axis index")
        plt.title(r"Midplane Potential ($\Phi$) for a wire at Center")
        plt.savefig("magnetic_potential_contour.png")
        np.savetxt("potential_midplane_wire.dat", self.phi[:, :, self.N // 2])
        plt.show()

    def B_field_vector_plot(self):
        """Create a quiver plot of the magnetic field around the wire.

        Args:
            None

        Returns:
            None: Saves data and displays the vector plot.

        Notes:
            Physics/formula used: the magnetic field circulates around the wire,
            so a midplane slice shows the azimuthal pattern.
            ASSUMPTION: intended for the wire case only.
        """
        B = self.magnetic_field()
        Bx_slice = B[:, :, self.N // 2, 0]
        By_slice = B[:, :, self.N // 2, 1]
        skip = 5
        y, z = np.indices((self.N, self.N))
        plt.figure(figsize=(8, 8))
        plt.quiver(
            y[::skip, ::skip],
            z[::skip, ::skip],
            Bx_slice[::skip, ::skip],
            By_slice[::skip, ::skip],
            color="red",
        )  # Quiver arrows display the circulating magnetic-field direction.
        combined = np.column_stack((Bx_slice.flatten(), By_slice.flatten()))
        np.savetxt("magnetic_field_midplane.dat", combined, header="Bx By")
        plt.savefig("magnetic_field_vectors.png")
        plt.show()

    def vector_plot(self):
        """Create a quiver plot of the electric field on a midplane slice.

        Args:
            None

        Returns:
            None: Saves data and displays the vector plot.

        Notes:
            Physics/formula used: the electric field points radially away from
            the monopole source in the central slice.
            ASSUMPTION: intended for the monopole case only.
        """
        E = self.electric_field()
        Ey_slice = E[self.N // 2, :, :, 1]
        Ez_slice = E[self.N // 2, :, :, 2]
        skip = 5
        y, z = np.indices((self.N, self.N))
        plt.figure(figsize=(8, 8))
        plt.quiver(
            y[::skip, ::skip],
            z[::skip, ::skip],
            Ey_slice[::skip, ::skip],
            Ez_slice[::skip, ::skip],
            color="red",
        )  # Quiver arrows display the electric-field direction on the chosen slice.
        plt.title("Electric Field Vectors (Midplane Slice)")
        combined = np.column_stack((Ey_slice.flatten(), Ez_slice.flatten()))
        np.savetxt("vector_field_midplane.dat", combined, header="Ey Ez")
        plt.savefig("electric_field_vectors.png")
        plt.show()

    def analyse(self):
        """Run the appropriate analysis suite for the chosen source type.

        Args:
            None

        Returns:
            None: Produces plots and saved data files.

        Notes:
            Physics/formula used: monopole and wire solutions are analysed with
            different field quantities and theoretical scalings.
            ASSUMPTION: ``self.rho_arg`` matches one of the supported source types.
        """
        if self.rho_arg == "monopole":
            self.vector_plot()
            self.contour_plot()
            self.field_strength_vs_distance()
            self.potential_vs_distance()
        elif self.rho_arg == "wire":
            self.magnetic_strength_vs_distance()
            self.B_field_contour_plot()
            self.B_field_vector_plot()
            self.vector_potential_vs_distance()

    def run(self):
        """Solve the Poisson problem and then run the matching analysis suite.

        Args:
            None

        Returns:
            None: Updates the solution and generates analysis outputs.

        Notes:
            Physics/formula used: separates numerical solving from physical
            post-processing for the two supported source geometries.
            ASSUMPTION: the selected solver converges before analysis.
        """
        self.solve()
        self.analyse()

    def w_tune_run(self):
        """Scan SOR relaxation weights and plot iteration count versus ``w``.

        Args:
            None

        Returns:
            None: Saves the tuning data and displays the plot.

        Notes:
            Physics/formula used: SOR convergence speed depends strongly on the
            relaxation weight, so this scan finds a near-optimal value.
            ASSUMPTION: the tested range is 1.9 to 1.95 as in the original script.
        """
        weights = np.linspace(1.9, 1.95, 50)
        initial_deep = np.copy(self.phi)
        convergences = []
        for w in weights:
            self.phi = np.copy(initial_deep)  # Reset the initial potential so each weight is compared fairly.
            iters = numba_sor_sweep(w, self.N, self.phi, self.rho, self.dx, self.threshold)
            convergences.append(iters)
            print(f"w={w:.3f} converged in {iters} iterations")
        plt.figure(figsize=(8, 8))
        plt.plot(weights, convergences)
        plt.xlabel("w value")
        plt.ylabel("iterations")
        plt.scatter(
            weights[np.argmin(convergences)],
            min(convergences),
        )  # Mark the fastest-converging relaxation weight.
        plt.title("Iterations to Convergence vs w in SOR")
        combined = np.column_stack((weights, convergences))
        np.savetxt("sor_convergences_2.dat", combined, header="weight iterations")
        plt.savefig("sor_convergences_2.png")
        plt.show()

    def animate(self):
        """Animate solver convergence on a central midplane slice.

        Args:
            None

        Returns:
            None: Displays a Matplotlib animation.

        Notes:
            Physics/formula used: repeatedly applies the selected relaxation
            update and displays the central slice of the potential.
            ASSUMPTION: animation is for qualitative solver inspection only.
        """
        self.phi, self.rho = self.initialize_grid()
        z, y, x = np.indices((self.N - 2, self.N - 2, self.N - 2))
        mask_red = (x + y + z) % 2 == 0
        mask_black = (x + y + z) % 2 != 0
        phi_interior = self.phi[1:-1, 1:-1, 1:-1]
        rho_interior = self.rho[1:-1, 1:-1, 1:-1]

        fig, ax = plt.subplots()
        im = ax.imshow(self.phi[self.N // 2, :, :], animated=True, cmap="viridis")
        plt.colorbar(im)
        ax.set_title(f"Solver: {self.solver} (Midplane Slice)")

        def update_frame(_):
            for _ in range(20):
                if self.solver == "sor":
                    self.sor_step(float(self.w), phi_interior, rho_interior, mask_red, mask_black)
                elif self.solver == "gauss_seidel":
                    self.gauss_seidel_step(phi_interior, rho_interior, mask_red, mask_black)
                elif self.solver == "jacobi":
                    self.phi = self.jacobi(self.phi, self.rho)  # Jacobi updates the whole interior from the previous iterate.
            im.set_array(self.phi[self.N // 2, :, :])
            return [im]

        animation.FuncAnimation(
            fig,
            update_frame,
            frames=1000,
            interval=1,
            blit=True,
            repeat_delay=1000,
        )
        plt.show()

    # ── Exam extensions ──────────────────────────────────────────
    # No class methods were requested for this file.


def autocorrelation(data):
    """Compute the normalised autocorrelation function and decorrelation time.

    Args:
        data (array-like): Time series such as solver residuals or sampled potentials.

    Returns:
        tuple:
            ac (np.ndarray): Normalised autocorrelation.
            tau (float): Integrated decorrelation time.

    Notes:
        EXAM: If you sample a residual or field value along an iterative solve,
        call ``ac, tau = autocorrelation(samples)`` and use ``tau`` to justify
        how frequently you should store measurements if the exam asks for
        statistically independent samples.
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
        EXAM: If you compute a fitted slope, average field magnitude, or any
        other scalar from repeated measurements, pass that statistic into
        ``bootstrap_error`` and quote the return value as the uncertainty.
    """
    n = len(data)
    return np.std(
        [
            stat_fn(np.random.choice(data, size=n, replace=True))  # Bootstrap replica of the measured quantity.
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
        EXAM: If the initial guess must be Gaussian rather than uniform, replace
        the current random-noise line with
        ``gaussian_noise((N, N, N), sigma=0.01)`` and state explicitly that you
        used the Box-Muller transform.
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
        is_active_fn (callable): Returns ``True`` while activity remains.
        n_runs (int): Number of independent runs.
        max_steps (int): Maximum number of steps per run.

    Returns:
        np.ndarray: Survival fraction versus time.

    Notes:
        EXAM: This helper is mainly for absorbing-state variants rather than the
        static Poisson equation. If the exam turns the solver into a thresholded
        growth/extinction problem, define an ``is_active_fn`` and use this to
        compare survival curves across parameter values.
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
    parser.add_argument(
        "-R",
        "--rho",
        type=str,
        choices=["monopole", "wire"],
        default="monopole",
        help="Inital charge distribution",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["sor", "gauss_seidel", "jacobi"],
        default="gauss_seidel",
        help="solving algorithm",
    )
    parser.add_argument("-w", default=1.94, type=float, help="weight for successive over relaxtion algorithm")
    parser.add_argument("--animate", action="store_true", help="argument to animate grid")
    parser.add_argument("--sor_iter", action="store_true", help="Iterate over w for sor solver")

    args = parser.parse_args()

    model = poisson(
        N=args.size,
        phi_0=args.phi0,
        dx=args.dx,
        dt=args.dt,
        num_iter=args.num_iter,
        threshold=args.threshold,
        rho=args.rho,
        solver=args.solver,
        w=args.w,
    )
    if args.animate:
        model.animate()
    elif args.sor_iter:
        model.w_tune_run()
    else:
        model.run()
