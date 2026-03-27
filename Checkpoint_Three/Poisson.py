"""
Numerical solve partial differential equation for and poisson equation
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import argparse
from scipy.signal import convolve2d
from collections import deque

from numba import njit

@njit
def numba_sor_sweep(w, N, phi, rho, dx, threshold):
    """
    Alternative to sor_sweep method in class which uses numba to optimise time

    returns:
    it: number of iterations until convergence
    """
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
                            
                            neighbor_sum = (phi[i+1, j, k] + phi[i-1, j, k] +
                                            phi[i, j+1, k] + phi[i, j-1, k] +
                                            phi[i, j, k+1] + phi[i, j, k-1])
                            
                            new_val = (1.0 - w) * old_val + \
                                      (w * inv_6) * (neighbor_sum + dx2 * rho[i, j, k])
                            
                            phi[i, j, k] = new_val
                            
                            diff += (new_val - old_val)**2
        
        final_diff = np.sqrt(diff)
        
        if final_diff <= threshold:
            return it

    return 50000



class poisson:

    def __init__(self,N,phi_0,dx,dt,num_iter,threshold,rho,solver,w):
        """
        N: grid length
        phi_0: average composition
        dx: length step
        dt: time step
        num iter: max number of run iterations
        threshold: threshold value for determining convergence
        rho: string to characterise charge distribution
        solver: algorithm to use for grid updates
        w: weight for SOR algorithm
        """
        self.N = N
        self.dt = dt
        self.dx =dx 
        self.phi_0 = phi_0
        self.num_iter = num_iter
        self.threshold = threshold
        self.rho = rho
        self.rho_arg = rho
        self.solver = solver
        self.w=w

        self.phi, self.rho = self.initialize_grid()

    def initialize_grid(self):
        """
        generate inital grid of random states with 0 boundary
        

        returns:
        grid: N x N x N numpy array of states
        """

        noise = 0.01 * (np.random.rand(self.N, self.N,self.N) - 0.5)
        grid = self.phi_0 + noise
        grid[0,:,:] = 0
        grid[self.N-1,:,:]=0
        grid[:,0,:] = 0
        grid[:,self.N-1,:]=0
        grid[:,:,0] = 0
        grid[:,:,self.N-1]=0

        rho = np.zeros_like(grid)
        if self.rho_arg == 'monopole':
                rho[self.N//2,self.N//2,self.N//2] = 1
        elif self.rho_arg == 'wire':
                rho[self.N//2,self.N//2,:] = 1
        return grid, rho
    
    def laplacian(self,f):
        """
        Calculates the 3D Laplacian using finite differences 
        and periodic boundary conditions.

        Avoid np.roll due to set boundary conditions

        returns:
        Laplacian of f
        """
        # Shift indices: left, right, up, down
        lap = np.zeros_like(f)
        lap[1:-1,1:-1,1:-1] = (
            f[2:,1:-1,1:-1] + f[:-2,1:-1,1:-1] +
            f[1:-1,2:,1:-1] + f[1:-1,:-2,1:-1] +
            f[1:-1,1:-1,2:] + f[1:-1,1:-1,:-2] -
            6*f[1:-1,1:-1,1:-1]
        ) / self.dx**2
        return lap
    
    def electric_field(self):
        """
        calulate the elctric field of the grid

        returns:
        E: array of (Ex,Ey,Ez) at each point
        """
        Ex = np.zeros_like(self.phi)
        Ey = np.zeros_like(self.phi)
        Ez = np.zeros_like(self.phi)

        Ex[1:-1,:,:] = -(self.phi[2:,:,:] - self.phi[:-2,:,:]) / (2*self.dx)
        Ey[:,1:-1,:] = -(self.phi[:,2:,:] - self.phi[:,:-2,:]) / (2*self.dx)
        Ez[:,:,1:-1] = -(self.phi[:,:,2:] - self.phi[:,:,:-2]) / (2*self.dx)

        return np.stack((Ex, Ey, Ez), axis=-1)
        
    def jacobi(self,f,rho):
        """
        update algortihm via jacobi

        returns:
        f_new: updated grid using jacobi algorithm
        """
        f_new = np.copy(f)

        f_new[1:-1, 1:-1, 1:-1] = (1/6.0) * (
                f[2:, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1] +  # x-neighbors
                f[1:-1, 2:, 1:-1] + f[1:-1, :-2, 1:-1] +  # y-neighbors
                f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, :-2] +  # z-neighbors
                (self.dx**2) * rho[1:-1, 1:-1, 1:-1]
            )
        
        return f_new
        
    def jacobi_sweep(self):
        """
        iterate using jacobi algorithm until convergence
        
        returns:
        None: updates grid in place
        """
        while True:
            self.phi_new = self.jacobi(self.phi,self.rho)
            dist = np.linalg.norm(self.phi_new - self.phi)
            if dist <= self.threshold:
                break
            self.phi = np.copy(self.phi_new)

    def gauss_seidel_step(self,phi_interior,rho_interior,mask_red,mask_black):
        """
        Single gauss seidel step to make animation easier

        returns:
        None: updates phi in place
        """
        neighbor_sum = (
            self.phi[2:,1:-1,1:-1] + self.phi[:-2,1:-1,1:-1] +
            self.phi[1:-1,2:,1:-1] + self.phi[1:-1,:-2,1:-1] +
            self.phi[1:-1,1:-1,2:] + self.phi[1:-1,1:-1,:-2]
        )
        phi_interior[mask_red] = (1/6) * neighbor_sum[mask_red] + self.dx**2 * rho_interior[mask_red]
        # update neighbors with new reds
        neighbor_sum = (
            self.phi[2:,1:-1,1:-1] + self.phi[:-2,1:-1,1:-1] +
            self.phi[1:-1,2:,1:-1] + self.phi[1:-1,:-2,1:-1] +
            self.phi[1:-1,1:-1,2:] + self.phi[1:-1,1:-1,:-2]
        )
        phi_interior[mask_black] = (1/6) * neighbor_sum[mask_black] + self.dx**2 * rho_interior[mask_black]    

    def gauss_seidel_sweep(self):
        """
        iterate using gauss_seidel algorithm until convergence

        uses red-black selection algorithm for optimisation
        
        returns:
        None: updates grid in place
        """
        z, y, x = np.indices((self.N-2, self.N-2, self.N-2))
        mask_red = ((x + y + z) % 2 == 0)
        mask_black = ((x + y + z) % 2 != 0)

        #exclude boundaries
        phi_interior = self.phi[1:-1, 1:-1, 1:-1]
        rho_interior = self.rho[1:-1, 1:-1, 1:-1]

        while True:
            phi_old = np.copy(self.phi)
            self.gauss_seidel_step(phi_interior,rho_interior,mask_red,mask_black)
            diff = np.max(np.abs(phi_old - self.phi))
            if diff <= self.threshold:
                break

    def sor_step(self,w,phi_interior,rho_interior,mask_red,mask_black):
        """
        Single sor step to make animation easier

        returns:
        None: updates phi in place
        """
        neighbor_sum = (
            self.phi[2:,1:-1,1:-1] + self.phi[:-2,1:-1,1:-1] +
            self.phi[1:-1,2:,1:-1] + self.phi[1:-1,:-2,1:-1] +
            self.phi[1:-1,1:-1,2:] + self.phi[1:-1,1:-1,:-2]
        )
        phi_interior[mask_red] = (1- w) * phi_interior[mask_red] + \
            (w/6) * (neighbor_sum[mask_red] + self.dx**2 * rho_interior[mask_red])
        # update neighbors with new reds
        neighbor_sum = (
            self.phi[2:,1:-1,1:-1] + self.phi[:-2,1:-1,1:-1] +
            self.phi[1:-1,2:,1:-1] + self.phi[1:-1,:-2,1:-1] +
            self.phi[1:-1,1:-1,2:] + self.phi[1:-1,1:-1,:-2]
        )
        phi_interior[mask_black] = (1- w) * phi_interior[mask_black] + \
            (w/6) * (neighbor_sum[mask_black] + self.dx**2 * rho_interior[mask_black])

    def sor_sweep(self,w):
        """
        iterate using successive over relaxation algorithm until convergence

        uses red-black selection algorithm for optimisation

        returns:
        iter: number of iterations until convergence
        """

        z, y, x = np.indices((self.N-2, self.N-2, self.N-2))
        mask_red = ((x + y + z) % 2 == 0)
        mask_black = ((x + y + z) % 2 != 0)

        #exclude boundaries
        phi_interior = self.phi[1:-1, 1:-1, 1:-1]
        rho_interior = self.rho[1:-1, 1:-1, 1:-1]

        iter = 0

        while True:
            iter += 1
            phi_old = np.copy(self.phi)
            self.sor_step(w,phi_interior,rho_interior,mask_red,mask_black)
            # neighbor_sum = (1,1:-1]-self.phi[1:-1,1:-1,1:-1])
            diff = np.max(np.abs(phi_old[1:-1,1:-1,1:-1] - self.phi[1:-1,1:-1,1:-1]))
            if diff <= self.threshold:
                print(f'w: {w} converged at {iter} iterations')
                break
            if iter == 50000:
                print(f'SOR did not converge for w: {w}')
                break

        return iter

        
    def magnetic_field(self):
        """
        Calculate B = curl(A) given A = (0,0,Az). Therefore Bz = 0, only consider Bx,bY

        returns:
        np.array(Bx,By,0): np array of magetic field vectors
        """

        Bx = np.zeros_like(self.phi)
        By = np.zeros_like(self.phi)

        # dAz/dy
        Bx[:, 1:-1, :] = (self.phi[:, 2:, :] - self.phi[:, :-2, :]) / (2*self.dx)
        # -dAz/dx
        By[1:-1, :, :] = -(self.phi[2:, :, :] - self.phi[:-2, :, :]) / (2*self.dx)

        return np.stack((Bx, By, np.zeros_like(Bx)), axis=-1)

    def contour_plot(self):
        """
        Contour plot of midplace potential

        returns:
        None: Saves figures to file
        """
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(self.phi[self.N//2, :, :], cmap='viridis')
        plt.colorbar(cp, label=r'Potential $\Phi$')
        plt.xlabel('y-axis index')
        plt.ylabel('z-axis index')
        plt.title(r'Midplane Potential ($\Phi$) for a Monopole at Center')
        plt.savefig('monopole_contour.png')
        np.savetxt('potential_midplane.dat', self.phi[self.N//2, :, :])
        plt.show()

    def B_field_contour_plot(self):
        """
        Contour plot of midplane vector potential

        returns:
        None: Saves figures to file
        """
        B = self.magnetic_field()
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(self.phi[:, :, self.N//2], cmap='viridis')
        plt.colorbar(cp, label=r'Vector Potential $A_z$')
        plt.xlabel('x-axis index')
        plt.ylabel('y-axis index')
        plt.title(r'Midplane Potential ($\Phi$) for a wire at Center')
        plt.savefig('magnetic_potential_contour.png')
        np.savetxt('potential_midplane_wire.dat', self.phi[:, :, self.N//2])
        plt.show()

    def B_field_vector_plot(self):
        """
        Vector plot of B field
        
        returns:
        None: saves file to directory
        """
        B = self.magnetic_field()
        # Take a slice perpendicular to the wire (constant z)
        Bx_slice = B[:, :, self.N//2, 0]
        By_slice = B[:, :, self.N//2, 1]
        title = 'Magnetic Field Vectors (B) around Wire'
        filename = 'magnetic_field_vectors.png'
        skip = 5
        y, z = np.indices((self.N, self.N))
        plt.figure(figsize=(8, 8))
        plt.quiver(y[::skip, ::skip], z[::skip, ::skip], 
        Bx_slice[::skip, ::skip], By_slice[::skip, ::skip], 
        color='red')
        combined = np.column_stack((Bx_slice.flatten(), By_slice.flatten()))
        np.savetxt('magnetic_field_midplane.dat', combined, header='Bx By')
        plt.savefig('magnetic_field_vectors.png')
        plt.show()
        
    def vector_plot(self):
        """
        Vectot plot of electric field

        returns:
        None: saves plot to directory
        """
        E = self.electric_field()

        Ey_slice = E[self.N//2, :, :, 1]
        Ez_slice = E[self.N//2, :, :, 2]

        skip = 5
        y, z = np.indices((self.N, self.N))
        plt.figure(figsize=(8, 8))
        plt.quiver(y[::skip, ::skip], z[::skip, ::skip], 
                Ey_slice[::skip, ::skip], Ez_slice[::skip, ::skip], 
                color='red')
        
        plt.title('Electric Field Vectors (Midplane Slice)')
        combined = np.column_stack((Ey_slice.flatten(), Ez_slice.flatten()))
        np.savetxt('vector_field_midplane.dat', combined, header='Ey Ez')
        plt.savefig('electric_field_vectors.png')
        
        plt.show()

    def magnetic_strength_vs_distance(self):
            field = self.magnetic_field()
            field_mag = np.linalg.norm(field, axis=-1)
            x, y, z = np.indices((self.N, self.N, self.N))
            
            r = np.sqrt((x - self.N//2)**2 + (y - self.N//2)**2) * self.dx
            r_flat, B_flat = r.flatten(), field_mag.flatten()

            bins = np.linspace(0.5 * self.dx, np.max(r_flat), 50)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            B_avg = np.array([np.mean(B_flat[(r_flat >= bins[i]) & (r_flat < bins[i+1])]) for i in range(len(bins)-1)])

            r_min_fit = 2 * self.dx
            r_max_fit = (self.N // 2) * 0.7 * self.dx

            # Masking: Target the middle 10% to 70% of the box radius
            fit_mask = (bin_centers > 2 * self.dx) & (bin_centers < (self.N // 2) * 0.7 * self.dx) & (~np.isnan(B_avg))
            coeffs = np.polyfit(np.log(bin_centers[fit_mask]), np.log(B_avg[fit_mask]), 1)
            print(f"Magnetic Field Slope (Target -1.0): {coeffs[0]:.4f}")

            plt.figure(figsize=(8, 6))
            plt.loglog(bin_centers, B_avg, 'o', label='Simulation Data', alpha=0.6)
            plt.loglog(bin_centers, np.exp(coeffs[1]) * bin_centers**coeffs[0], 'r-', label=f'Fit (Slope: {coeffs[0]:.2f})')
            
            # Reference r^-1
            ref = bin_centers**-1
            plt.loglog(bin_centers, ref * B_avg[fit_mask][0]/(bin_centers[fit_mask][0]**-1), 'k--', label=r'Theoretical $r^{-1}$', alpha=0.5)

            plt.axvspan(0, r_min_fit, color='gray', alpha=0.1, label='Singularity Region')
            plt.axvspan(r_max_fit, np.max(r_flat), color='red', alpha=0.1, label='Boundary Effects')
            np.savetxt('magnetic_strength_vs_distance.dat',
                    np.column_stack((bin_centers, B_avg)),
                    header='r |B|')

            plt.xlabel('Distance r')
            plt.ylabel('|B|')
            plt.title('Magnetic Field Strength vs Distance (Wire)')
            plt.legend()
            plt.savefig('Magnetic_vs_distance.png')
            plt.show()

    def field_strength_vs_distance(self):
        E = self.electric_field()
        field_mag = np.linalg.norm(E, axis=-1)

        x, y, z = np.indices((self.N, self.N, self.N))
        cx = cy = cz = self.N // 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2) * self.dx

        r_flat = r.flatten()
        E_flat = field_mag.flatten()

        bins = np.linspace(0.5 * self.dx, np.max(r_flat), 50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        E_avg = []

        for i in range(len(bins)-1):
            m = (r_flat >= bins[i]) & (r_flat < bins[i+1])
            if np.any(m):
                E_avg.append(np.mean(E_flat[m]))
            else:
                E_avg.append(np.nan)

        E_avg = np.array(E_avg)

        r_min_fit = 2 * self.dx
        r_max_fit = (self.N // 2) * 0.7 * self.dx
        
        fit_mask = (bin_centers > r_min_fit) & (bin_centers < r_max_fit) & (~np.isnan(E_avg))
        
        r_to_fit = bin_centers[fit_mask]
        E_to_fit = E_avg[fit_mask]

        coeffs = np.polyfit(np.log(r_to_fit), np.log(E_to_fit), 1)
        print(f"Electric Field Strength Slope (Target -2.0): {coeffs[0]:.4f}")

        plt.figure(figsize=(8, 6))
        plt.loglog(bin_centers, E_avg, 'o', label='Binned Simulation Data', alpha=0.6)
        
        fit_line = np.exp(coeffs[1]) * bin_centers**coeffs[0]
        plt.loglog(bin_centers, fit_line, 'r-', label=f'Fit (Slope: {coeffs[0]:.2f})')
        
        # Reference r^-2
        ref = bin_centers**-2
        plt.loglog(bin_centers, ref * E_to_fit[0]/(r_to_fit[0]**-2), 'k--', label=r'Theoretical $r^{-2}$', alpha=0.5)

        plt.axvspan(0, r_min_fit, color='gray', alpha=0.1, label='Singularity Region')
        plt.axvspan(r_max_fit, np.max(r_flat), color='red', alpha=0.1, label='Boundary Effects')

        np.savetxt('vector_potential_vs_distance.dat',
                np.column_stack((bin_centers, E_avg)),
                header='r E')

        plt.xlabel('Log(Distance r)')
        plt.ylabel('Log(|E|)')
        plt.legend()
        plt.title('Electric Field Strength vs Distance (Monopole)')
        plt.savefig('Electric_vs_distance.png')
        plt.show()
        
    def vector_potential_vs_distance(self):
            """
            Compute and plot Az vs ln(r). 
            For a wire, Az should be proportional to ln(r).
            """
            phi_flat = self.phi.flatten()
            x, y, z = np.indices((self.N, self.N, self.N))
            
            r = np.sqrt((x - self.N//2)**2 + (y - self.N//2)**2) * self.dx
            r_flat = r.flatten()

            bins = np.linspace(0.5 * self.dx, np.max(r_flat), 50)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            phi_avg = np.array([
                np.mean(phi_flat[(r_flat >= bins[i]) & (r_flat < bins[i+1])]) 
                for i in range(len(bins)-1)
            ])

            valid_mask = (~np.isnan(phi_avg)) & (bin_centers > 0)
            r_clean = bin_centers[valid_mask]
            phi_clean = phi_avg[valid_mask]

            r_min_fit = 2 * self.dx
            r_max_fit = (self.N // 2) * 0.7 * self.dx

            fit_mask = (r_clean > 2 * self.dx) & (r_clean < (self.N // 2) * 0.7 * self.dx)
            
            coeffs = np.polyfit(np.log(r_clean[fit_mask]), phi_clean[fit_mask], 1)
            
            print(f"Vector Potential Semi-log Slope (m in m*ln(r)+c): {coeffs[0]:.4f}")

            plt.figure(figsize=(8, 6))
            
            plt.semilogx(r_clean, phi_clean, 'o', label='Simulation $A_z$', alpha=0.6)
            
            fit_line = coeffs[0] * np.log(r_clean) + coeffs[1]
            plt.semilogx(r_clean, fit_line, 'r-', label=f'Fit: {coeffs[0]:.2f} ln(r) + {coeffs[1]:.2f}')

            plt.axvspan(0, r_min_fit, color='gray', alpha=0.1, label='Singularity Region')
            plt.axvspan(r_max_fit, np.max(r_flat), color='red', alpha=0.1, label='Boundary Effects')

            plt.xlabel('Distance r (log scale)')
            plt.ylabel('Vector Potential $A_z$')
            plt.title('Semi-log plot of Vector Potential vs Distance (Wire)')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.legend()
            
            plt.savefig('vector_potential_vs_distance.png')
            plt.show()

            np.savetxt('vector_potential_vs_distance.dat',
                    np.column_stack((r_clean, phi_clean)),
                    header='r Az')

    def potential_vs_distance(self):
            phi_flat = self.phi.flatten()
            x, y, z = np.indices((self.N, self.N, self.N))
            r = np.sqrt((x - self.N//2)**2 + (y - self.N//2)**2 + (z - self.N//2)**2) * self.dx
            r_flat = r.flatten()

            bins = np.linspace(0.5 * self.dx, np.max(r_flat), 50)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            phi_avg = np.array([np.mean(phi_flat[(r_flat >= bins[i]) & (r_flat < bins[i+1])]) for i in range(len(bins)-1)])

            # Apply mask to avoid boundary dip
            fit_mask = (bin_centers > 2 * self.dx) & (bin_centers < (self.N // 2) * 0.5 * self.dx) & (phi_avg > 0)
            coeffs = np.polyfit(np.log(bin_centers[fit_mask]), np.log(phi_avg[fit_mask]), 1)
            print(f"Electric Potential Slope (Target -1.0): {coeffs[0]:.4f}")

            r_min_fit = 2 * self.dx
            r_max_fit = (self.N // 2) * 0.7 * self.dx

            np.savetxt('electric_potential_vs_distance.dat',
                    np.column_stack((bin_centers, phi_avg)),
                    header='r |V|')

            plt.figure(figsize=(8, 6))
            plt.loglog(bin_centers, phi_avg, 'o', label='Simulation Data')
            plt.loglog(bin_centers, np.exp(coeffs[1]) * bin_centers**coeffs[0], 'r-', label=f'Fit (Slope: {coeffs[0]:.2f})')

            plt.axvspan(0, r_min_fit, color='gray', alpha=0.1, label='Singularity Region')
            plt.axvspan(r_max_fit, np.max(r_flat), color='red', alpha=0.1, label='Boundary Effects')
            plt.xlabel('Distance r')
            plt.ylabel(r'Potential $\Phi$')
            plt.title('Potential vs Distance (Monopole)')
            plt.legend()
            plt.savefig('potential_vs_distance.png')
            plt.show()

    def solve(self):
        """
        Run a called solver until convergence

        returns:
        None: updates grid in place
        """
        solver_dict = {
            'sor': self.sor_sweep,
            'gauss_seidel': self.gauss_seidel_sweep,
            'jacobi': self.jacobi_sweep
        }
        solver = solver_dict[self.solver]
        if self.solver == 'sor':
            solver(w=float(self.w))
        else:
            solver()

    def analyse(self):
        """
        Calls analysis methods on current plot

        returns:
        None: saves plots as per respective method
        """
        if self.rho_arg == 'monopole':
            self.vector_plot()
            self.contour_plot()
            self.field_strength_vs_distance()
            self.potential_vs_distance()
        elif self.rho_arg == 'wire':
            self.magnetic_strength_vs_distance()
            self.B_field_contour_plot()
            self.B_field_vector_plot()
            self.vector_potential_vs_distance()

    def run(self):
        """
        Solve then analyse
        """
        self.solve()
        self.analyse()

    def w_tune_run(self):
        """
        Iterate through w hyperparameter until convergence for sor solver

        returns:
        None: plots w vs iterations plot and saves to figure
        """
        weights = np.linspace(1.9, 1.95, 50)
        initial_deep = np.copy(self.phi)  # store initial state so each run uses the same
        convergences = []
        for w in weights:
            self.phi = np.copy(initial_deep) #reset to initial state
            iters = numba_sor_sweep(w, self.N, self.phi, self.rho, self.dx, self.threshold)
            convergences.append(iters)
            print(f"w={w:.3f} converged in {iters} iterations")
        plt.figure(figsize=(8, 8))
        plt.plot(weights,convergences)
        plt.xlabel('w value')
        plt.ylabel('iterations')
        plt.scatter(weights[np.argmin(convergences)],min(convergences))
        
        plt.title('Iterations to Convergence vs w in SOR')
        combined = np.column_stack((weights, convergences))
        np.savetxt('sor_convergences_2.dat', combined, header='weight iterations')
        plt.savefig('sor_convergences_2.png')
        
        plt.show()


    def animate(self):
        """
        Animate the evolution of the composition grid over time

        returns:
        None: displays an animation of the grid evolution
        """
        self.phi,self.rho = self.initialize_grid()

        solver_dict = {
            'sor': self.sor_step,
            'gauss_seidel': self.gauss_seidel_step,
            'jacobi': self.jacobi
        }

        solver = solver_dict[self.solver]
        z, y, x = np.indices((self.N-2, self.N-2, self.N-2))
        mask_red = ((x + y + z) % 2 == 0)
        mask_black = ((x + y + z) % 2 != 0)

        #exclude boundaries
        phi_interior = self.phi[1:-1, 1:-1, 1:-1]
        rho_interior = self.rho[1:-1, 1:-1, 1:-1]

        fig, ax = plt.subplots()
        im = ax.imshow(self.phi[self.N//2, :, :], animated=True, cmap='viridis')
        plt.colorbar(im)
        ax.set_title(f"Solver: {self.solver} (Midplane Slice)")

        def update_frame(frame):
            # Perform 20 iterations per frame to make the animation faster
            for _ in range(20):
                if self.solver == 'sor':
                    self.sor_step(float(self.w), phi_interior, rho_interior, mask_red, mask_black)
                elif self.solver == 'gauss_seidel':
                    self.gauss_seidel_step(phi_interior, rho_interior, mask_red, mask_black)
                elif self.solver == 'jacobi':
                    self.phi = self.jacobi(self.phi, self.rho)
            
            # Update the image with the new slice
            im.set_array(self.phi[self.N//2, :, :])
            return [im]

        ani = animation.FuncAnimation(fig, update_frame, frames=1000, interval=1, blit=True, repeat_delay=1000)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cahn Hilliard Equation')
    parser.add_argument('-N','--size', type=int, default=100, help='Size of the lattice (N x N)')
    parser.add_argument('--phi0',type=float,default=0,help='Average composition of grid')
    parser.add_argument('-dx',type=float,default=1,help= 'Length step')
    parser.add_argument('-dt',type=float,default=1e-4,help= 'Time step')
    parser.add_argument('--num_iter',type=int,default=50000,help='Number of Iteration to Run')
    parser.add_argument('--threshold',type=float,default=1e-6,help='Change thershold to determine equilibrium')
    parser.add_argument('-R','--rho',type=str,choices=['monopole','wire'],default='monopole',help='Inital charge distribution')
    parser.add_argument('--solver',type=str,choices=['sor','gauss_seidel','jacobi'],default='gauss_seidel', help = 'solving algorithm')
    parser.add_argument('-w',default=1.94,type=float,help='weight for successive over relaxtion algorithm')
    parser.add_argument('--animate',action='store_true',help='argument to animate grid')
    parser.add_argument('--sor_iter',action='store_true',help='Iterate over w for sor solver')

    args = parser.parse_args()

    model = poisson(
        N = args.size,
        phi_0 = args.phi0,
        dx = args.dx,
        dt = args.dt,
        num_iter=args.num_iter,
        threshold = args.threshold,
        rho=args.rho,
        solver=args.solver,
        w=args.w
    )
    if args.animate:
        model.animate()
    elif args.sor_iter:
        model.w_tune_run()
    else:
        model.run()
