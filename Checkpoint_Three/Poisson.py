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
        
        # We perform two passes: one for Red, one for Black
        # This ensures the Red-Black SOR logic is mathematically correct
        for pass_type in range(2):
            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    for k in range(1, N - 1):
                        # Red-Black condition: (i + j + k) % 2 == 0 or 1
                        if (i + j + k) % 2 == pass_type:
                            old_val = phi[i, j, k]
                            
                            # Standard 7-point stencil
                            neighbor_sum = (phi[i+1, j, k] + phi[i-1, j, k] +
                                            phi[i, j+1, k] + phi[i, j-1, k] +
                                            phi[i, j, k+1] + phi[i, j, k-1])
                            
                            # SOR update formula
                            # phi_new = (1-w)*phi_old + w/6 * (neighbors + dx^2 * rho)
                            new_val = (1.0 - w) * old_val + \
                                      (w * inv_6) * (neighbor_sum + dx2 * rho[i, j, k])
                            
                            phi[i, j, k] = new_val
                            
                            # Update L2 norm difference (on the fly to save memory)
                            diff += (new_val - old_val)**2
        
        # Calculate final norm
        final_diff = np.sqrt(diff)
        
        if final_diff <= threshold:
            # Note: print() inside @njit works but is slightly slower; 
            # ideally handle reporting outside
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
            # neighbor_sum = (
            #     self.phi[2:,1:-1,1:-1] + self.phi[:-2,1:-1,1:-1] +
            #     self.phi[1:-1,2:,1:-1] + self.phi[1:-1,:-2,1:-1] +
            #     self.phi[1:-1,1:-1,2:] + self.phi[1:-1,1:-1,:-2]
            # )
            # phi_interior[mask_red] = (1/6) * neighbor_sum[mask_red] + self.dx**2 * rho_interior[mask_red]
            # # update neighbors with new reds
            # neighbor_sum = (
            #     self.phi[2:,1:-1,1:-1] + self.phi[:-2,1:-1,1:-1] +
            #     self.phi[1:-1,2:,1:-1] + self.phi[1:-1,:-2,1:-1] +
            #     self.phi[1:-1,1:-1,2:] + self.phi[1:-1,1:-1,:-2]
            # )
            # phi_interior[mask_black] = (1/6) * neighbor_sum[mask_black] + self.dx**2 * rho_interior[mask_black]
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
            # neighbor_sum = (
            #     self.phi[2:,1:-1,1:-1] + self.phi[:-2,1:-1,1:-1] +
            #     self.phi[1:-1,2:,1:-1] + self.phi[1:-1,:-2,1:-1] +
            #     self.phi[1:-1,1:-1,2:] + self.phi[1:-1,1:-1,:-2]
            # )
            # phi_interior[mask_red] = (1- w) * phi_interior[mask_red] + \
            #     (w/6) * (neighbor_sum[mask_red] + self.dx**2 * rho_interior[mask_red])
            # # update neighbors with new reds
            # neighbor_sum = (
            #     self.phi[2:,1:-1,1:-1] + self.phi[:-2,1:-1,1:-1] +
            #     self.phi[1:-1,2:,1:-1] + self.phi[1:-1,:-2,1:-1] +
            #     self.phi[1:-1,1:-1,2:] + self.phi[1:-1,1:-1,:-2]
            # )
            # phi_interior[mask_black] = (1- w) * phi_interior[mask_black] + \
            #     (w/6) * (neighbor_sum[mask_black] + self.dx**2 * rho_interior[mask_black])
        
            # diff = np.linalg.norm(phi_old[1:-1,1:-1,1:-1]-self.phi[1:-1,1:-1,1:-1])
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
        """
        Plot magnetic of magnetic field against distance vs reference
        
        returns:
        None: saves plot to directory
        """
        field = self.magnetic_field()
        field_mag = np.linalg.norm(field, axis=-1)
        x, y, z = np.indices((self.N, self.N, self.N))
        # Distance from the Z-axis (the wire)
        r = np.sqrt((x - self.N//2)**2 + (y - self.N//2)**2) * self.dx
        title = 'Magnetic field strength vs Distance'
        filename = 'Magnetic_vs_distance.png'

        r_flat = r.flatten()
        B_flat = field_mag.flatten()

        mask = r_flat > 0
        r_flat = r_flat[mask]
        B_flat = B_flat[mask]


        # bin data
        bins = np.linspace(0, np.max(r_flat), 50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        B_avg = []


        for i in range(len(bins)-1):
            m = (r_flat >= bins[i]) & (r_flat < bins[i+1])
            if np.any(m):
                B_avg.append(np.mean(B_flat[m]))
            else:
                B_avg.append(np.nan)

        B_avg = np.array(B_avg)

        # plot
        plt.figure()
        plt.loglog(bin_centers, B_avg, 'o', label='Simulation')

        # reference r^-1
        ref = bin_centers**-1
        plt.loglog(bin_centers, ref * B_avg[1]/ref[1], '--', label=r'$r^{-1}$')

        plt.xlabel('Distance r')
        plt.ylabel('|B|')
        plt.legend()
        plt.title(title)
        plt.savefig(filename)
        plt.show()

        # save data
        np.savetxt('magnetic_field_vs_distance.dat',
                np.column_stack((bin_centers, B_avg)),
                header='r |E|')

    def field_strength_vs_distance(self):
        """
        compute and plot the field strength of |E| as function of distance

        returns:
        None: saves file to directory
        """
        E = self.electric_field()
        field_mag = np.linalg.norm(E,axis=-1)

        x, y, z = np.indices((self.N, self.N, self.N))
        cx = cy = cz = self.N // 2

        r = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2) * self.dx
        title = 'Electric field strength vs Distance'
        filename = 'Electric_vs_distance.png'

        r_flat = r.flatten()
        E_flat = field_mag.flatten()

        mask = r_flat > 0
        r_flat = r_flat[mask]
        E_flat = E_flat[mask]


        # bin data
        bins = np.linspace(0, np.max(r_flat), 50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        E_avg = []


        for i in range(len(bins)-1):
            m = (r_flat >= bins[i]) & (r_flat < bins[i+1])
            if np.any(m):
                E_avg.append(np.mean(E_flat[m]))
            else:
                E_avg.append(np.nan)

        E_avg = np.array(E_avg)

        # plot
        plt.figure()
        plt.loglog(bin_centers, E_avg, 'o', label='Simulation')

        # reference r^-2
        ref = bin_centers**-2
        plt.loglog(bin_centers, ref * E_avg[1]/ref[1], '--', label=r'$r^{-2}$')

        plt.xlabel('Distance r')
        plt.ylabel('|E|')
        plt.legend()
        plt.title(title)
        plt.savefig(filename)
        plt.show()

        # save data
        np.savetxt('field_vs_distance.dat',
                np.column_stack((bin_centers, E_avg)),
                header='r |E|')
        
    def vector_potential_vs_distance(self):
        """
        compute and plot the potential as function of distance

        returns:
        None: saves file to directory
        """
        phi_flat = self.phi.flatten()

        x, y,z = np.indices((self.N, self.N,self.N))
        cx = cy = self.N // 2

        r = np.sqrt((x - cx)**2 + (y - cy)**2) * self.dx # distance from z axis

        r_flat = r.flatten()

        mask = r_flat > 0
        r_flat = r_flat[mask]
        phi_flat = phi_flat[mask]


        # bin data
        bins = np.linspace(0, np.max(r_flat), 50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        phi_avg = []


        for i in range(len(bins)-1):
            m = (r_flat >= bins[i]) & (r_flat < bins[i+1])
            if np.any(m):
                phi_avg.append(np.mean(phi_flat[m]))
            else:
                phi_avg.append(np.nan)

        phi_avg = np.array(phi_avg)

        # plot
        plt.figure()
        plt.loglog(bin_centers, phi_avg, 'o', label='Simulation')

        # reference r^-2
        ref = bin_centers**-1
        plt.loglog(bin_centers, ref * phi_avg[1]/ref[1], '--', label=r'$r^{-1}$')

        plt.xlabel('Distance r')
        plt.ylabel('|V|')
        plt.legend()
        plt.title('Vector Potential vs Distance')
        plt.savefig('vector_potential_vs_distance.png')
        plt.show()

        # save data
        np.savetxt('vector_potential_vs_distance.dat',
                np.column_stack((bin_centers, phi_avg)),
                header='r |V|')

    def potential_vs_distance(self):
        """
        compute and plot the potential as function of distance

        returns:
        None: saves file to directory
        """
        phi_flat = self.phi.flatten()

        x, y, z = np.indices((self.N, self.N, self.N))
        cx = cy = cz = self.N // 2

        r = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2) * self.dx

        r_flat = r.flatten()

        mask = r_flat > 0
        r_flat = r_flat[mask]
        phi_flat = phi_flat[mask]


        # bin data
        bins = np.linspace(0, np.max(r_flat), 50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        phi_avg = []


        for i in range(len(bins)-1):
            m = (r_flat >= bins[i]) & (r_flat < bins[i+1])
            if np.any(m):
                phi_avg.append(np.mean(phi_flat[m]))
            else:
                phi_avg.append(np.nan)

        phi_avg = np.array(phi_avg)

        # plot
        plt.figure()
        plt.loglog(bin_centers, phi_avg, 'o', label='Simulation')

        # reference r^-2
        ref = bin_centers**-1
        plt.loglog(bin_centers, ref * phi_avg[1]/ref[1], '--', label=r'$r^{-1}$')

        plt.xlabel('Distance r')
        plt.ylabel('|V|')
        plt.legend()
        plt.title('Potential vs Distance')
        plt.savefig('potential_vs_distance.png')
        plt.show()

        # save data
        np.savetxt('potential_vs_distance.dat',
                np.column_stack((bin_centers, phi_avg)),
                header='r |V|')

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
