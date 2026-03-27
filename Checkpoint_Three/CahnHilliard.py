"""
Numerical solve partial differential equation for Cahn-Hilliard equation
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import argparse

class cahn_hilliard:

    def __init__(self,N,phi_0,dx,dt,num_iter,threshold):
        """
        N: grid length
        phi_0: average composition
        dx: length step
        dt: time step
        threshold: float value to determine if equlibrium has reached in energy change
        """
        self.N = N
        self.dt = dt
        self.dx =dx 
        self.phi_0 = phi_0
        self.num_iter = num_iter
        self.threshold = threshold

        self.phi = self.initialize_grid()

    def initialize_grid(self):
        """
        generate inital grid of average composition with random noise
        

        returns:
        grid: N x N numpy array of states
        """

        noise = 0.01 * (np.random.rand(self.N, self.N) - 0.5)
        return self.phi_0 + noise
    
    def laplacian(self,f):
        """
        Calculates the 2D Laplacian using finite differences 
        and periodic boundary conditions.
        """
        # Shift indices: left, right, up, down
        f_left  = np.roll(f, -1, axis=1) # i, j+1
        f_right = np.roll(f,  1, axis=1) # i, j-1
        f_up    = np.roll(f, -1, axis=0) # i+1, j
        f_down  = np.roll(f,  1, axis=0) # i-1, j
        
        return (f_left + f_right + f_up + f_down - 4*f) / (self.dx**2)
    
    def chemical_potential(self):
        """
        calculate chemical potential of current step
        a,k,M are set to 1 in dimensionless form

        returns:
        mu: NxN grid of chemical potential at each site
        """
        # mu = -self.a * self.phi*(1-self.phi**2) - self.k * self.laplacian(self.phi)
        mu = -self.phi + self.phi**3 - self.laplacian(self.phi)
        return mu
    
    def sweep_phi(self):
        """
        update composition of grid in one sweep

        returns:
        None: updates grid in place
        """
        mu = self.chemical_potential()
        self.phi += self.dt * self.laplacian(mu)
        self.phi -= np.mean(self.phi) - self.phi_0 # ensure mass conservation

    def free_energy(self):
        """
        calculates the free energy density of the grid

        returns:
        f: free energy density
        """
        f = - 1/2 * self.phi**2 + 1/4 * self.phi**4 
        grad_x = (np.roll(self.phi, -1, axis=0) - np.roll(self.phi, 1, axis=0)) / (2 * self.dx)
        grad_y = (np.roll(self.phi, -1, axis=1) - np.roll(self.phi, 1, axis=1)) / (2 * self.dx)

        # grad_x = (np.roll(self.phi, -1, axis=0) - self.phi) / (2 * self.dx)
        # grad_y = (np.roll(self.phi, -1, axis=1) - self.phi) / (2 * self.dx)

        f += 0.5 * (grad_x**2 + grad_y **2)
        return np.sum(f)

    def animate(self):
        """
        Animate the evolution of the composition grid over time

        returns:
        None: displays an animation of the grid evolution
        """
        self.phi = self.initialize_grid()
    
        fig = plt.figure()
        im = plt.imshow(self.phi, animated=True, cmap='coolwarm')
        
        def update_frame(_):
            for j in range(10): # perform 10 sweeps per frame
                self.sweep_phi()
            im.set_array(self.phi)
            return [im]
        
        ani = animation.FuncAnimation(fig, update_frame, frames=1000, interval=1, blit=True, repeat_delay=1000)
        plt.show()

    def run_energy(self):
        """
        Run evolution of composition and plot the evolution of free energy over time until equilibrium
        or max iterations is reached

        returns:
        None: saves plot and data to file
        """
        energy_history = [self.free_energy()]
        converged = False
        iters=[0]
        window = 50

        for i in range(1,self.num_iter):
            self.sweep_phi()
            energy_history.append(self.free_energy())
            iters.append(i)

            if i > window: # some equilibriation
                recent = energy_history[-window:]
                if np.std(recent) < self.threshold:
                    # if np.argmax(recent) > np.argmin(recent): #ensure it's going down
                    print(f"Equilibrium reached at iteration {i} (Threshold: {self.threshold})")
                    converged = True
                    break
        
        plt.figure(figsize=(8, 5))
        plt.plot(iters, energy_history)
        plt.title(f'Free Energy Minimisation (phi_0={self.phi_0})')
        plt.xlabel('Iterations')
        plt.ylabel('Total Free Energy')
        plt.savefig(f'free_energy_phi{self.phi_0}.png')
        plt.show()
        
        np.savetxt(f'energy_phi{self.phi_0}.dat', energy_history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cahn Hilliard Equation')
    parser.add_argument('-N','--size', type=int, default=100, help='Size of the lattice (N x N)')
    parser.add_argument('--phi0',type=float,default=0,help='Average composition of grid')
    parser.add_argument('-dx',type=float,default=1,help= 'Length step')
    parser.add_argument('-dt',type=float,default=1e-4,help= 'Time step')
    parser.add_argument('--num_iter',type=int,default=50000,help='Number of Iteration to Run')
    parser.add_argument('--threshold',type=float,default=1e-6,help='Change thershold to determine equilibrium')
    parser.add_argument('--animate',action='store_true',help='argument to animate grid')

    args = parser.parse_args()

    model = cahn_hilliard(
        N = args.size,
        phi_0 = args.phi0,
        dx = args.dx,
        dt = args.dt,
        num_iter=args.num_iter,
        threshold = args.threshold
    )
    if args.animate:
        model.animate()
    else:
        model.run_energy()
