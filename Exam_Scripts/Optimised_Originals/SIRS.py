"""
simulate the SIRS Cellular Automaton model on a 2D grid with periodic boundary conditions.

"""

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import argparse
from collections import deque
from matplotlib.colors import ListedColormap
from numba import njit

@njit
def find_nearest_neighbors(grid,i,j):
        """
        finds the nearest neighbors for a given point in the lattice assuming periodic boundary counditions
        returns as list of neighbors

        returns:

        nearest_neighbors: set of tuples representing nearest neighbor state in grid
        Set chosen for computational efficiency in checking for membership when determining if a cell has infected neighbors, 
        since we only need to know if at least one neighbor is infected rather than counting the number of infected neighbors
        """

        nearest_neighbors_coordinates = set([((i+1)%grid.shape[0],j), # directly right
                             ((i-1)%grid.shape[0],j), # directly left
                             (i,(j+1)%grid.shape[1]), # directly up
                             (i,(j-1)%grid.shape[1]), # directly down
                             ])
        nearest_neighbors = set()
        for coord in nearest_neighbors_coordinates:
                nearest_neighbors.add(grid[coord])
        return nearest_neighbors

@njit
def update_cell(grid, x, y, S, I, R):
        current_state = grid[x,y]

        if current_state == 0: # susceptible
            neighbor_states = find_nearest_neighbors(grid, x, y)
            if 1 in neighbor_states: # at least one neighbor is infected
                grid[x,y] = 1 if np.random.rand() <= S else 0
        elif current_state == 1: # infected
            grid[x,y] = 2 if np.random.rand() <= I else 1
        elif current_state == 2: # recovered
            grid[x,y] = 0 if np.random.rand() <= R else 2

@njit
def sweep(grid, S, I ,R):
        for _ in range(grid.shape[0]**2):
            i,j = np.random.randint(0,grid.shape[0]), np.random.randint(0,grid.shape[1]) # randomly select a cell to update
            update_cell(grid, i, j, S, I, R)


class SIRS:
    def __init__(self,N,debug,num_runs,S,I,R,f,resolution):
        """
    
        N: size of the grid (N x N)
        initial_state: string specifying the initial state of the grid (random, glider, square, oscillator)
        debug: boolean flag to enable debug mode for printing additional information during simulation
        num_runs: number of simulation runs to perform for averaging equilibrium times
        S: Probability of a susceptible cell becoming infected
        I: Probability of an infected cell recovering
        R: Probability of a recovered cell becoming susceptible again
        f: fraction of recovered permanently immune cells
        resolution: update resolution for probability updates in run
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
        if self.debug:
            print(f'Initialized SIRS Model with N={self.N}, S={self.S}, I={self.I}, R={self.R}, f={self.f}')

        self.grid = self.initialize_grid()


    def initialize_grid(self):
        """
        generate inital grid of states, dimension N x N
        character states: 0 = susceptible, 1 = infected, 2 = recovered, 3 = permanently immune      

        returns:
        grid: N x N numpy array of states
        """

        grid = np.random.choice(
            [0, 1, 3],  
            size = (self.N, self.N),
            p = [0.5  - self.f/2, 0.5 - self.f/2, self. f]
        )
        # Only two states are needed for the initial grid since the recovered state can only be reached after infection
        return grid

    def find_nearest_neighbors(self,i,j):
        """
        finds the nearest neighbors for a given point in the lattice assuming periodic boundary counditions
        returns as set of neighbors

        returns:

        nearest_neighbors: set of ints representing nearest neighbor state in grid
        Set chosen for computational efficiency in checking for membership when determining if a cell has infected neighbors, 
        since we only need to know if at least one neighbor is infected rather than counting the number of infected neighbors
        """

        nearest_neighbors_coordinates = set([((i+1)%self.N,j), # directly right
                             ((i-1)%self.N,j), # directly left
                             (i,(j+1)%self.N), # directly up
                             (i,(j-1)%self.N), # directly down
                             ])
        
        nearest_neighbors = set()
        for coord in nearest_neighbors_coordinates:
                nearest_neighbors.add(self.grid[coord])
        if self.debug:
            print(f'nearest neighbors states for cell ({i},{j}): {nearest_neighbors}')
        return nearest_neighbors

    def update_cell(self, x, y):
        """
        Update the state of a single cell based on the SIRS model rules

        returns:
        None: updates the state of the cell in place in the grid
        """
        current_state = self.grid[x,y]

        if current_state == 0: # susceptible
            neighbor_states = self.find_nearest_neighbors(x,y)
            if 1 in neighbor_states: # at least one neighbor is infected
                self.grid[x,y] = 1 if np.random.rand() <= self.S else 0
        elif current_state == 1: # infected
            self.grid[x,y] = 2 if np.random.rand() <= self.I else 1
        elif current_state == 2: # recovered
            self.grid[x,y] = 0 if np.random.rand() <= self.R else 2

    def sweep(self):
        """
        Perform a single sweep of the grid, updating each cell once on average
        
        returns:
        None: updates the state of the grid in place after performing a sweep
        """
        for _ in range(self.N**2):
            i,j = np.random.randint(0,self.N), np.random.randint(0,self.N) # randomly select a cell to update
            self.update_cell(i,j)


    def animate(self):
        """
        Animate the evolution of the SIRS Model grid over time

        returns:
        None: displays an animation of the grid evolution
        """
        self.grid = self.initialize_grid()
    
        fig = plt.figure()
        cmap = ListedColormap(['white', 'red', 'blue','black'])
        im = plt.imshow(self.grid, animated=True, cmap=cmap,vmin=0,vmax=3)

        
        def update_frame(_):
            self.sweep()
            im.set_array(self.grid)
            return [im]
        
        ani = animation.FuncAnimation(fig, update_frame, frames=1000, interval=20, blit=True, repeat_delay=1000)
        plt.show()

    def plot_single_frame(self):
        """
        Plot a single frame of the game of life grid

        returns:
        None: displays a single frame of the grid
        """
        self.grid = self.initialize_grid()
        plt.imshow(self.grid, cmap='binary')
        plt.title('Initial State of Game of Life')
        plt.axis('off')
        plt.show()

    def count_infected(self):
        """
        Count the number of infected cells in the current grid

        returns:
        int: number of infected cells in the current grid
        """
        return np.sum(self.grid == 1)

    def run(self):
        """
        Run the simulation until equilibrium is reached and record the time taken to reach equilibrium for each run

        returns:
        None: prints the average time taken to reach equilibrium across all runs

        stores the data for the plot in a .npz file for later analysis
        """
        self.I = 0.5

        res_steps = len(np.arange(0, 1 + self.resolution, self.resolution))
        heatmap_data = np.zeros((res_steps, res_steps))

        for i,p_s in enumerate(np.arange(0,1+self.resolution,self.resolution)):
            for j,p_r in enumerate(np.arange(0,1+self.resolution,self.resolution)):
                self.S = p_s
                self.R = p_r
                self.grid = self.initialize_grid()
                infected_count = []
                for _ in range(100): # equilibriration
                    # self.sweep()
                    sweep(self.grid,self.S,self.I,self.R)
                for run in range(self.num_runs):
                    # self.sweep()
                    sweep(self.grid,self.S,self.I,self.R)
                    infected_count.append(self.count_infected())
                heatmap_data[j, i] = np.mean(infected_count) / self.N**2
        
        plt.figure(figsize=(10,6))
        plt.heatmap = plt.imshow(heatmap_data,
                                  extent=(0,1,0,1), origin='lower', cmap='viridis')
        plt.colorbar(plt.heatmap, label='Average Fraction of Infected Cells at Equilibrium')
        plt.xlabel('Probability of Infection (S)')
        plt.ylabel('Probability of Resusceptibility (R)')
        plt.title('SIRS Model: Average Fraction of Infected Cells at Equilibrium')
        plt.savefig(f'{self.resolution}_sirs_equilibrium_heatmap.png')
        plt.show()

        np.savez(f'{self.resolution}_sirs_infection_heatmap_data.npz',
                 heatmap_data=heatmap_data,
                 S_vales=np.arange(0,1+self.resolution,self.resolution),
                 R_values=np.arange(0,1+self.resolution,self.resolution)
                 )

    def run_variance(self):
        """
        Run the simulation until equilibrium is reached and record the variance of the number of infected cells at equilibrium for each run

        returns:
        None: prints the average variance of the number of infected cells at equilibrium across all runs

        stores the data for the plot in a .npz file for later analysis
        """
        self.I = 0.5
        self.R = 0.5

        variance = []
        variance_errors = []

        for i,p_s in enumerate(np.arange(0.2,0.5+self.resolution,self.resolution)):
            self.S = p_s
            self.grid = self.initialize_grid()
            infected_count = []
            infected_squared_count = []
            for _ in range(100): # equilibriration
                sweep(self.grid,self.S,self.I,self.R)

            for run in range(self.num_runs * 10):
                sweep(self.grid,self.S,self.I,self.R)
                if run % 10 == 0:
                    I = self.count_infected()
                    infected_count.append(I)
                    infected_squared_count.append(I**2)
            mean_infected = np.mean(infected_count)
            mean_infected_squared = np.mean(infected_squared_count)
            variance.append((mean_infected_squared - mean_infected**2) / self.N**2)

            #calculate variance error using bootstrap method
            bootstrap_variances = []
            for _ in range(1000):
                bootstrap_sample = np.random.choice(infected_count, size=len(infected_count), replace=True)
                bootstrap_mean = np.mean(bootstrap_sample)
                bootstrap_mean_squared = np.mean(bootstrap_sample**2)
                bootstrap_variances.append((bootstrap_mean_squared - bootstrap_mean**2) / self.N**2)
            variance_errors.append(np.std(bootstrap_variances))

        plt.figure(figsize=(10,6))
        plt.errorbar(np.arange(0.2,0.5+self.resolution,self.resolution), variance, yerr=variance_errors, label='Variance of Fraction of Infected Cells at Equilibrium')
        plt.xlabel('Probability of Infection (S)')
        plt.ylabel('Variance of Fraction of Infected Cells at Equilibrium')
        plt.title('SIRS Model: Variance of Fraction of Infected Cells at Equilibrium')
        plt.legend()
        plt.grid()
        plt.savefig(f'{self.resolution}_sirs_equilibrium_variance_plot.png')
        plt.show()

        np.savez(f'{self.resolution}_sirs_infection_variance_plot_data.npz',
                 variance_data=variance,
                 S_vales=np.arange(0.2,0.5+self.resolution,self.resolution),
                 variance_errors=variance_errors
                 )
        
    def run_immunity(self):
        """
        Run the simulation until equilibrium is reached and record the average fraction of infected cells at 
        equilibrium for different values of f, the fraction of recovered cells that become permanently immune

        returns:
        None: plots the average fraction of infected cells at equilibrium for different values of f

        stores the data for the plot in a .npz file for later analysis
        """
        self.S = 0.5
        self.I = 0.5
        self.R = 0.5

        mean_infected = []
        
        for f in np.arange(0,1+self.resolution,self.resolution):
            self.f = f
            self.grid = self.initialize_grid()
            infected_count = []
            for _ in range(100): # equilibriration
                sweep(self.grid,self.S,self.I,self.R)
            for run in range(self.num_runs):
                sweep(self.grid,self.S,self.I,self.R)
                infected_count.append(self.count_infected())
            mean_infected.append(np.mean(infected_count) / self.N**2)
        
        plt.figure(figsize=(10,6))
        plt.plot(np.arange(0,1+self.resolution,self.resolution), mean_infected, label='Average Fraction of Infected Cells at Equilibrium')
        plt.xlabel('Fraction of Recovered Cells that are Permanently Immune (f)')
        plt.ylabel('Average Fraction of Infected Cells at Equilibrium')
        plt.title('SIRS Model: Effect of Permanent Immunity on Equilibrium Infection Levels')
        plt.legend()
        plt.grid()
        plt.savefig(f'{self.resolution}_sirs_equilibrium_immunity_plot.png')
        plt.show()

        np.savez(f'{self.resolution}_sirs_equilibrium_immunity_plot_data.npz',
                 infected_count=mean_infected,
                 f_values=np.arange(0,1+self.resolution,self.resolution)
                 )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SIRS Model Simulation')

    # General simulation parameters
    parser.add_argument('-N','--size', type=int, default=50, help='Size of the lattice (N x N)')
    parser.add_argument('--num_runs',type=int,default=1000,help='Number of simulation runs to perform')
    parser.add_argument('--resolution',type=float,default=0.05,help='Update resolution for probability updates in run method (default: 0.05)')

    # Model Parameters for SIRS model
    parser.add_argument('-S','--infected_prob',type=float,default=0.5,help='Probability of a susceptible cell becoming infected')
    parser.add_argument('-I','--recovery_prob',type=float,default=0.5,help='Probability of an infected cell recovering')
    parser.add_argument('-R','--resusceptibility_prob',type=float,default=0.5,help='Probability of a recovered cell becoming susceptible again')
    parser.add_argument('-f','--immune_fraction',type=float,default=0.0,help='Fraction of recovered cells that become permanently immune')

    # Function level arguments for running specific analyses or enabling debug mode
    parser.add_argument('--animate',action='store_true',help='Animate the evolution of the grid over time')
    parser.add_argument('--run_variance',action='store_true',help='Run the simulation to calculate the variance of the number of infected cells at equilibrium')
    parser.add_argument('--run_immunity',action='store_true',help='Run the simulation to analyze the effect of permanent immunity on equilibrium infection levels')
    parser.add_argument('--debug',action='store_true',help ='Enable debug mode to print additional information during simulation')


    args = parser.parse_args()

    model = SIRS(
        N=args.size,
        debug = args.debug,
        num_runs = args.num_runs,
        S = args.infected_prob,
        I = args.recovery_prob,
        R = args.resusceptibility_prob,
        f = args.immune_fraction,
        resolution = args.resolution
        )
    
    if args.debug:
        # Print the inital grid state for debugging purposes
        model.plot_single_frame()
    if args.animate:
        if args.debug:
            print('Starting animation of SIRS model evolution...')
        model.animate()

    elif args.run_variance:
        if args.debug:
            print('Starting variance analysis of SIRS model at equilibrium...')
        model.run_variance()
    elif args.run_immunity:
        if args.debug:
            print('Starting analysis of effect of permanent immunity on SIRS model equilibrium infection levels...')
        model.run_immunity()
    else:
        if args.debug:
            print('Starting main simulation run to analyze equilibrium infection levels across S and R parameter space...')
        model.run()