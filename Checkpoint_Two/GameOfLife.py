"""
simulate the Game of life

"""

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import argparse
from scipy.signal import convolve2d
from collections import deque

class GameOfLife:
    def __init__(self,N,initial_state,debug,alive_fraction,num_runs):
        """
    
        N: size of the grid (N x N)
        initial_state: string specifying the initial state of the grid (random, glider, square, oscillator)
        debug: boolean flag to enable debug mode for printing additional information during simulation
        alive_fraction: fraction of cells that are initially alive (only used if initial_state is random)
        num_runs: number of simulation runs to perform for averaging equilibrium times
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
            print(f'Initialized GameOfLife with N={self.N}, initial_state={self.initial_state}, alive_fraction={self.alive_fraction}')

        self.equilibrium_times = [] # list to store the time taken to reach equilibrium for each simulation runs
        self.centres_of_mass = [] # list to store the centre of mass of the alive cells at each time step
        self.initialize_grid()


    def initialize_grid(self):
        """
        generate inital grid of states, dimension N x N
        sets values to 1 if cell is "alive" and 0 if cell is "dead"
        

        returns:
        grid: N x N numpy array of states
        """

        grid = np.zeros((self.N, self.N), dtype=np.int8)
        if self.initial_state == 'random':
            grid = np.random.choice(
                [0, 1],
                size = (self.N, self.N),
                p = [1 - self.alive_fraction, self.alive_fraction]
            )
        
        if self.initial_state == 'glider':
            grid = np.zeros((self.N,self.N))
            glider_pattern = np.array([[0,2],[1,0],[2,1],[2,2],[1,2]])
            for cell in glider_pattern:
                grid[cell[0],cell[1]] = 1
            
        if self.initial_state == 'square':
            grid = np.zeros((self.N,self.N))
            random_x = np.random.randint(0,self.N)
            random_y = np.random.randint(0,self.N)
            square_pattern = np.array([[random_x,random_y],[(random_x+1)%self.N,random_y],[random_x,(random_y+1)%self.N],[(random_x+1)%self.N,(random_y+1)%self.N]])
            for cell in square_pattern:
                grid[cell[0],cell[1]] = 1

        if self.initial_state == 'blinker':
            grid = np.zeros((self.N,self.N))
            random_x = np.random.randint(0,self.N)
            random_y = np.random.randint(0,self.N)
            oscillator_pattern = np.array([[random_x,random_y],[random_x-1,random_y],[random_x+1,random_y]])
            for cell in oscillator_pattern:
                grid[cell[0],cell[1]] = 1
        return grid
    

    def determine_equilibriation(self):
        current_hash = self.current_grid.tobytes()
        
        # Check if current state matches any of the stored previous states
        if current_hash in self.history:
            return True
            
        self.history.append(current_hash)
        return False


    def sweep(self):
        # Compute neighbor count using convolution
        kernel = np.array([[1,1,1],
                        [1,0,1],
                        [1,1,1]])

        neighbor_count = convolve2d(
            self.current_grid,
            kernel,
            mode='same',
            boundary='wrap'   # periodic boundary conditions
        )

        # Apply Game of Life rules
        self.future_grid = (
            (neighbor_count == 3) |
            ((self.current_grid == 1) & (neighbor_count == 2))
        ).astype(int)

    def get_centre_of_mass(self):
        indicies = np.argwhere(self.current_grid == 1)
        if len(indicies) == 0: return None
        
        return np.mean(indicies,axis = 0)



    def animate(self):
        """
        Animate the evolution of the game over life grid over time

        returns:
        None: displays an animation of the grid evolution
        """
        self.current_grid = self.initialize_grid()
        # self.future_grid = np.copy(self.current_grid) # initialize future grid to be the same as current grid at the start of the animation
    
        fig = plt.figure()
        im = plt.imshow(self.current_grid, animated=True, cmap='binary')
        
        def update_frame(_):
            self.sweep()
            self.current_grid = self.future_grid.copy()
            im.set_array(self.current_grid)
            return [im]
        
        ani = animation.FuncAnimation(fig, update_frame, frames=1000, interval=20, blit=True, repeat_delay=1000)
        plt.show()

    def plot_single_frame(self):
        """
        Plot a single frame of the game of life grid

        returns:
        None: displays a single frame of the grid
        """
        self.current_grid = self.initialize_grid()
        plt.imshow(self.current_grid, cmap='binary')
        plt.title('Initial State of Game of Life')
        plt.axis('off')
        plt.show()

    def plot_equilibrium_times(self):
        """
        Plot a histogram of the equilibrium times across all runs

        returns:
        None: displays a histogram of equilibrium times
        """
        plt.hist(self.equilibrium_times, bins=50, color='orange', alpha=0.7)
        plt.title('Distribution of Equilibrium Times')
        plt.xlabel('Time to Equilibrium (steps)')
        plt.ylabel('Frequency')
        # plt.grid(True)
        plt.savefig(f'{self.initial_state}_equilibrium_times_histogram.png')
        plt.show()

    def plot_centres_of_mass(self):
        """
        Plot the trajectory of the centre of mass of the alive cells over time

        returns:
        None: displays a plot of the centre of mass trajectory
        """

        centres_of_mass_array = np.array(self.centres_of_mass)

        slope, cum_dist, time, intercept = self.calculate_speed()
        plt.plot(time, cum_dist, 'bo', markersize=2, color='blue', label='Centre of Mass Trajectory')
        plt.plot(time, slope*time + intercept, 'r-', label=f'Fit: v={slope:.3f}') 
        plt.legend()
        plt.title('Trajectory of Centre of Mass of Alive Cells')
        plt.xlabel('Time iteration')
        plt.ylabel('Speed of Centre of Mass')
        # plt.grid(True)
        plt.savefig(f'{self.initial_state}_centre_of_mass_trajectory.png')
        plt.show()

    def run(self):
        """
        Run the simulation until equilibrium is reached and record the time taken to reach equilibrium for each run

        returns:
        None: prints the average time taken to reach equilibrium across all runs
        """
        for run in range(self.num_runs):
            self.current_grid = self.initialize_grid()
            self.history.clear() # clear the history of previous states at the start of each run

            if self.initial_state in ['random']:
                max_time_steps = 15000
            else:
                max_time_steps = 100 # add a lower time step for initial states that never reach equilibrium to save computation time

            self.time_steps = 0
            while self.time_steps < max_time_steps: # add a maximum time step limit to prevent infinite loops in cases where equilibrium is not reached:
                self.sweep()
                if self.determine_equilibriation():
                    break

                 # Update current grid
                self.current_grid = self.future_grid.copy()
                self.time_steps += 1

            self.equilibrium_times.append(self.time_steps)
            if self.debug:
                print(f'Run {run+1}/{self.num_runs}: Time to equilibrium = {self.time_steps} steps')

        average_time = np.mean(self.equilibrium_times)
        print(f'Average time to reach equilibrium over {self.num_runs} runs: {average_time:.2f} steps')

        if self.initial_state in ['random']:
            self.plot_equilibrium_times()
        self.plot_centres_of_mass()

        
    def calculate_speed(self):
        """
        Calculate the speed of the centre of mass of the alive cells over time

        returns:
        speed: float representing the average speed of the centre of mass
        """
        centres_of_mass_array = np.array(self.centres_of_mass)

        diffs =  np.diff(centres_of_mass_array,axis=0) # axis zero to calcualate the difference between consecutive time steps
        diffs = (diffs + self.N/2) % self.N - self.N/2 # adjust for periodic boundary conditions
        distances = np.linalg.norm(diffs, axis =1) # calculate the distance traveled by the centre of mass between consecutive time steps
        cum_dist = np.cumsum(distances) # cumulative distance traveled by the centre of mass over time
        time = np.arange(len(cum_dist)) # time steps corresponding to each cumulative distance value

        slope, intercept = np.polyfit(time, cum_dist, 1) # fit a linear model to the cumulative distance vs time data to estimate speed

        
        if self.debug:
            print(f'Centres of Mass: {centres_of_mass_array}'
                  f'\nDiffs: {diffs}',
                  f'\nDistances: {distances}',
                  f'\ntime:steps: {np.arange(0,distances.size)}')

        return slope, cum_dist, time, intercept
    
    def glider_run(self):
        """
        Run a simulation of the glider pattern to demonstrate its movement across the grid

        returns:
        None: displays an animation of the glider pattern moving across the grid
        """
        self.current_grid = self.initialize_grid()
        
        max_time_steps = 100
        self.time_steps = 0
        while self.time_steps < max_time_steps: # add a maximum time step limit to prevent infinite loops in cases where equilibrium is not reached:
                self.sweep()
                 # Update current grid
                self.current_grid = self.future_grid.copy()
                self.time_steps += 1
                self.centres_of_mass.append(self.get_centre_of_mass())

        self.plot_centres_of_mass()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ising Model Simulation')
    parser.add_argument('--N', type=int, default=50, help='Size of the lattice (N x N)')
    parser.add_argument('--initial_state', type=str,default='random',choices= ['random','glider','square','blinker'], help='Initial state of the lattice (random or ordered)')
    parser.add_argument('--debug',action='store_true',help ='Enable debug mode to print additional information during simulation')
    parser.add_argument('--alive_fraction',type=float,default=0.5,help='Fraction of cells that are initially alive (only used if initial_state is random)')
    parser.add_argument('--animate',action='store_true',help='Animate the evolution of the grid over time')
    parser.add_argument('--num_runs',type=int,default=1000,help='Number of simulation runs to perform for averaging equilibrium times')

    args = parser.parse_args()

    model = GameOfLife(
        N=args.N,
        initial_state=args.initial_state,
        debug = args.debug,
        alive_fraction = args.alive_fraction,
        num_runs = args.num_runs
        )
    if args.debug:
        # Print the inital grid state for debugging purposes
        model.plot_single_frame()
    if args.animate:
        model.animate()
    
    elif args.initial_state == 'glider':
        model.glider_run()

    else:
        model.run()