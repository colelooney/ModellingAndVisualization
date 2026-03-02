import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import argparse
from collections import deque
from matplotlib.colors import ListedColormap
# from numba import njit
from scipy.signal import convolve2d

from SIRS import SIRS
from GameOfLife import GameOfLife

GOL_PATTERNS = {
        #spaceships
        'glider': np.array([[0,2],[1,0],[2,1],[2,2],[1,2]]),
        'LWSS': np.array([[0,0], [0,2], [1,3], [2,3], [3,0], [3,3], [4,1], [4,2], [4,3]]),
        'MWSS': np.array([[0,2], [1,0], [1,4], [2,5], [3,0], [3,5], [4,1], [4,2], [4,3], [4,4], [4,5]]),
        'HWSS': np.array([[0,2], [0,3], [1,0], [1,5], [2,6], [3,0], [3,6], [4,1], [4,2], [4,3], [4,4], [4,5], [4,6]]),

        # still lifes
        'block': np.array([[0,0],[1,0],[0,1],[1,1]]),
        'beehive': np.array([[1,0],[2,0],[0,1],[3,1],[1,2],[2,2]]),
        'loaf': np.array([[1,0],[2,0],[0,1],[3,1],[1,2],[3,2],[2,3]]),
        'boat': np.array([[0,0],[1,0],[0,1],[2,1],[1,2]]),
        'tub': np.array([[1,0],[0,1],[2,1],[1,2]]),

        #oscillators
        'blinker': np.array([[0,0],[1,0],[2,0]]),
        'toad': np.array([[1,0],[2,0],[3,0],[0,1],[1,1],[2,1]]),
        'beacon': np.array([[0,0],[1,0],[0,1],[3,2],[2,3],[3,3]]),
        'pulsar': np.array([[2,0],[3,0],[4,0],[8,0],[9,0],[10,0],
                            [0,2],[5,2],[7,2],[12,2],
                            [0,3],[5,3],[7,3],[12,3],
                            [0,4],[5,4],[7,4],[12,4],
                            [2,5],[3,5],[4,5],[8,5],[9,5],[10,5],
                            [2,7],[3,7],[4,7],[8,7],[9,7],[10,7],
                            [0,8],[5,8],[7,8],[12,8],
                            [0,9],[5,9],[7,9],[12,9],
                            [0,10],[5,10],[7,10],[12,10],
                            [2,12],[3,12],[4,12],[8,12],[9,12],[10,12]])

    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cellular Automata Simulation: SIRS and Game of Life')
    
    # Model selection argument
    parser.add_argument('--model', type = str, required=True, choices=['sirs', 'game_of_life'], help='Select the model to run: sirs or game_of_life')

    # General simulation parameters
    parser.add_argument('-N','--size', type=int, default=50, help='Size of the lattice (N x N)')
    parser.add_argument('--num_runs',type=int,default=1000,help='Number of simulation runs to perform')
    parser.add_argument('--resolution',type=float,default=0.05,help='Update resolution for probability updates in SIRS run method (default: 0.05)')
    
    # Model Parameters for Game of Life
    state_choices = list(GOL_PATTERNS.keys()) + ['random']
    parser.add_argument('--initial_state', type=str,default='random',choices= state_choices, help='Initial state of the Game of Life Lattice lattice (random or set patterns)')
    parser.add_argument('-F','--alive_fraction',type=float,default=0.5,help='Fraction of cells that are initially alive in Game Of Life (only used if initial_state is random)')

    # Model Parameters for SIRS model
    parser.add_argument('-S','--infected_prob',type=float,default=0.5,help='Probability of a susceptible cell becoming infected in SIRS')
    parser.add_argument('-I','--recovery_prob',type=float,default=0.5,help='Probability of an infected cell recovering in SIRS')
    parser.add_argument('-R','--resusceptibility_prob',type=float,default=0.5,help='Probability of a recovered cell becoming susceptible again in SIRS')
    parser.add_argument('-f','--immune_fraction',type=float,default=0.0,help='Fraction of recovered cells that become permanently immune in SIRS')

    # Function level arguments for running specific analyses or enabling debug mode
    parser.add_argument('--animate',action='store_true',help='Animate the evolution of the grid over time')
    parser.add_argument('--run_variance',action='store_true',help='Run the SIRS simulation to calculate the variance of the number of infected cells at equilibrium')
    parser.add_argument('--run_immunity',action='store_true',help='Run the SIRS simulation to analyze the effect of permanent immunity on equilibrium infection levels')
    parser.add_argument('--debug',action='store_true',help ='Enable debug mode to print additional information during simulation')


    args = parser.parse_args()

    if args.model == 'sirs':
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
    elif args.model == 'game_of_life':

        model = GameOfLife(
        N=args.size,
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
        
        elif args.initial_state in ['glider','LWSS','MWSS','HWSS']:
            model.glider_run()

        else:
            model.run()