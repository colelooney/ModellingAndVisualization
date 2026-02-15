"""
simulate the ising model

"""

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import argparse

class IsingModel:
    def __init__(self,N,T,dynamic, J = 1):
        self.N = N
        self.T = T
        self.dynamic = dynamic
        self.J = J
        self.grid = None

    def pair_energy(self,i,j):
        """
        calculate the pairwise energy of two atoms
        i,j represent spin of respective particles
        """

        return -self.J * i * j

    def initalize_grid(self):
        """
        generate random initial grid of spin states, dimension N x N
        sets spin up as +1 and spin down as -1
        """

        grid = np.random.rand(self.N,self.N)
        grid[grid >= 0.5] = 1
        grid[grid < 0.5] = -1

        return grid

    def flip_probablity(self,delta_energy):
        P = math.exp(-delta_energy/self.T)
        return P

    def find_nearest_neighbors(self,i,j):
        """
        finds the nearest neighbors for a given point in the lattice assuming periodic boundary counditions
        returns as list of neighbors
        """

        nearest_neighbors = [((i+1)%self.N,j),((i-1)%self.N,j),(i,(j+1)%self.N),(i,(j-1)%self.N)]
        return nearest_neighbors

    def glauber_energy(self):
        i = np.random.randint(0,self.N)
        j = np.random.randint(0,self.N)
        energy = 0
        flip_energy = 0

        nearest_neighbors = self.find_nearest_neighbors(i,j)

        for neighbor in nearest_neighbors:
            energy +=  self.pair_energy(self.grid[(i,j)],self.grid[neighbor])
            flip_energy += self.pair_energy(-self.grid[(i,j)],self.grid[neighbor])
        delta_energy = flip_energy - energy
        return delta_energy, (i,j)
        

    def glauber_update(self):
        delta_energy, (i,j) = self.glauber_energy()

        if delta_energy <= 0:
            self.grid[(i,j)] = -self.grid[(i,j)]

        elif self.flip_probablity(delta_energy) > np.random.rand():
            self.grid[(i,j)] = -self.grid[(i,j)]
        
    def kawasaki_energy(self):
        i_1 = np.random.randint(0,self.N)
        j_1 = np.random.randint(0,self.N)

        i_2 = np.random.randint(0,self.N)
        j_2 = np.random.randint(0,self.N)

        energy = 0
        swap_energy = 0

        nearest_neighbors_one = self.find_nearest_neighbors(i_1,j_1)
        nearest_neighbors_two = self.find_nearest_neighbors(i_2,j_2)

        if (i_2,j_2) not in nearest_neighbors_one:
            for neighbor in nearest_neighbors_one:
                energy += self.pair_energy(self.grid[i_1,j_1],self.grid[neighbor])
                swap_energy += self.pair_energy(self.grid[i_2,j_2],self.grid[neighbor])

            for neighbor in nearest_neighbors_two:
                energy += self.pair_energy(self.grid[i_2,j_2],self.grid[neighbor])
                swap_energy += self.pair_energy(self.grid[i_1,j_1],self.grid[neighbor])

        else:
            for neighbor in nearest_neighbors_one:
                energy += self.pair_energy(self.grid[i_1,j_1],self.grid[neighbor])
                temp_neighbor = (i_1,j_1) if neighbor ==  (i_2,j_2) else neighbor
                swap_energy += self.pair_energy(self.grid[i_2,j_2],self.grid[temp_neighbor])

            for neighbor in nearest_neighbors_two:
                energy += self.pair_energy(self.grid[i_2,j_2],self.grid[neighbor])
                temp_neighbor = (i_2,j_2) if neighbor ==  (i_1,j_1) else neighbor
                swap_energy += self.pair_energy(self.grid[i_1,j_1],self.grid[temp_neighbor])

        delta_energy = swap_energy - energy

        return delta_energy, (i_1,j_1),(i_2,j_2)

    def kawasaki_update(self):
        delta_energy, (i_1,j_1),(i_2,j_2) = self.kawasaki_energy()

        if delta_energy <= 0:
            self.grid[(i_1,j_1)],self.grid[(i_2,j_2)] = self.grid[(i_2,j_2)], self.grid[(i_1,j_1)]

        elif self.flip_probablity(delta_energy) > np.random.rand():
            self.grid[(i_1,j_1)],self.grid[(i_2,j_2)] = self.grid[(i_2,j_2)], self.grid[(i_1,j_1)]

    
    def determine_magnetisation(self):
        # Sum the spin values over the whole grid, calculates M and M^2
        M = np.sum(self.grid)
        M_squared = M**2
        return M, M_squared
    
    def average_magnetisation(self):

        self.grid = self.initalize_grid()
        
        mags = []
        mags_squared = []
        for j in range(100):
            for _ in range(self.N * self.N):
                if self.dynamic == 'glauber':
                    self.glauber_update()
                elif self.dynamic == 'kawasaki':
                    self.kawasaki_update()
        
        for j in range(10000):
            for _ in range(self.N * self.N):
                if self.dynamic == 'glauber':
                    self.glauber_update()
                elif self.dynamic == 'kawasaki':
                    self.kawasaki_update()
            
            if j % 10 == 0:
                # print(f"Sampling at step {j}") #debugging line
                M,M_squared = self.determine_magnetisation()
                mags.append(M)
                mags_squared.append(M_squared)
        avg_mag = sum(mags)/len(mags)
        avg_mag_squared = sum(mags_squared)/len(mags_squared)

        return avg_mag, avg_mag_squared
    
    def magnetic_susceptibility(self, avg_mag, avg_mag_squared):
        chi = (avg_mag_squared - avg_mag**2) / (self.N**2 * self.T)
        return chi

    def animate(self):
        self.grid = self.initalize_grid()
    
        fig = plt.figure()
        ims = []

        for _ in range(1000):
            for _ in range(self.N*self.N):
                if self.dynamic == 'glauber':
                    self.glauber_update()
                elif self.dynamic == 'kawasaki':
                    self.kawasaki_update()
            
            im = plt.imshow(self.grid, animated=True, cmap='binary')
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ising Model Simulation')
    parser.add_argument('--N', type=int, default=50, help='Size of the lattice (N x N)')
    parser.add_argument('--T', type=float, default=2.5, help='Temperature of the system')
    parser.add_argument('--dynamic', type=str, choices=['glauber', 'kawasaki'], default='glauber', help='Type of dynamics to use')
    args = parser.parse_args()

    model = IsingModel(N=args.N,T=args.T,dynamic=args.dynamic)
    m,mm = model.average_magnetisation()
    print(m,mm)
    # model.animate()