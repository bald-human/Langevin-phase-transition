'''
gas fluid solid phase transition of particles in improper solvent using Langevin dynamics, Van der Waals forces and Brownian Motion
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import math
from matplotlib.colors import LinearSegmentedColormap   
from matplotlib.patches import Polygon, Circle
from itertools import combinations , product

class Simulation:
    def __init__(self, N, sim_time, grid_size=400, dt=0.1, gamma=1, T=0.2,cooling_rate=0.01, epsilon=2.5, sigma=2 ):
        self.N = N
        self.sim_time = sim_time
        self.grid_size = grid_size
        self.time = 0
        self.dt = dt

        self.epsilon = epsilon
        self.sigma = sigma
        self.gamma = gamma
        self.T = T
        self.cooling_rate = cooling_rate
        self.noise_strength = np.sqrt(2 * self.gamma * self.T * self.dt)

        np.random.seed(42)   # any fixed integer
    
        self.pos = np.zeros((N,2))
        self.v   = np.zeros((N,2))
        self.a   = np.zeros((N,2))

        # force plotting
        self.force_magnitudes = []   # flat list of |F| for every pair, every timestep

        # physical correctness with velocity
        self.v_mean_history = []
        self.v_sq_mean_history = []

        # patches object
        self.point_patches = []

        # plotting
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.ax.grid(True)
        self.ax.set_ylim(-2, self.grid_size+2)
        self.ax.set_xlim(-2, self.grid_size+2)
        
        # Set the border thickness and color
        self.ax.spines['top'].set_linewidth(2)
        self.ax.spines['top'].set_color('black')
        self.ax.spines['right'].set_linewidth(2)
        self.ax.spines['right'].set_color('black')
        self.ax.spines['bottom'].set_linewidth(2)
        self.ax.spines['bottom'].set_color('black')
        self.ax.spines['left'].set_linewidth(2)
        self.ax.spines['left'].set_color('black')

        self.title3 = self.ax.text(0.10, 0.98, '', transform=self.ax.transAxes, ha='center', va='top', animated=True)
        self.title2 = self.ax.text(0.90, 0.98, '', transform=self.ax.transAxes, ha='center', va='top', animated=True)
        self.title = self.ax.text(0.70, 0.98, '', transform=self.ax.transAxes, ha='center', va='top', animated=True)

    """
    def spawn_points(self,):
        # make 2 points that can move in environment
        for i in range(self.N):
            if not self.find_spawn_pos(i):
                continue
            self.v[i] = np.random.normal(0, np.sqrt(self.T), size=2)

            circle = Circle(self.pos[i], radius=0.5, facecolor='black', edgecolor='black', zorder=3)
            self.ax.add_patch(circle)
            self.point_patches.append(circle)
    """

    def spawn_points(self):
        spawned = 0
        for i in range(self.N):
            if self.find_spawn_pos(spawned):
                self.v[spawned] = np.random.normal(0, np.sqrt(self.T), size=2)
                circle = Circle(self.pos[spawned], radius=0.5, facecolor='black', edgecolor='black', zorder=3)
                self.ax.add_patch(circle)
                self.point_patches.append(circle)
                spawned += 1

        if spawned < self.N:
            print(f"Warning: only spawned {spawned}/{self.N} points (grid too dense for spacing constraints)")
            self.pos = self.pos[:spawned]
            self.v = self.v[:spawned]
            self.a = self.a[:spawned]
            self.N = spawned

    def find_spawn_pos(self, i):    # search for position in grid away from other shapes        
        tries = 1000
        num_attempts = 0
        valid_pos = False
        edge_distance = 1*self.sigma
        while num_attempts < tries and not valid_pos:
            pos = np.random.uniform(edge_distance, self.grid_size - edge_distance, 2)   # take random position
            valid_pos = True            # assume true until not
            min_distance = np.random.uniform(1*self.sigma,1.25*self.sigma)        # min distance between shapes from is 50 from edge to edge

            if i == 0:
                self.pos[i] = pos
                return True

            for j_pos in self.pos:                                         # check against all other shapes
                distance = np.linalg.norm(pos - j_pos)
                if distance < min_distance:
                    valid_pos = False   # too close to other shape
                    break               # go next possible position
            if valid_pos:               # if correct position found -> return pos
                self.pos[i] = pos
                return True
            else:
                num_attempts += 1
        return False
           
    def init(self):
        """Initialize the background of the plot."""
        # Reset titles
        self.title.set_text('')  # Clear the main title text
        self.title2.set_text('')  # Clear the secondary title text
        self.title3.set_text('')
        # Return all elements that will be animated
        return self.point_patches + [self.title, self.title2, self.title3]

    def update_point_patches(self,):
        for pos, circle in zip(self.pos, self.point_patches):
            circle.set_center(pos)

    def compute_velocity_stats(self):
        """
        points: list of Points objects at current timestep
        returns v_mean (2,), v_squared_mean (scalar)
        """
        velocities = self.v   # shape (N,2)

        v_mean = velocities.mean(axis=0)        #  (2,) — should be ~[0,0]
        speeds_squared = np.sum(velocities**2, axis=1)      #|v_i|^2 per particle, shape (N,)
        v_squared_mean = speeds_squared.mean()          # scalar

        return v_mean, v_squared_mean

    def theoretical_v_squared(self, m=1.0, dim=2):
        """
        Equipartition prediction: <v^2> = dim * k_B*T / m
        dim=2 because you have 2 velocity components (x and y), each contributing (1/2)k_BT
        Using k_B = 1 convention (standard in simulations working in reduced units)
        """
        return dim * self.T / m

    def compute_lj_force(self, log_step):
        pos = self.pos
        N = self.N

        # all pairwise displacement vectors, vectorized numpy
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]    #(N,N,2)
        diff -= self.grid_size * np.round(diff / self.grid_size)    # minimum image force range

        # distance
        dist = np.linalg.norm(diff, axis=-1)

        # make mask indicies
         # --- only need upper triangle (i<j) to avoid double-counting and self-pairs (i==j gives dist=0) ---
        i_idx, j_idx = np.triu_indices(N, k=1)   # 1D arrays of pair indices, length N*(N-1)/2

        r = dist[i_idx, j_idx]                   # (num_pairs,)
        r_vec = diff[i_idx, j_idx]               # (num_pairs, 2)

        # optional cutoff range with boolean mask
        cut = True
        if cut:
            cutoff = 2.5*self.sigma
            in_cut = r < cutoff
            i_idx = i_idx[in_cut]
            j_idx = j_idx[in_cut]
            r     = r[in_cut]
            r_vec = r_vec[in_cut] 

        r = np.maximum(r, 1e-4)                  # avoid singularity, vectorized
        r_hat = r_vec / r[:, np.newaxis]         # (num_pairs, 2)

        # --- LJ force magnitude, vectorized over all pairs at once ---
        F_scalar = 24 * self.epsilon / r * (2*(self.sigma/r)**12 - (self.sigma/r)**6)   # (num_pairs,)

        if log_step:
            self.force_magnitudes.extend(np.abs(F_scalar))   # log all pair forces this step

        F_max = 60
        F_scalar = np.clip(F_scalar, -F_max, F_max)

        F_vec = F_scalar[:, np.newaxis] * r_hat   # (num_pairs, 2)

        # --- scatter forces back onto each particle's acceleration ---
        self.a = np.zeros((N, 2))
        np.add.at(self.a, i_idx, F_vec)
        np.add.at(self.a, j_idx, -F_vec)   # Newton's third law

    def move(self,):
        noise = self.noise_strength * np.random.normal(0, 1, size=(self.N,2))

        self.v += (self.a - self.gamma*self.v) * self.dt + noise
        self.pos += self.v * self.dt
        self.pos = np.mod(self.pos, self.grid_size)

    def update_T(self, cooling_rate=0.001, T_min = 0.01):
        self.T = max(self.T - cooling_rate * self.dt, T_min)
        self.noise_strength = np.sqrt(2 * self.gamma * self.T * self.dt)


    def update(self,frame):     
        if self.time == 0:
            self.spawn_points() # begin simulation with spawning of shape   

        log_step = (self.time % 20 == 0)
        self.compute_lj_force(log_step)
        self.move()

        self.update_T(self.cooling_rate)

        v_mean, v_sq_mean = self.compute_velocity_stats()
        self.v_mean_history.append(v_mean)
        self.v_sq_mean_history.append(v_sq_mean)

        self.update_point_patches()
        self.title.set_text(f"num_Shapes: {len(self.pos)}")
        self.title2.set_text(f"time: {self.time}")
        self.title3.set_text(f"temp: {self.T:4f}")
        
        self.time += 1
          
        return self.point_patches + [self.title, self.title2, self.title3]

    def plot_force(self,):
        plt.hist(sim.force_magnitudes, bins=100, log=True)   # log-scale y, since most forces are small, a few are huge
        plt.xlabel("Force magnitude")
        plt.ylabel("Count (log scale)")
        plt.title("Distribution of pairwise LJ force magnitudes")
        plt.show()

 
    def v_mean_correctness(self):
        theory = self.theoretical_v_squared()
        plt.plot(self.v_sq_mean_history, label=r"measured $\langle v^2 \rangle$")
        plt.axhline(theory, color='red', linestyle='--', label=r"theory $2k_BT/m$")
        # mean of measured v^2 from timestep 100 onward
        if len(self.v_sq_mean_history) > 400:
            tail = self.v_sq_mean_history[400:]
            tail_mean = np.mean(tail)
            plt.plot(range(400, len(self.v_sq_mean_history)), [tail_mean]*len(tail),
                    color='green', linewidth=2, label=f"mean (t>100) = {tail_mean:.3f}")
        plt.xlabel("timestep")
        plt.ylabel(r"$\langle v^2 \rangle$")
        plt.legend()
        plt.title("Equipartition check")
        print("mean v (should be ~0):", np.mean(self.v_mean_history, axis=0))
        plt.show()

    def compute_gr(self, n_bins=60, r_max=None):
        """Radial distribution function on the current configuration (self.pos)."""
        pos = self.pos
        N = self.N

        if r_max is None:
            r_max = self.grid_size / 2   # minimum image only valid up to here

        # --- all pairwise displacement vectors, vectorized (same pattern as compute_lj_force) ---
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]        # (N, N, 2)
        diff -= self.grid_size * np.round(diff / self.grid_size)     # minimum image

        dist = np.linalg.norm(diff, axis=-1)                         # (N, N)

        # --- only unique pairs (i<j), avoids self-pairs and double counting ---
        i_idx, j_idx = np.triu_indices(N, k=1)
        r = dist[i_idx, j_idx]                                       # (num_pairs,)

        # --- restrict to r < r_max ---
        r = r[r < r_max]

        # --- histogram ---
        bin_edges = np.linspace(0, r_max, n_bins + 1)
        counts, _ = np.histogram(r, bins=bin_edges)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        dr = bin_edges[1] - bin_edges[0]

        # --- normalize against ideal gas expectation ---
        area = self.grid_size**2
        rho = N / area
        shell_area = 2 * np.pi * bin_centers * dr
        ideal_counts = rho * shell_area * N / 2   # /2 since each pair counted once above but g(r) convention counts twice

        g_r = np.nan_to_num(counts / ideal_counts)

        return bin_centers, g_r


    def plot_gr(self, n_bins=100, r_max=None):
        bin_centers, g_r = self.compute_gr(n_bins=n_bins, r_max=r_max)

        plt.figure(figsize=(7,5))
        plt.plot(bin_centers, g_r, color='C0')
        plt.axhline(1, color='gray', linestyle='--', linewidth=1, label='ideal gas (g(r)=1)')
        plt.axvline(self.sigma * 2**(1/6), color='red', linestyle=':', linewidth=1,
                    label=r'theory min: $\sigma \cdot 2^{1/6}$')
        plt.xlabel("r")
        plt.ylabel("g(r)")
        plt.title("Radial Distribution Function")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_gr_with_shells(self, n_bins=100, r_max=None, max_shell=6):
        bin_centers, g_r = self.compute_gr(n_bins=n_bins, r_max=r_max)

        a = self.sigma * 2**(1/6)   # nearest-neighbor spacing (LJ force zero-crossing)

        # generate distinct triangular-lattice shell distances: a*sqrt(i^2+ij+j^2)
        shell_values = set()
        for i in range(-max_shell, max_shell+1):
            for j in range(-max_shell, max_shell+1):
                if i == 0 and j == 0:
                    continue
                n2 = i*i + i*j + j*j
                shell_values.add(n2)
        shell_values = sorted(shell_values)

        r_max_plot = bin_centers[-1] if r_max is None else r_max
        shell_radii = [a*np.sqrt(n2) for n2 in shell_values if a*np.sqrt(n2) <= r_max_plot]

        plt.figure(figsize=(8,5))
        plt.plot(bin_centers, g_r, color='C0', label='g(r)')
        plt.axhline(1, color='gray', linestyle='--', linewidth=1, label='ideal gas (g(r)=1)')

        for k, r_shell in enumerate(shell_radii):
            label = f'shell {k+1}: a·√{shell_values[k]}' if k < 3 else None  # avoid legend clutter past a few
            plt.axvline(r_shell, color='red', linestyle=':', linewidth=1, alpha=0.6, label=label)

        plt.xlabel("r")
        plt.ylabel("g(r)")
        plt.title("Radial Distribution Function with triangular-lattice shell markers")
        plt.legend()
        plt.grid(True)
        plt.show()

    def start_animation(self):
        """Start the animation."""
        self.ani = animation.FuncAnimation(
            self.fig, self.update, frames=100, init_func=self.init,
            blit=True, interval=100
        )
        plt.show()
        #save animation
        #self.plot_energy()
        #self.plot_force()

    def save_animation(self):
        """Start the animation and save it as a video."""
        self.ani = animation.FuncAnimation(
            self.fig, self.update, frames=int(self.sim_time), init_func=self.init,
            blit=True, interval=100
        )
        print("starting")
        # Save the animation as a video
        self.ani.save(animation_output.mp4', writer='ffmpeg', fps=40)  # You can adjust the fps and filename
        print("done")
        self.plot_energy()

"""
GAMMA (drag)      : how strongly the solvent couples to the particle — controls how fast
                     equilibrium is reached and how jittery motion looks, not the equilibrium itself.
TEMPERATURE (T)    : thermal energy of the bath — sets noise strength and is the main knob
                     controlling phase (solid/liquid/gas).
SIGMA (size)       : particle size / length scale — sets the equilibrium spacing between particles.
EPSILON (bond)     : depth of the attractive well — how much thermal energy is needed to break a bond.
"""

gamma = 1.3
T = 1.2
cooling_rate = 0.01
epsilon = 1.5
sigma = 3.0
dt = 0.1

num_points= 100
grid_size = 50


sim = Simulation(N=num_points, sim_time=20, grid_size=grid_size, dt=dt, gamma=gamma, T=T, cooling_rate=cooling_rate, epsilon=epsilon, sigma=sigma)
sim.start_animation()
#sim.save_animation()


sim.plot_force()
sim.v_mean_correctness()
sim.compute_gr()
sim.plot_gr_with_shells()
