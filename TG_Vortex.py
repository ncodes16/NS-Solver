from num_solver import Solver
import numpy as np
import matplotlib.pyplot as plt

def accurate_solution_at_step(N, Lxy, nu, T, dt, k, step, times):
    """Compute accurate solution at a single timestep"""
    accurate_psi: np.ndarray = np.zeros((N, N), dtype=np.float64)
    t = times[step]
    accurate_psi = -(1 / nu) * (1 / k) ** 3 * np.cos(k * Y)
    # accurate_psi = np.sin(X) * np.sin(Y) * np.exp(-2 * nu * t)
    return accurate_psi
def initial_conditions(N, k, noise_amp=0):
    psi = np.sin(k*X) * np.sin(k*Y)  # TG vortex initial condition
    omega = 2 * k * k * np.sin(k*X) * np.sin(k*Y) # vorticity initial condition
    noise = noise_amp * np.random.normal(size=(N, N), loc=0, scale=1)
    psi += noise
    omega += noise
    return psi, omega
N = 128
X, Y = np.meshgrid(np.linspace(0, 2 * np.pi, N, False), np.linspace(0, 2 * np.pi, N, False), indexing='ij')
psi, omega = initial_conditions(N, 2, 0.0)
# psi, omega = 1/(4 * np.pi) * Y * Y, np.array(/np.pi * np.ones((N, N)), dtype=np.float64)
#psi, omega = -np.cos(X) + 0.05 * np.random.normal(size=(N, N), loc = 0, scale=1), np.cos(X) + 0.05 * np.random.normal(size=(N, N), loc = 0, scale=1)

# psi, omega = initial_conditions(N, 1, 0.0)
# psi = np.random.normal(size=(N, N), loc = 0, scale=0.05)
# psi = np.sin(4* Y)
nu = 0.01
T = 100
dt = 0.0001
TG_vortex = Solver(N, 2 * np.pi, dt, nu, T, psi, False, 1, 4)

import gc
import os

output_dir = './snapshots'
os.makedirs(output_dir, exist_ok=True)

# Run solver with memory-efficient snapshot saving
num_steps = int(T / dt)
num_snapshots = 100
snapshot_indices = [i * num_steps // num_snapshots for i in range(num_snapshots)]
snapshot_labels = [f"{i} %" for i in range(num_snapshots)]

times = np.linspace(0, T, num_steps, endpoint=True, dtype=np.float64)

# Run solver - accurate solution computed on-the-fly for each snapshot
psis_final, qs_final = TG_vortex.run_with_snapshots(snapshot_indices, times, 
                                                      accurate_solution_at_step, N, 2*np.pi, nu, T, dt, 4,
                                                      output_dir, C_MAX=1)

print(f"Snapshots saved to {output_dir}/")

# TG_vortex.animate_snapshots(psis, 50)
plt.show()