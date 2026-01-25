from num_solver import Solver
import numpy as np
import matplotlib.pyplot as plt

def accurate_solution(N, Lxy, nu, T, dt, k):
    accurate_psis: np.ndarray = np.zeros((len(times), N, N), dtype=np.float64)
    for i, t in enumerate(times):
        accurate_psis[i] = np.sin(k*X) * np.sin(k*Y) * np.exp(-2 * k * k * nu * t)
    return accurate_psis
def initial_conditions(N, k, noise_amp=0):
    psi = np.sin(k*X) * np.sin(k*Y)  # TG vortex initial condition
    omega = 2 * k * k * np.sin(k*X) * np.sin(k*Y) # vorticity initial condition
    noise = noise_amp * np.random.normal(size=(N, N), loc=0, scale=1)
    psi += noise
    omega += noise
    return psi, omega
N = 64
X, Y = np.meshgrid(np.linspace(0, 2 * np.pi, N, False), np.linspace(0, 2 * np.pi, N, False))
# psi, omega = initial_conditions(N, 1, 0.02)
# psi, omega = 1/(4 * np.pi) * Y * Y, np.array(/np.pi * np.ones((N, N)), dtype=np.float64)
#psi, omega = -np.cos(X) + 0.05 * np.random.normal(size=(N, N), loc = 0, scale=1), np.cos(X) + 0.05 * np.random.normal(size=(N, N), loc = 0, scale=1)

psi, omega = initial_conditions(N, 1, 0.0)
# psi = np.random.normal(size=(N, N), loc = 0, scale=0.05)
# psi = np.sin(4* Y)
nu = 0.01
T = 100
dt = 0.01
TG_vortex = Solver(N, 2 * np.pi, dt, nu, T, psi, False, 0.25, 8)
psis, qs = TG_vortex.run()

times = np.linspace(0, T, int(T / dt), endpoint = True, dtype=np.float64)
accurate_psis = accurate_solution(N, 2 * np.pi, nu, T, dt, 1)

errors = psis - accurate_psis
nrmse = np.sqrt(np.mean(np.square(errors), axis=(1, 2))/(np.mean(np.square(accurate_psis), axis=(1, 2))))

# C_MAX = 0.4
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# im1 = ax[0].imshow(psis[-1], origin='lower', cmap='viridis', extent=(0, TG_vortex.Lxy, 0, TG_vortex.Lxy), vmin=-C_MAX, vmax=C_MAX)
# im1 = ax[0].imshow(psis[int(len(psis)/2)], origin='lower', cmap='viridis', extent=(0, TG_vortex.Lxy, 0, TG_vortex.Lxy), vmin=-0.5, vmax=0.5)
# im2 = ax[1].imshow(accurate_psis[int(len(psis)/2)], origin='lower', cmap='viridis', extent=(0, TG_vortex.Lxy, 0, TG_vortex.Lxy), vmin=-0.5, vmax=0.5)
# im2 = ax[1].imshow(accurate_psis[-1], origin='lower', cmap='viridis', extent=(0, TG_vortex.Lxy, 0, TG_vortex.Lxy), vmin=-C_MAX, vmax=C_MAX)
# ax[0].set_title('Numerical Solution at t=100')
# ax[1].set_title('Analytical Solution at t=100')
# plt.colorbar(im1, ax=ax[0])
# plt.colorbar(im2, ax=ax[1])
# Plot RMSE in a separate new figure (don't reuse figure 1)
# fig2, ax2 = plt.subplots(figsize=(6, 4))
# ax2.set_yscale('log')
# ax2.plot(times, nrmse, label='NRMSE over time')
# ax2.set_xlabel('Time')
# ax2.set_ylabel('NRMSE')
# ax2.set_title('Normalized Root Mean Square Error of Numerical Solution')
# ax2.legend()
# ax2.grid()

TG_vortex.animate_snapshots(psis, 50)
plt.show()