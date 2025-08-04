from num_solver import Solver
import numpy as np
import matplotlib.pyplot as plt

N = 128
X, Y = np.meshgrid(np.linspace(0, 2 * np.pi, N, False), np.linspace(0, 2 * np.pi, N, False))
psi = np.sin(X) * np.sin(Y)  # TG vortex initial condition
omega = 2 * np.sin(X) * np.sin(Y) # vorticity initial condition
nu = 1
T = 1
dt = 0.001
TG_vortex = Solver(N, 2 * np.pi, dt, nu, T, psi, omega)
psis = TG_vortex.run()

times = np.linspace(0, T, int(T / dt), endpoint = True, dtype=np.float64)
accurate_psis: np.ndarray = np.zeros((len(times), N, N), dtype=np.float64)
for i, t in enumerate(times):
    accurate_psis[i] = np.sin(X) * np.sin(Y) * np.exp(-2 * nu * t)

errors = psis - accurate_psis
rmse = np.sqrt(np.mean(np.square(errors), axis=(1, 2)))

# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# im1 = ax[0].imshow(psis[-1], origin='lower', cmap='viridis', extent=(0, TG_vortex.Lxy, 0, TG_vortex.Lxy))
# im2 = ax[1].imshow(accurate_psis[-1], origin='lower', cmap='viridis', extent=(0, TG_vortex.Lxy, 0, TG_vortex.Lxy))
# ax[0].set_title('Numerical Solution at t=1')
# ax[1].set_title('Analytical Solution at t=1')
# plt.colorbar(im1, ax=ax[0])
# plt.colorbar(im2, ax=ax[1])
# plt.imshow(errors[0], origin='lower', cmap='hot', extent=(0, TG_vortex.Lxy, 0, TG_vortex.Lxy))
plt.figure(1)
plt.plot(times, rmse, label='RMSE over time')
plt.xlabel('Time')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error of Numerical Solution')
plt.legend()
plt.grid()
TG_vortex.animate_snapshots(psis, 10)
plt.show()