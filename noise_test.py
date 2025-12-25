"""Test the solver for its durability against different noise levels by detecting how many steps go before crash"""
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import num_solver
num_trials = 20
noise_levels = [1 + 0.2 * i for i in range(50)]
# noise_levels = [0.5, 1.0, 10.0, 100.0]
T_max = 100
N=64
nu = 0.01
dt = 0.01
results = []
X, Y = np.meshgrid(np.linspace(0, 2 * np.pi, N, False), np.linspace(0, 2 * np.pi, N, False))
for level in noise_levels:
    result_list = np.ndarray(num_trials)
    psi = np.sin(X) * np.sin(Y)
    omega = 2 * psi
    for i in (range(num_trials)):
        noise = level * np.random.normal(0, 1, (N, N))
        psi += noise
        omega += noise
        Model = num_solver.Solver(N, 2*np.pi, dt, nu, T_max, psi, omega)
        test = Model.run(debug = True)
        if isinstance(test, int):
            result_list[i] = test
        else:
            result_list[i] = T_max/dt
    results.append(result_list.mean())
    if result_list.mean() < 10:
        break
results = np.pad(results, (0, len(noise_levels) - len(results)))
fig, ax = plt.subplots(figsize = (6, 4))
ax.plot(noise_levels, results, label='Max step reached vs noise level')
ax.set_xlabel("Noise Level (%)")
ax.set_ylabel("Max step reached")
ax.legend()
ax.grid()

now = datetime.now().strftime("%Y.%m.%d._%H.%M.%S.")
plt.savefig(f"noise_test_{now}.png")