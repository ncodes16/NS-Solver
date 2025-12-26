import numpy as np
import matplotlib.pyplot as plt

noise_levels = np.load('noise_levels.npy')
results = np.load('results.npy')

fig, ax = plt.subplots()
ax.plot(noise_levels, results, marker=',')
ax.set_xlabel('Noise Level')
ax.set_ylabel('Average Steps Before Crash')
ax.set_title('Solver Durability Against Noise Levels')
ax.grid(True)
plt.show()