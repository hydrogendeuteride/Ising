import numpy as np
import matplotlib.pyplot as plt

L = 50
T = 2.239
steps = 10

spins = np.random.choice([-1, 1], size=(L, L))

def metropolis_step(spins, beta):
    for i in range(L):
        for j in range(L):
            x, y = np.random.randint(0, L, size=2)
            s = spins[x, y]

            neighbors = spins[(x+1)%L, y] + spins[(x-1)%L, y] + spins[x, (y+1)%L] + spins[x, (y-1)%L]
            delta_E = 2 * s * neighbors

            if delta_E < 0 or np.random.rand() < np.exp(-delta_E * beta):
                spins[x, y] *= -1

beta = 1.0 / T
for step in range(steps):
    metropolis_step(spins, beta)

cmap = plt.colormaps.get_cmap('bwr')

plt.imshow(spins, cmap=cmap, interpolation='none')
plt.colorbar(label='Spin value')
plt.title('Ising Model Simulation')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
