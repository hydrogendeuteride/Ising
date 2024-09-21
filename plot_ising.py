import numpy as np
import matplotlib.pyplot as plt

spins = np.loadtxt('cmake-build-debug/spins.txt', dtype=np.int32)

cmap = plt.colormaps.get_cmap('bwr')

plt.imshow(spins, cmap=cmap, interpolation='none')
plt.colorbar(label='Spin value')
plt.title('Ising Model Simulation')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
