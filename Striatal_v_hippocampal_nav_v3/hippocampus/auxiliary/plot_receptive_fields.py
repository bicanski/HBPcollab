import numpy as np
import matplotlib.pyplot as plt

from hippocampus.agents import LandmarkCells
from hippocampus.utils import angle_to_landmark
from hippocampus.environments import HexWaterMaze


g = HexWaterMaze(6)
location = g.grid.cart_coords[g.platform_state]
my_location = (0, 0)
my_orientation = 60
angle = angle_to_landmark(my_location, location, my_orientation)
LC = LandmarkCells()
xs = np.linspace(-180, 180, 1000)
responses = np.zeros((len(xs), LC.n_cells))

for i, x in enumerate(xs):
    responses[i, :] = LC.compute_response(i)

for col in responses.T:
    plt.plot(xs, col)
plt.show()
