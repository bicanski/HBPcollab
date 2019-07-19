import numpy as np
import matplotlib.pyplot as plt
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from BVC import PlaceCell, FastPlaceCell

from fastBVC import BVC, Environment

b_left = np.array(((0, 0), (0, 1000)), dtype=np.float64)
b_top = np.array(((0, 1000), (1000, 1000)), dtype=np.float64)
b_right = np.array(((1000, 0), (1000, 1000)), dtype=np.float64)
b_bottom = np.array(((0, 0), (1000, 0)), dtype=np.float64)

env = Environment([b_left, b_top, b_right, b_bottom])
pc = PlaceCell()
fpc = FastPlaceCell()

r = np.arange(1, 1000, 10, dtype=np.float64)

frm = fpc.compute_ratemap(env)
#rm = pc.compute_ratemap(r, r, env)


plt.imshow(frm, cmap=plt.cm.jet)
plt.show()