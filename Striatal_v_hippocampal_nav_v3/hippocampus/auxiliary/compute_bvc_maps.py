from hippocampus.environments import BlockingStudy
from hippocampus.BVC import FastPlaceCell
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from definitions import ROOT_FOLDER
from hippocampus.fastBVC import BVC
# from hippocampus.slowBVC import BVC

map_folder = os.path.join(ROOT_FOLDER, 'data', 'bvc_maps')
figure_folder = os.path.join(ROOT_FOLDER, 'results', 'figures', 'bvc_pc_maps')
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

en = BlockingStudy(6)

tqdm.write('Computing BVC ratemaps for both boundaries present')
en.compute_bvc_maps_multiproc()

tqdm.write('Computing BVC ratemaps for only right boundary')
en.remove_left_boundary()
en.compute_bvc_maps_multiproc()

tqdm.write('Computing BVC ratemaps for only left boundary')
en.remove_right_boundary()
en.compute_bvc_maps_multiproc()


env_both = BlockingStudy(6)
env_onlyleft = BlockingStudy(6)
env_onlyleft.remove_right_boundary()
env_onlyright = BlockingStudy(6)
env_onlyright.remove_left_boundary()

# Now we want a matrix with states as rows and place cell responses on the columns, for each condition
n_place_cells = 150

place_cells = [FastPlaceCell() for _ in range(n_place_cells)]

conditions = ['both', 'right', 'left']

envs = [env_both, env_onlyleft, env_onlyright]

for e in tqdm(envs, desc='Conditions'):
    place_cell_activity_mat = np.empty((en.nr_states, n_place_cells))

    firing_peaks = []
    for i, pc in enumerate(tqdm(place_cells, desc='Place cells', leave=False)):

        rm = pc.compute_ratemap(e)

        if e.boundaries_present == 'both':
            peak = rm.argmax()
            it = 0
            while peak in firing_peaks and it < 10:
                pc = FastPlaceCell()
                place_cells[i] = pc
                rm = pc.compute_ratemap(e)
                peak = rm.argmax()
                it += 1
            firing_peaks.append(peak)

        rm = rm / rm.max()

        place_cell_activity_mat[:, i] = rm

        fig, ax = plt.subplots()
        en.plot_grid(ax=ax, c_mappable=place_cell_activity_mat[:, i], c_map='viridis')
        ax.axis('off')
        plt.savefig(os.path.join(figure_folder, 'pc{}_{}'.format(i, e.boundaries_present)))
        plt.close()


    np.save(os.path.join(map_folder, 'place_cell_activity_{}_boundary.npy'.format(e.boundaries_present)),
            place_cell_activity_mat)

