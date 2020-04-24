import matplotlib.pyplot as plt

from hippocampus.agents import LandmarkCells

if __name__ == '__main__':
    import os
    from definitions import ROOT_FOLDER
    from tqdm import tqdm

    results_folder = os.path.join(ROOT_FOLDER, 'results/figures/receptive_fields')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    lm = LandmarkCells()

    for i in tqdm(range(lm.n_cells)):
        fig = plt.figure()
        ax = lm.plot_receptive_field(i)
        plt.savefig(os.path.join(results_folder, 'rf{}.png'.format(i)))
        plt.close(fig)
