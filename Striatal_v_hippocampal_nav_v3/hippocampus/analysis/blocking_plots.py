import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from definitions import ROOT_FOLDER
import os
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle
import matplotlib.patches

from hippocampus.plotting import tsplot_boot

results_folder = os.path.join(ROOT_FOLDER, 'results', 'blocking')
figure_folder = os.path.join(ROOT_FOLDER, 'figures')

boundary_blocking_data = np.load(os.path.join(results_folder, 'boundary_blocking_results.npy'))
landmark_blocking_data = np.load(os.path.join(results_folder, 'landmark_blocking_results.npy'))


# define colours
colour_palette = sns.color_palette()
cue1_colour = colour_palette[8]
cue2_colour = colour_palette[9]


fig = plt.figure()

gs = GridSpec(1, 2)

inner = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0], wspace=0.01, hspace=0.1)

ax1 = plt.Subplot(fig, inner[0, 0])
ax1.axis('equal')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.text(-.1, .7, 'Initial\nlearning', fontsize=12, transform=ax1.transAxes, ha='left', va='top')
ax1.scatter([.65], [.5], marker='P', color=cue1_colour, s=400, linestyle='None')
ax1.text(.65, .43, 'L1', va='top', ha='center', color=cue1_colour)
ax1.scatter([.8], [.3], marker='P', color=cue2_colour, s=400, linestyle='dotted', facecolors='none')
platform = Circle((.45, .3), .05, fill=False, linestyle='--')
ax1.add_artist(platform)


ax2 = plt.Subplot(fig, inner[1, 0])
ax2.axis('equal')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.text(-.1, .7, 'Compound\nlearning', fontsize=12, transform=ax2.transAxes, ha='left', va='top')
ax2.scatter([.65], [.5], marker='P', color=cue1_colour, s=400, linestyle='None')
ax2.scatter([.8], [.3], marker='P', color=cue2_colour, s=400)
platform = Circle((.45, .3), .05, fill=False, linestyle='--')
ax2.add_artist(platform)
ax2.text(.65, .43, 'L1', va='top', ha='center', color=cue1_colour)
ax2.text(.8, .23, 'L2', va='top', ha='center', color=cue2_colour)


ax3 = plt.Subplot(fig, inner[2, 0])
ax3.axis('equal')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.text(-.1, .7, 'Testing', fontsize=12, transform=ax3.transAxes, ha='left', va='top')
ax3.scatter([.65], [.5], marker='P', color=cue1_colour, s=400, linestyle='dotted', facecolors='none')
ax3.scatter([.8], [.3], marker='P', color=cue2_colour, s=400)
platform = Circle((.45, .3), .05, fill=False, linestyle='--')
ax3.add_artist(platform)
ax3.text(.8, .23, 'L2', va='top', ha='center', color=cue2_colour)

for ax in [ax1, ax2, ax3]:
    ax.text(.45, .23, 'Platform', va='top', ha='center', fontstyle='italic')

for ax in [ax2, ax3]:
    ax.axhline(y=.7, xmin=0.3, xmax=.9, color='k', linestyle='dashed')

fig.add_subplot(ax1)
fig.add_subplot(ax2)
fig.add_subplot(ax3)

results_ax = fig.add_subplot(gs[1])

tsplot_boot(results_ax, boundary_blocking_data, color=colour_palette[0])
tsplot_boot(results_ax, landmark_blocking_data, color=colour_palette[3])

results_ax.set_ylabel('Time steps', fontsize=12)
results_ax.set_xlabel('Trials', fontsize=12)
# Hide the right and top spines
results_ax.spines['right'].set_visible(False)
results_ax.spines['top'].set_visible(False)
# Show landmark timelines
results_ax.axhline(y=results_ax.get_ylim()[1] * .90, xmin=0, xmax=.6667, color=cue1_colour, LineWidth=5)
results_ax.axhline(y=results_ax.get_ylim()[1] * .85, xmin=.3333, xmax=1, color=cue2_colour, LineWidth=5)
results_ax.text(results_ax.get_xlim()[1] * .01, results_ax.get_ylim()[1] * .92, 'Landmark 1 present', color=cue1_colour)
results_ax.text(results_ax.get_xlim()[1] * .99, results_ax.get_ylim()[1] * .83, 'Landmark 2 present',
                ha='right', va='top', color=cue2_colour)


results_ax.legend(['HPC', 'DLS'], loc=(.2, .6), fontsize=12)

# Add figure labels
ax1.text(-.15, 1.15, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
results_ax.text(-.15, 1.05, 'B', transform=results_ax.transAxes,
                     fontsize=16, fontweight='bold', va='top', ha='right')


# Now some arrows connecting them
# 1. Get transformation operators for axis and figure
ax1tr = ax1.transData  # Axis 1 -> Display
ax2tr = ax2.transData  # Axis 2 -> Display
ax3tr = ax3.transData  # Axis 3 -> Display
figtr = fig.transFigure.inverted()  # Display -> Figure
# 2. Transform arrow start point from axis 0 to figure coordinates
# 2. Transform arrow start point from axis 0 to figure coordinates
ptB = figtr.transform(ax1tr.transform((0, .4)))
# 3. Transform arrow end point from axis 1 to figure coordinates
ptE = figtr.transform(ax2tr.transform((0, .95)))

ptB2 = figtr.transform(ax2tr.transform((0, .4)))
# 3. Transform arrow end point from axis 1 to figure coordinates
ptE2 = figtr.transform(ax3tr.transform((0, .95)))

# 4. Create the patch
arrow = matplotlib.patches.FancyArrowPatch(
    ptB, ptE, transform=fig.transFigure, fc="k", arrowstyle='simple', alpha=1., mutation_scale=40.)
fig.patches.append(arrow)
arrow2 = matplotlib.patches.FancyArrowPatch(
    ptB2, ptE2, transform=fig.transFigure, fc="k", arrowstyle='simple', alpha=1., mutation_scale=40.)
fig.patches.append(arrow2)

for ex in ['.png', '.svg', '.pdf']:
    plt.savefig(os.path.join(figure_folder, 'Blocking{}'.format(ex)), dpi=fig.dpi,
                bbox_inches='tight', pad_inches=0.5)

#plt.tight_layout()

plt.show()
