import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, show
import tables

from compare_variance_residual.common_utils.hdf_utils import map_to_flat


def plot_model_comparison(corrs1, corrs2, name1, name2, thresh=0.35):
    fig = figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    good1 = corrs1 > thresh
    good2 = corrs2 > thresh
    better1 = corrs1 > corrs2
    # both = np.logical_and(good1, good2)
    neither = np.logical_not(np.logical_or(good1, good2))
    only1 = np.logical_and(good1, better1)
    only2 = np.logical_and(good2, np.logical_not(better1))

    ptalpha = 0.3
    ax.plot(corrs1[neither], corrs2[neither], 'ko', alpha=ptalpha)
    # ax.plot(corrs1[both], corrs2[both], 'go', alpha=ptalpha)
    ax.plot(corrs1[only1], corrs2[only1], 'ro', alpha=ptalpha)
    ax.plot(corrs1[only2], corrs2[only2], 'bo', alpha=ptalpha)

    lims = [-0.5, 1.0]

    ax.plot([thresh, thresh], [lims[0], thresh], 'r-')
    ax.plot([lims[0], thresh], [thresh, thresh], 'b-')

    ax.text(lims[0] + 0.05, thresh, "$n=%d$" % np.sum(good2), horizontalalignment="left", verticalalignment="bottom")
    ax.text(thresh, lims[0] + 0.05, "$n=%d$" % np.sum(good1), horizontalalignment="left", verticalalignment="bottom")

    ax.plot(lims, lims, '-', color="gray")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)

    show()
    return fig


import matplotlib.colors

bwr = matplotlib.colors.LinearSegmentedColormap.from_list("bwr", ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)))
bkr = matplotlib.colors.LinearSegmentedColormap.from_list("bkr", ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
bgr = matplotlib.colors.LinearSegmentedColormap.from_list("bgr", ((0.0, 0.0, 1.0), (0.5, 0.5, 0.5), (1.0, 0.0, 0.0)))


def plot_model_comparison2(corrFile1, corrFile2, name1, name2, thresh=0.35):
    fig = figure(figsize=(9, 10))
    # ax = fig.add_subplot(3,1,[1,2], aspect="equal")
    ax = fig.add_axes([0.25, 0.4, 0.6, 0.5], aspect="equal")

    corrs1 = tables.openFile(corrFile1).root.semcorr.read()
    corrs2 = tables.openFile(corrFile2).root.semcorr.read()
    maxcorr = np.clip(np.vstack([corrs1, corrs2]).max(0), 0, thresh) / thresh
    corrdiff = (corrs1 - corrs2) + 0.5
    colors = (bgr(corrdiff).T * maxcorr).T
    colors[:, 3] = 1.0  ## Don't scale alpha

    ptalpha = 0.8
    ax.scatter(corrs1, corrs2, s=10, c=colors, alpha=ptalpha, edgecolors="none")
    lims = [-0.5, 1.0]

    ax.plot([thresh, thresh], [lims[0], thresh], color="gray")
    ax.plot([lims[0], thresh], [thresh, thresh], color="gray")

    good1 = corrs1 > thresh
    good2 = corrs2 > thresh
    ax.text(lims[0] + 0.05, thresh, "$n=%d$" % np.sum(good2), horizontalalignment="left", verticalalignment="bottom")
    ax.text(thresh, lims[0] + 0.05, "$n=%d$" % np.sum(good1), horizontalalignment="left", verticalalignment="bottom")

    ax.plot(lims, lims, '-', color="gray")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(name1 + " model")
    ax.set_ylabel(name2 + " model")

    fig.canvas.draw()
    show()
    ## Add over-under comparison
    # ax_left = ax.get_window_extent()._bbox.x0
    # ax_right = ax.get_window_extent()._bbox.x1
    # ax_width = ax_right-ax_left
    # print ax_left, ax_right
    # ax2 = fig.add_axes([ax_left, 0.1, ax_width, 0.2])
    ax2 = fig.add_axes([0.25, 0.1, 0.6, 0.25])  # , sharex=ax)
    # ax2 = fig.add_subplot(3, 1, 3)
    # plot_model_overunder_comparison(corrs1, corrs2, name1, name2, thresh=thresh, ax=ax2)
    plot_model_histogram_comparison(corrs1, corrs2, name1, name2, thresh=thresh, ax=ax2)

    fig.suptitle("Model comparison: %s vs. %s" % (name1, name2))
    show()
    return fig


def plot_model_overunder_comparison(corrs1, corrs2, name1, name2, thresh=0.35, ax=None):
    """Plots over-under difference between two models.
    """
    if ax is None:
        fig = figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

    maxcorr = max(corrs1.max(), corrs2.max())
    vals = np.linspace(0, maxcorr, 500)
    overunder = lambda c: np.array([np.sum(c > v) - np.sum(c < -v) for v in vals])

    ou1 = overunder(corrs1)
    ou2 = overunder(corrs2)

    oud = ou2 - ou1

    ax.fill_between(vals, 0, np.clip(oud, 0, 1e9), facecolor="blue")
    ax.fill_between(vals, 0, np.clip(oud, -1e9, 0), facecolor="red")

    yl = np.max(np.abs(np.array(ax.get_ylim())))
    ax.plot([thresh, thresh], [-yl, yl], '-', color="gray")
    ax.set_ylim(-yl, yl)
    ax.set_xlim(0, maxcorr)
    ax.set_xlabel("Voxel correlation")
    ax.set_ylabel("%s better           %s better" % (name1, name2))

    show()
    return ax


def plot_model_histogram_comparison(corrs1, corrs2, name1, name2, thresh=0.35, ax=None):
    """Plots over-under difference between two models.
    """
    if ax is None:
        fig = figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

    maxcorr = max(corrs1.max(), corrs2.max())
    nbins = 100
    hist1 = np.histogram(corrs1, nbins, range=(-1, 1))
    hist2 = np.histogram(corrs2, nbins, range=(-1, 1))

    ouhist1 = hist1[0][nbins // 2:] - hist1[0][:nbins // 2][::-1]
    ouhist2 = hist2[0][nbins // 2:] - hist2[0][:nbins // 2][::-1]

    oud = ouhist2 - ouhist1
    bwidth = 2.0 / nbins
    barlefts = hist1[1][nbins // 2:-1]

    # ax.fill_between(vals, 0, np.clip(oud, 0, 1e9), facecolor="blue")
    # ax.fill_between(vals, 0, np.clip(oud, -1e9, 0), facecolor="red")

    ax.bar(barlefts, np.clip(oud, 0, 1e9), bwidth, facecolor="blue")
    ax.bar(barlefts, np.clip(oud, -1e9, 0), bwidth, facecolor="red")

    yl = np.max(np.abs(np.array(ax.get_ylim())))
    ax.plot([thresh, thresh], [-yl, yl], '-', color="gray")
    ax.set_ylim(-yl, yl)
    ax.set_xlim(0, maxcorr)
    ax.set_xlabel("Voxel correlation")
    ax.set_ylabel("%s better           %s better" % (name1, name2))

    show()
    return ax


def plot_model_comparison_rois(corrs1, corrs2, name1, name2, roivoxels, roinames, thresh=0.35):
    """Plots model correlation comparisons per ROI.
    """
    fig = figure()
    ptalpha = 0.3

    for ri in range(len(roinames)):
        ax = fig.add_subplot(4, 4, ri + 1)
        ax.plot(corrs1[roivoxels[ri]], corrs2[roivoxels[ri]], 'bo', alpha=ptalpha)
        lims = [-0.3, 1.0]
        ax.plot(lims, lims, '-', color="gray")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(roinames[ri])

    show()
    return fig


def save_table_file(filename, filedict):
    """Saves the variables in [filedict] in a hdf5 table file at [filename].
    """
    hf = tables.openFile(filename, mode="w", title="save_file")
    for vname, var in filedict.items():
        hf.createArray("/", vname, var)
    hf.close()


def flatmap_subject_voxel_data(subject_number, data):
    # Get path to data files
    fdir = os.path.abspath('../data')
    # Map to subject flatmap
    map_file = os.path.join(fdir, f'subject{subject_number}_mappers.hdf')
    flatmap = map_to_flat(data, map_file)
    # Plot flatmap
    fig, ax = plt.subplots()
    _ = ax.imshow(flatmap, cmap='inferno')
    ax.axis('off')
    # add legend


    plt.show()
    return fig
