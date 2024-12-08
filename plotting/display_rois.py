import os

import numpy as np
import voxelwise_tutorials.viz as viz
from matplotlib import pyplot as plt
from voxelwise_tutorials.io import load_hdf5_array

from compare_variance_residual.common_utils.feature_utils import get_prediction_path


def plot_flatmap_from_mapper(voxels, mapper_file, ax=None, alpha=0.7,
                             cmap='inferno', vmin=None, vmax=None,
                             with_curvature=True, with_rois=True,
                             with_colorbar=True,
                             colorbar_location=(.4, .9, .2, .05)):
    """Plot a flatmap from a mapper file, with 1D data.

    This function is equivalent to the pycortex functions:
    cortex.quickshow(cortex.Volume(voxels, ...), ...)

    Note that this function does not have the full capability of pycortex,
    since it is based on flatmap mappers and not on the original brain
    surface of the subject.

    Parameters
    ----------
    voxels : array of shape (n_voxels, )
        Data to be plotted.
    mapper_file : str
        File name of the mapper.
    ax : matplotlib Axes or None.
        Axes where the figure will be plotted.
        If None, a new figure is created.
    alpha : float in [0, 1], or array of shape (n_voxels, )
        Transparency of the flatmap.
    cmap : str
        Name of the matplotlib colormap.
    vmin : float or None
        Minimum value of the colormap. If None, use the 1st percentile of the
        `voxels` array.
    vmax : float or None
        Minimum value of the colormap. If None, use the 99th percentile of the
        `voxels` array.
    with_curvature : bool
        If True, show the curvature below the data layer.
    with_rois : bool
        If True, show the ROIs labels above the data layer.
    colorbar_location : [left, bottom, width, height]
        Location of the colorbar. All quantities are in fractions of figure
        width and height.

    Returns
    -------
    ax : matplotlib Axes
        Axes where the figure has been plotted.
    """
    # create a figure
    if ax is None:
        flatmap_mask = load_hdf5_array(mapper_file, key='flatmap_mask')
        figsize = np.array(flatmap_mask.shape) / 100.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis('off')

    # process plotting parameters
    vmin = np.percentile(voxels, 1) if vmin is None else vmin
    vmax = np.percentile(voxels, 99) if vmax is None else vmax
    if isinstance(alpha, np.ndarray):
        alpha = viz.map_voxels_to_flatmap(alpha, mapper_file)

    # plot the data
    image = viz.map_voxels_to_flatmap(voxels, mapper_file)
    cimg = ax.imshow(image, aspect='equal', zorder=1, alpha=alpha, cmap=cmap,
                     vmin=vmin, vmax=vmax)

    if with_colorbar:
        try:
            cbar = ax.inset_axes(colorbar_location)
        except AttributeError:  # for matplotlib < 3.0
            cbar = ax.figure.add_axes(colorbar_location)
        colorbar = ax.figure.colorbar(cimg, cax=cbar, orientation='horizontal')
        colorbar.ax.set_title("Pearson Correlation of Y_true and Y_pred", fontsize=14)

    # plot additional layers if present
    viz._plot_addition_layers(ax=ax, n_voxels=voxels.shape[0],
                          mapper_file=mapper_file,
                          with_curvature=with_curvature, with_rois=with_rois)

    return ax


def display_flatmap(language_model: str, feature: str, modality: str, subject: int, low_level_feature=None, layer=None,
                    title = None, save_fig: bool = False):
    path = get_prediction_path(language_model, feature, modality, subject, low_level_feature, layer)
    correlation = np.nan_to_num(np.load(path, allow_pickle=True))
    mapper_path = os.path.join("../data", f"subject{subject:02}_mappers.hdf")

    flatmap_mask = load_hdf5_array(mapper_path, key='flatmap_mask')
    figsize = np.array(flatmap_mask.shape) / 100.
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis('off')
    plot_flatmap_from_mapper(correlation, mapper_path, ax=ax, with_curvature=False, alpha=1, cmap='inferno',
                                 vmin=0,
                                 # vmin=np.min(correlation),
                                 vmax=np.max(correlation),
                                 colorbar_location=[0.75, 0.05, 0.2, 0.05])

    fig.suptitle(title, fontsize=16)
    if save_fig:
        path = os.path.relpath(path, start="../predictions")
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/flatmap_{path.replace(os.path.sep, '_')}.png")
    plt.show()


if __name__ == "__main__":
    language_model = "bert"
    feature = "semantic"
    modality = "reading"
    subject = 1
    layer = 9
    low_level_feature = "letters"

    title = f"{feature} stimulus\nsubject {subject} - {modality} - layer {layer}\nalphas = grid search of 10 values from 10^0 to 10^4"
    display_flatmap(language_model, feature, modality, subject, low_level_feature, layer, title, True)
