import numpy as np
from matplotlib import pyplot as plt


def tight_subplot_2x5(pyramids_gaussian, pyramids_laplace, vmin=None, vmax=None):
    """
    vmin / vmax:
        1) None: will use min/max value of each array for its plot;
        2) int: constant value for all plots;
        3) list of shape 2x5: specified value for each plot
    """
    def get_v(v, array, index, mode):
        if v is None:
            return array.min() if mode == 'min' else array.max()
        elif isinstance(v, int):
            return v
        else:
            return v[index[0]][index[1]]

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 6), sharex=True, sharey=True)
    for i in range(5):
        axes[0, i].imshow(pyramids_gaussian[i], cmap='gray',
                          vmin=get_v(vmin, pyramids_gaussian[i], [0, i], 'min'),
                          vmax=get_v(vmax, pyramids_gaussian[i], [0, i], 'max'))
        axes[0, i].xaxis.set_visible(False)
        axes[0, i].yaxis.set_visible(False)
    for i in range(5):
        if i < len(pyramids_laplace):
            axes[1, i].imshow(pyramids_laplace[i], cmap='gray',
                          vmin=get_v(vmin, pyramids_laplace[i], [1, i], 'min'),
                          vmax=get_v(vmax, pyramids_laplace[i], [1, i], 'max'))
        else:
            axes[1, i].imshow(np.ones_like(pyramids_laplace[0]) * 255, cmap='gray', vmin=0, vmax=255)
            axes[1, i].axis('off')
        axes[1, i].xaxis.set_visible(False)
        axes[1, i].yaxis.set_visible(False)
    plt.close()
    return fig