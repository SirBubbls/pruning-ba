import numpy as np
import keras
import matplotlib.pyplot as plt

fig_font_size = 30

def prune(model, sparsity):
    new_weights = []
    for layer in model.get_weights():
        new_weights.append(weight_prune_layer(layer, sparsity))
    model.set_weights(new_weights)
    return model

def weight_prune_layer(layer, sparsity):
    """
    Prunes the lowest p% of
    args:
      - layer: The target layer that should get new[0::2][layer]
      - sparsity: The percentage of desired sparsity
    """

    original_shape = layer.shape
    layer = layer.flatten()

    s = np.argsort(np.abs(layer.flatten()))
    layer[s > int(len(layer) * (1 - sparsity))] = 0
    return layer.reshape(original_shape)

def visualize_kernel(kernel, fig):
    """
    Displays kernel weight as a plot.
    returns: plot of weights
    """
    im = fig.imshow(np.abs(kernel),
               interpolation='none',
               cmap='inferno',
            clim=0)
    plt.colorbar(im, ax=fig)
    return fig

def unit_prune_layer(layer, sparsity):
    shape = layer.shape
    layer = np.array(layer)
    neuron_weights = abs(layer).sum(0)

    layer[:, np.argsort(neuron_weights) < int(len(neuron_weights) * sparsity)] = 0

    return layer.reshape(shape)

def quad_plot(new, old, output):
    fig = plt.figure(constrained_layout=True, figsize=(24,12))
    gs = plt.GridSpec(2, 2, figure=fig)
    ax2 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])

    ax4.hist(old.flatten(), density=False, width=.1, bins='auto')
    ax4.set_title("Original Weight Distribution", fontsize=fig_font_size)

    ax5.hist(new.flatten(), density=False, width=.1, bins='auto')
    ax5.set_title("Pruned Weight Distribution", fontsize=fig_font_size)

    visualize_kernel(old, ax2)
    ax2.set_title("Original Weight Plot", fontsize=fig_font_size)

    visualize_kernel(new, ax3)
    ax3.set_title("Pruned Weight Plot", fontsize=fig_font_size)

    plt.savefig(output)
    return output

def calculate_layer_sparsity(layer):
    """
    Calculates the fraction of values in a tensor that are 0.
    args:
      - layer: the desired layer to be evaluated
    returns: fraction of zero values in ndarray
    """

    return 1 - np.count_nonzero(layer) / np.product(layer.shape)


def calculate_model_sparsity(model):
    """
    Calculates the fraction of values in a model that are 0.
    args:
      - model: the desired model to be evaluated
    returns: fraction of zero values in model
    """
    return np.average([calculate_layer_sparsity(layer) for layer in model],
                      weights=[np.product(layer.shape) for layer in model])


def visualize_kernel(kernel, fig):
    """
    Displays kernel weight as a plot.
    returns: plot of weights
    """
    im = fig.imshow(np.abs(kernel),
               interpolation='none',
               cmap='inferno')
    # plt.colorbar()
    # fig.clim(0)
    return fig
