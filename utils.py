from tensorflow.python.client import device_lib
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np

def get_tensorflow_devices():
    "list available devices from TensorFlow"
    local_device_protos = device_lib.list_local_devices()
    return [(x.name, x.device_type) for x in local_device_protos]


def load_image(path, width=224, height=224, expand_dim=False):
    img = imread(path)
    img = imresize(img, (width, height))
    if expand_dim :
        img = img.reshape((1, width, height, img.shape[-1]))

    return img

def save_image(path, image):
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    imsave(path, image)


def get_activations(sess, model, layer, fmap=-1, nb_col=6):
    activations = sess.run(model[layer])
    # shape : [1, w, h, nb_fmap]
    nb_fmap = activations.shape[3]
    w = activations.shape[1]
    h = activations.shape[2]

    nb_row = int(math.ceil(nb_fmap // nb_col)) + 1

    if fmap < 0:

        fig = plt.figure(figsize=(15, int(nb_row * 2.8)))
        outer_grid = gridspec.GridSpec(nb_row, nb_col, wspace=0.1, hspace=0.1)

        for i in range(nb_fmap):
            ax = plt.Subplot(fig, outer_grid[i])
            ax.imshow(activations[0, :, :, i], cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

        all_axes = fig.get_axes()
        # show only the outside spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ax.is_first_row():
                ax.spines['top'].set_visible(True)
            if ax.is_last_row():
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)
            if ax.is_last_col():
                ax.spines['right'].set_visible(True)

        plt.show()

    elif fmap < nb_fmap:
        plt.imshow(activations[0, :, :, fmap], cmap='gray')
    else:
        print("")