import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.python.client import device_lib
from scipy.misc import imread, imresize, imsave
import numpy as np

import threading
import math
import os
import random


def get_tensorflow_devices():
    "list available devices from TensorFlow"
    local_device_protos = device_lib.list_local_devices()
    return [(x.name, x.device_type) for x in local_device_protos]


def save_image(path, img):
    """
    Save a float32 image with scipy.misc.imsave
    - perform np.clip
    :param path: filename
    :param img: numpy array (dtype=np.float32)
    :return: nothing
    """
    img_shape = img.shape
    if len(img_shape) == 4 :
        img = img[0]

    img = np.clip(img, 0, 255).astype('uint8')
    imsave(path, img)


def get_activations(sess, model, layer, fmap=-1, nb_col=6):
    """

    :param sess: tensorflow session, to use sess.run()
    :param model: a VGG network, represented as a python dict
    :param layer: layer name (ex : conv4_2)
    :param fmap: if fmap=-1 show all feature maps, otherwise show only one fmap
    :param nb_col: of fmap=-1, to tune display config.
    :return: nothing
    """
    activations = sess.run(model[layer])

    # shape : [1, w, h, nb_fmap]
    nb_fmap = activations.shape[3]

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
        raise ValueError("Only {} feature maps in \"{}\" with this model.".format(nb_fmap, layer))


def load_img(path,reshape_mode='resize', grayscale=False, target_size=(224, 224), expand_dim=False):
    """

    :param path: image file
    :param reshape_mode: "resize" to perform directly resize operation. if "crop", to perform crop before resize operation.
    :param grayscale: if True, load into grayscale, with shape [W,H,1]
    :param target_size: (w,h)
    :param expand_dim: add a 'first' dim [1, W, H, C]
    :return: numpy array
    """

    if grayscale:
        img = imread(path, mode='L')
    else:
        img = imread(path, mode='RGB')

    if reshape_mode == "crop":
        shape = img.shape[:2]
        if shape[0] == shape[1]:
            pass
        else:
            short_axis = np.argmin(shape)
            short_edge = shape[short_axis]

            off_set = abs(shape[0] - shape[1])
            off_set = random.randint(0, off_set)

            if short_axis == 0:
                img = img[:, off_set:off_set + short_edge]
            else:
                img = img[off_set:off_set + short_edge, :]

    if target_size:
        img = imresize(img, size=target_size)

    if grayscale:
        img = np.expand_dims(img, axis=-1)

    if expand_dim :
        img = np.expand_dims(img, axis=0)

    return img


# inspired from  https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
# without 'followlinks' argument
def _count_valid_files_in_directory(directory, white_list_formats):
    """Count files with extension in `white_list_formats` contained in a directory.
    # Arguments
        directory: absolute path to the directory containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

    samples = 0
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                samples += 1
    return samples


# inspired from  https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
# without 'followlinks' argument
def _list_valid_filenames_in_directory(directory, white_list_formats):
    """List paths of files in `subdir` relative from `directory` whose extensions are in `white_list_formats`.
    # Arguments
        directory: absolute path to a directory containing the files to list.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        class_indices: dictionary mapping a class name to its index.
    # Returns
        filenames: the path of valid files in `directory`
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

    filenames = []
    basedir = os.path.dirname(directory)
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                # add filename relative to directory
                absolute_path = os.path.join(root, fname)
                filenames.append(os.path.relpath(absolute_path, basedir))
    return filenames


# inspired from  https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
# only with resize (and optionally cropping) !
class BatchGenerator(object):

    def __init__(self, directory,
                 target_size=(256,256),
                 reshape_mode='resize',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 color_mode='rgb'):
        """

        :param directory:
        :param target_size:
        :param reshape_mode:
        :param batch_size:
        :param shuffle:
        :param seed:
        :param color_mode:
        """

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        self.directory = directory
        self.target_size = tuple(target_size)

        if reshape_mode not in {'resize', 'crop'}:
            raise ValueError('Invalid reshape mode:', reshape_mode, '; expected "resize" or "crop".')
        self.reshape_mode = reshape_mode

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode, '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode

        if color_mode == 'rgb':
            self.image_shape = self.target_size + (3,)
        else: # grayscale
            self.image_shape = self.target_size + (1,)

        self.samples = _count_valid_files_in_directory(directory, white_list_formats)
        print("Found {} images".format(self.samples))

        self.filenames = _list_valid_filenames_in_directory(directory, white_list_formats)

        self._batch_index = 0
        self._total_batches_seen = 0
        self._lock = threading.Lock()
        self._index_generator = self._flow_index()

    def _flow_index(self):

        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self._total_batches_seen)
            if self._batch_index == 0:
                index_array = np.arange(self.samples)
                if self.shuffle:
                    index_array = np.random.permutation(self.samples)

            current_index = (self.batch_index * self.batch_size) % self.samples
            if self.samples > current_index + self.batch_size:
                current_batch_size = self.batch_size
                self._batch_index += 1
            else:
                current_batch_size = self.samples - current_index
                self._batch_index = 0
            self._total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def reset(self):
        self.batch_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self._lock:
            index_array, current_index, current_batch_size = next(self._index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=np.float32)
        grayscale = self.color_mode == 'grayscale'

        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            batch_x[i] = load_img(os.path.join(self.directory, fname),
                                  reshape_mode=self.reshape_mode,
                                  grayscale=grayscale,
                                  target_size=self.target_size)

        return batch_x

