import h5py
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from pykitPIV.checks import *
from pykitPIV.image import Image

########################################################################################################################
########################################################################################################################
####
####    Class: Postprocess
####
########################################################################################################################
########################################################################################################################

class Postprocess:
    """
    Postprocesses images.

    **Example:**

    .. code:: python

        from pykitPIV import Postprocess

        # Generate images:




        # Initialize a postprocessing object:
        postprocess = Postprocess(image_tensor, random_seed=100)


    :param image_tensor:
        ``numpy.ndarray`` specifying image or images to postprocess. It can be an array of size ``(n_images, image_height, image_width)``
            or ``(n_images, 2, image_height, image_width)``.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``. If specified, all image generation is reproducible.

    **Attributes:**

    - **random_seed** - (read-only) as per user input.
    - **image_tensor** - (read-only) ``numpy.ndarray`` storing the image tensor to postprocess.
    - **processed_image_tensor** - (read-only) ``numpy.ndarray`` storing the postprocessed image tensor.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __init__(self,
                 image_tensor,
                 random_seed=None):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:
        
        if not isinstance(image_tensor, np.ndarray):
            raise ValueError("Parameter `image_tensor` has to be of type `numpy.ndarray`.")

        # Check the images tensor size:
        try:
            (n_images, n_pairs, _, _) = np.shape(image_tensor)
            if n_pairs == 2:
                self.__image_pair = True
            else:
                raise ValueError(
                    "Parameter `image_tensor` has to be of size `(n_images, 2, image_height, image_width)` or `(n_images, image_height, image_width)`.")
        except:
            try:
                (n_images, _, _) = np.shape(image_tensor)
                self.__image_pair = False
            except:
                raise ValueError(
                    "Parameter `image_tensor` has to be of size `(n_images, 2, image_height, image_width)` or `(n_images, image_height, image_width)`.")

        if random_seed is not None:
            if type(random_seed) != int:
                raise ValueError("Parameter `random_seed` has to be of type 'int'.")
            else:
                np.random.seed(seed=random_seed)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:
        self.__random_seed = random_seed
        self.__image_tensor = image_tensor

        # Initialize processed image tensor:
        self.__processed_image_tensor = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Properties coming from user inputs:
    @property
    def random_seed(self):
        return self.__random_seed

    @property
    def image_tensor(self):
        return self.__image_tensor

    # Properties computed at class init:
    @property
    def processed_image_tensor(self):
        return self.__processed_image_tensor

    @property
    def image_pair(self):
        return self.__image_pair

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def log_transform_images(self,
                             addition=1):
        """
        Log-transforms PIV image pairs tensor or any image-like array of size ``(n_images, image_height, image_width)``
        or ``(n_images, 2, image_height, image_width)``.

        .. math::

            \mathbf{T}_{\\text{log}} = \log_{10} (\mathbf{T} + a)

        :param addition: (optional)
            ``int`` specifying the added constant, :math:`a`, whose purpose is to eliminate zero elements in the images tensor.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(addition, int) and not isinstance(addition, float):
            raise ValueError("Parameter `addition` has to be of type `int` or `float`.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        log_transformed_image_tensor = np.zeros_like(self.image_tensor)

        for i in range(0, self.image_tensor.shape[0]):

            if self.__image_pair:
                log_transformed_image_tensor[i, 0, :, :] = np.log10(self.image_tensor[i, 0, :, :] + addition)
                log_transformed_image_tensor[i, 1, :, :] = np.log10(self.image_tensor[i, 1, :, :] + addition)
            else:
                log_transformed_image_tensor[i, :, :] = np.log10(self.image_tensor[i, :, :] + addition)
                log_transformed_image_tensor[i, :, :] = np.log10(self.image_tensor[i, :, :] + addition)

        self.__processed_image_tensor = log_transformed_image_tensor

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def filter_images(self,
                      image_tensor,
                      filter='gaussian',
                      padding='zeros'):
        """
        Filters PIV image pairs tensor by applying a convolution with specified kernel properties.

        .. math::

            \\widetilde{\mathbf{T}} = \mathbf{T} : \mathbf{k}

        where :math:`:` denotes tensor contraction.

        Image filtering is done with ``torch.nn.functional.conv2d``.

        :param filter:
            ``str`` specifying the kernel type.
        :param padding:
            ``str`` specifying the type of padding.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:



        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        filtered_image_tensor = np.zeros_like(image_tensor)






        self.__processed_image_tensor = filtered_image_tensor

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def save_to_h5(self,
                   tensors_dictionary,
                   save_individually=False,
                   filename=None,
                   verbose=False):
        """
        Saves the image pairs tensor and the associated flow targets tensor to ``.h5`` data format.

        :param tensors_dictionary:
            ``dict`` specifying the tensors to save.
        :param save_individually: (optional)
            ``bool`` specifying if each image pair and the associated targets should be saved to a separate file.
            It is recommended to save individually for large datasets that will be uploaded by **PyTorch**, since at
            any iteration of a machine learning algorithm, only a small batch of samples is uploaded to memory.
        :param filename: (optional)
            ``str`` specifying the path and filename to save the ``.h5`` data. Note that ``'-pair-#'`` will be added
            automatically to your filename for each saved image pair.
            If set to ``None``, a default name ``'PIV-dataset-pair-#.h5'`` will be used.
        :param verbose: (optional)
            ``bool`` for printing verbose details.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(tensors_dictionary, dict):
            raise ValueError("Parameter `tensors_dict` has to be of type 'dict'.")

        if not isinstance(save_individually, bool):
            raise ValueError("Parameter `save_individually` has to be of type 'bool'.")

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        if filename is None:
            filename = 'PIV-dataset.h5'

        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be of type 'bool'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if save_individually:

            dictionary_keys = list(tensors_dictionary.keys())

            n_images, _, _ ,_ = tensors_dictionary[dictionary_keys[0]].shape

            for i in range(0,n_images):

                individual_filename = filename.split('.')[0] + '-sample-' + str(i) + '.h5'

                with h5py.File(individual_filename, 'w', libver='latest') as f:
                    for name_tag, data_item in tensors_dictionary.items():
                        dataset = f.create_dataset(name_tag, data=data_item[i], compression='gzip', compression_opts=9)
                    f.close()

                if verbose: print(individual_filename + '\tsaved.')

        else:

            with h5py.File(filename, 'w', libver='latest') as f:
                for name_tag, data_item in tensors_dictionary.items():
                    dataset = f.create_dataset(name_tag, data=data_item, compression='gzip', compression_opts=9)
                f.close()

            if verbose: print('Dataset saved.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def upload_from_h5(self,
                       filename=None):
        """
        Upload image pairs tensor and the associated flow targets tensor from ``.h5`` data format.

        :param filename: (optional)
            ``str`` specifying the path and filename to save the ``.h5`` data. Note that ``'-pair-#'`` will be added
            automatically to your filename for each saved image pair.
            If set to ``None``, a default name ``'PIV-dataset-pair-#.h5'`` will be used.

        :return:
            - **tensors_dictionary** - ``dict`` specifying the dataset tensors.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        if filename is None:
            filename = 'PIV-dataset.h5'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        tensors_dictionary = {}

        f = h5py.File(filename, 'r')

        for key in f.keys():
            tensors_dictionary[key] = np.array(f[key])

        f.close()

        return tensors_dictionary

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # ##################################################################################################################

    # Plotting functions

    # ##################################################################################################################

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot(self,
             idx,
             instance=1,
             xlabel=None,
             ylabel=None,
             title=None,
             cmap='Greys_r',
             figsize=(5,5),
             dpi=300,
             filename=None):
        """
        Plots a single, post-processed static image. For PIV images, the user can select between :math:`I_1` or :math:`I_2`.

        :param idx:
            ``int`` specifying the index of the image to plot out of ``n_images`` number of images.
        :param instance: (optional)
            ``int`` specifying whether :math:`I_1` (``instance=1``) or :math:`I_2` (``instance=2``) should be plotted.
        :param xlabel: (optional)
            ``str`` specifying :math:`x`-label.
        :param ylabel: (optional)
            ``str`` specifying :math:`y`-label.
        :param title: (optional)
            ``str`` specifying figure title.
        :param cmap: (optional)
            ``str`` or an object of `matplotlib.colors.ListedColormap <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html>`_ specifying the color map to use.
        :param figsize: (optional)
            ``tuple`` of two numerical elements specifying the figure size as per ``matplotlib.pyplot``.
        :param dpi: (optional)
            ``int`` specifying the dpi for the image.
        :param filename: (optional)
            ``str`` specifying the path and filename to save an image. If set to ``None``, the image will not be saved.

        :return:
            - **plt** - ``matplotlib.pyplot`` image handle.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(idx, int):
            raise ValueError("Parameter `idx` has to be of type 'int'.")
        if idx < 0:
            raise ValueError("Parameter `idx` has to be non-negative.")

        if not isinstance(instance, int):
            raise ValueError("Parameter `instance` has to be of type 'int'.")
        if instance not in [1,2]:
            raise ValueError("Parameter `instance` has to be 1 (for image I1) or 2 (for image I2).")

        if (xlabel is not None) and (not isinstance(xlabel, str)):
            raise ValueError("Parameter `xlabel` has to be of type 'str'.")

        if (ylabel is not None) and (not isinstance(ylabel, str)):
            raise ValueError("Parameter `ylabel` has to be of type 'str'.")

        if (title is not None) and (not isinstance(title, str)):
            raise ValueError("Parameter `title` has to be of type 'str'.")

        check_two_element_tuple(figsize, 'figsize')

        if not isinstance(dpi, int):
            raise ValueError("Parameter `dpi` has to be of type 'int'.")

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if self.processed_image_tensor is None:

            print('Note: Image tensor has not been post-processed yet!\n\n')

        else:

            fig = plt.figure(figsize=figsize)

            if self.__image_pair:

                plt.imshow(self.processed_image_tensor[idx, instance-1, :, :], cmap=cmap, origin='lower')

            else:

                plt.imshow(self.processed_image_tensor[idx, :, :], cmap=cmap, origin='lower')

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)

        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')

        return plt

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_image_pair(self,
                        idx,
                        with_buffer=False,
                        xlabel=None,
                        ylabel=None,
                        title=None,
                        cmap='Greys_r',
                        figsize=(5,5),
                        dpi=300,
                        filename=None):
        """
        Plots an animated PIV image pair, :math:`\mathbf{I} = (I_1, I_2)^{\\top}`, at time :math:`t`
        and :math:`t + \\Delta t` respectively.

        :param idx:
            ``int`` specifying the index of the image to plot out of ``n_images`` number of images.
        :param with_buffer:
            ``bool`` specifying whether the buffer for the image size should be visualized. If set to ``False``, the true PIV image size is visualized. If set to ``True``, the PIV image with a buffer is visualized and buffer outline is marked with a red rectangle.
        :param xlabel: (optional)
            ``str`` specifying :math:`x`-label.
        :param ylabel: (optional)
            ``str`` specifying :math:`y`-label.
        :param title: (optional)
            ``str`` specifying figure title.
        :param cmap: (optional)
            ``str`` or an object of `matplotlib.colors.ListedColormap <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html>`_ specifying the color map to use.
        :param figsize: (optional)
            ``tuple`` of two numerical elements specifying the figure size as per ``matplotlib.pyplot``.
        :param dpi: (optional)
            ``int`` specifying the dpi for the image.
        :param filename: (optional)
            ``str`` specifying the path and filename to save an image. If set to ``None``, the image will not be saved.

        :return:
            - **plt** - ``matplotlib.pyplot`` image handle.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(idx, int):
            raise ValueError("Parameter `idx` has to be of type 'int'.")
        if idx < 0:
            raise ValueError("Parameter `idx` has to be non-negative.")

        if not isinstance(with_buffer, bool):
            raise ValueError("Parameter `with_buffer` has to be of type 'bool'.")

        if (xlabel is not None) and (not isinstance(xlabel, str)):
            raise ValueError("Parameter `xlabel` has to be of type 'str'.")

        if (ylabel is not None) and (not isinstance(ylabel, str)):
            raise ValueError("Parameter `ylabel` has to be of type 'str'.")

        if (title is not None) and (not isinstance(title, str)):
            raise ValueError("Parameter `title` has to be of type 'str'.")

        check_two_element_tuple(figsize, 'figsize')

        if not isinstance(dpi, int):
            raise ValueError("Parameter `dpi` has to be of type 'int'.")

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        imagelist = [self.images_I1[idx], self.images_I2[idx]]

        if self.__motion is None:

            print('Note: Movement of particles has not been added to the image yet!\n\n')

        else:

            fig = plt.figure(figsize=figsize)

            # Check if particles were generated with a buffer:
            if self.__particles.size_buffer == 0:

                im = plt.imshow(imagelist[0], cmap=cmap, origin='lower', animated=True)

            else:

                if with_buffer:

                    im = plt.imshow(imagelist[0], cmap=cmap, origin='lower', animated=True)

                    # Extend the imshow area with the buffer:
                    f = lambda pixel: pixel - self.__particles.size_buffer
                    im.set_extent([f(x) for x in im.get_extent()])

                    # Visualize a rectangle that separates the proper PIV image area and the artificial buffer outline:
                    rect = patches.Rectangle((-0.5, -0.5), self.__particles.size[1], self.__particles.size[0], linewidth=1, edgecolor='r', facecolor='none')
                    ax = plt.gca()
                    ax.add_patch(rect)

                else:

                    im = plt.imshow(imagelist[0][self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer], cmap=cmap, origin='lower', animated=True)

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)

        def updatefig(j):

            if self.__particles.size_buffer == 0:
                im.set_array(imagelist[j])
            else:
                if with_buffer:
                    im.set_array(imagelist[j])
                else:
                    im.set_array(imagelist[j][self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer])

            return [im]

        anim = animation.FuncAnimation(fig, updatefig, frames=range(2), interval=50, blit=True)

        anim.save(filename, fps=2, bitrate=-1, dpi=dpi, savefig_kwargs={'bbox_inches' : 'tight'})

        return anim

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
