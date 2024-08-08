import h5py
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

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

    # ##################################################################################################################

    # Plotting functions

    # ##################################################################################################################

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot(self,
             original,
             idx,
             instance=1,
             xlabel=None,
             ylabel=None,
             title=None,
             vmin=None,
             vmax=None,
             cmap='Greys_r',
             cbar=False,
             figsize=(5,5),
             dpi=300,
             filename=None):
        """
        Plots a single, post-processed static image. For PIV images, the user can select between :math:`I_1` or :math:`I_2`.

        :param original:
            ``bool`` specifying whether the original or the post-processed image tensor should be plotted.
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

        if not isinstance(original, bool):
            raise ValueError("Parameter `original` has to be of type 'bool'.")

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

        if (vmin is not None) and (not isinstance(vmin, int)) and (not isinstance(vmin, float)):
            raise ValueError("Parameter `vmin` has to be of type 'int' or 'float'.")

        if (vmax is not None) and (not isinstance(vmax, int)) and (not isinstance(vmax, float)):
            raise ValueError("Parameter `vmax` has to be of type 'int' or 'float'.")

        if not isinstance(cbar, bool):
            raise ValueError("Parameter `cbar` has to be of type 'bool'.")

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

            if original:

                if self.__image_pair:
                    if vmin is None: vmin = np.min(self.image_tensor[idx, instance-1, :, :])
                    if vmax is None: vmax = np.max(self.image_tensor[idx, instance-1, :, :])
                    plt.imshow(self.image_tensor[idx, instance-1, :, :], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
                else:
                    if vmin is None: vmin = np.min(self.image_tensor[idx, :, :])
                    if vmax is None: vmax = np.max(self.image_tensor[idx, :, :])
                    plt.imshow(self.image_tensor[idx, :, :], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

            else:

                if self.__image_pair:
                    if vmin is None: vmin = np.min(self.processed_image_tensor[idx, instance-1, :, :])
                    if vmax is None: vmax = np.max(self.processed_image_tensor[idx, instance-1, :, :])
                    plt.imshow(self.processed_image_tensor[idx, instance-1, :, :], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
                else:
                    if vmin is None: vmin = np.min(self.processed_image_tensor[idx, :, :])
                    if vmax is None: vmax = np.max(self.processed_image_tensor[idx, :, :])
                    plt.imshow(self.processed_image_tensor[idx, :, :], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

        if cbar:
            plt.colorbar()

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
