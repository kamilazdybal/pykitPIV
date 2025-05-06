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

    .. note::

        Running multiple postprocessing functions on a single object of the ``Postprocess`` class will create
        a superposition of effects that those functions have. This way, the user can chain multiple transformations, such as:

        adding shot noise :math:`\\rightarrow` adding Gaussian noise :math:`\\rightarrow` log-transformation.

        The final outcome will always be stored in the class attribute ``processed_image_tensor``.

    **Example:**

    .. code:: python

        from pykitPIV import Image, Postprocess

        # Upload saved images:
        image = Image()
        images_tensor_dic = image.upload_from_h5(filename='pykitPIV-dataset.h5')
        images_tensor = images_tensor_dic['I']

        # Initialize a postprocessing object:
        postprocess = Postprocess(image_tensor, random_seed=100)

        # Check if paired or single image frames have been uploaded:
        postprocess.image_pair

    :param image_tensor:
        ``numpy.ndarray`` specifying image or images to postprocess. It can be an array of size :math:`(N, H, W)`
        or :math:`(N, 2, H, W)`.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``.
        If specified, all operations are reproducible.

    **Attributes:**

    - **random_seed** - (read-only) as per user input.
    - **image_tensor** - (read-only) as per user input.
    - **image_pair** - (read-only) ``bool`` specifying whether paired or single images have been uploaded.
    - **processed_image_tensor** - (read-only) ``numpy.ndarray`` storing the postprocessed image tensor.
      Note that when multiple postprocessing operations are called on the same object of class ``Postprocess``,
      this quantity will store a superposition of them, corresponding to the sequence of execution.
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
        self.__processed_image_tensor = image_tensor

        # Initialize parameters of the Gaussian noise:
        self.__scale_per_image = None

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
    def image_pair(self):
        return self.__image_pair

    @property
    def processed_image_tensor(self):
        return self.__processed_image_tensor

    @property
    def scale_per_image(self):
        return self.__scale_per_image

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_gaussian_noise(self,
                           loc=0.0,
                           scale=(500, 1000),
                           clip=None):
        """
        Adds Gaussian noise to the image tensor or any image-like array of size :math:`(N, H, W)`
        or :math:`(N, 2, H, W)`.

        **Example:**

        .. code:: python

            from pykitPIV import Image, Postprocess

            # Upload saved images:
            image = Image()
            images_tensor_dic = image.upload_from_h5(filename='pykitPIV-dataset.h5')
            images_tensor = images_tensor_dic['I']

            # Initialize a postprocessing object:
            postprocess = Postprocess(image_tensor, random_seed=100)

            # Add noise to the uploaded images:
            postprocess.add_gaussian_noise(loc=0.0,
                                           scale=(500,1000),
                                           clip=2**16-1)

        :param loc: (optional)
            ``int`` or ``float`` specifying the center of the Gaussian distribution.
        :param scale: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
            standard deviation for the noise on an image to randomly sample from.
            It can also be set to ``int`` or ``float`` to generate a fixed standard deviation value across all :math:`N` image pairs.
            The unit of the standard deviation is image intensity.
        :param clip: (optional)
            ``int`` or ``float`` specifying whether maximum values on images should be clipped to a specified value.
            If set to ``None``, values are not clipped.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(loc, int) and not isinstance(loc, float):
            raise ValueError("Parameter `loc` has to be of type `int` or `float`.")

        if isinstance(scale, tuple):
            check_two_element_tuple(scale, 'scale')
            check_min_max_tuple(scale, 'scale')
        elif isinstance(scale, int) or isinstance(scale, float):
            check_non_negative_int_or_float(scale, 'scale')
        else:
            raise ValueError("Parameter `scale` has to be of type 'tuple' or 'int' or 'float'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if isinstance(scale, tuple):
            __scales = scale
        else:
            __scales = (scale, scale)

        n_images = self.image_tensor.shape[0]

        self.__scale_per_image = np.random.rand(n_images) * (__scales[1] - __scales[0]) + __scales[0]

        image_tensor_with_noise = np.zeros_like(self.image_tensor)

        for i in range(0, n_images):

            if self.__image_pair:
                image_tensor_with_noise[i,:,:,:] = self.processed_image_tensor[i,:,:,:] + np.random.normal(loc=loc,
                                                                                                           scale=self.__scale_per_image[i],
                                                                                                           size=np.shape(self.image_tensor[i,:,:,:]))
            else:
                image_tensor_with_noise[i,:,:] = self.processed_image_tensor[i,:,:] + np.random.normal(loc=loc,
                                                                                                       scale=self.__scale_per_image[i],
                                                                                                       size=np.shape(self.image_tensor[i,:,:]))

        if clip is not None:
            image_tensor_with_noise = np.clip(image_tensor_with_noise, 0, clip)

        self.__processed_image_tensor = image_tensor_with_noise

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_shot_noise(self,
                       strength=1,
                       clip=None):
        """
        Adds shot noise to the image tensor or any image-like array of size :math:`(N, H, W)`
        or :math:`(N, 2, H, W)`. The shot noise models a Poisson distribution of photons reaching the camera lens.

        If :math:`\\lambda` is interpreted as the expected image intensity at a particular pixel,
        then the shot noise corrects that intensity by drawing the new intensity, :math:`k`, from the Poisson distribution:

        .. math::

            f(k, \\lambda) = \\frac{\\lambda^k e^{- \\lambda}}{k!}

        where :math:`f` denotes the probability of observing intensity :math:`k`,
        given the expected intensity :math:`\\lambda`.

        **Example:**

        .. code:: python

            from pykitPIV import Image, Postprocess

            # Upload saved images:
            image = Image()
            images_tensor_dic = image.upload_from_h5(filename='pykitPIV-dataset.h5')
            images_tensor = images_tensor_dic['I']

            # Initialize a postprocessing object:
            postprocess = Postprocess(image_tensor, random_seed=100)

            # Add shot noise to the uploaded images:
            postprocess.add_shot_noise(strength=1,
                                       clip=2**16-1)

        :param strength: (optional)
            ``int`` or ``float`` specifying the strength of the shot noise.
        :param clip: (optional)
            ``int`` or ``float`` specifying whether maximum values on images should be clipped to a specified value.
            If set to ``None``, values are not clipped.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        check_non_negative_int_or_float(strength, 'strength')

        if clip is not None:
            check_non_negative_int_or_float(clip, 'clip')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        n_images = self.image_tensor.shape[0]
        height = self.image_tensor.shape[-2]
        width = self.image_tensor.shape[-1]

        image_tensor_with_noise = np.zeros_like(self.image_tensor)

        if self.__image_pair:

            for i in range(0, n_images):
                for h in range(0, height):
                    for w in range(0, width):

                        image_tensor_with_noise[i,0,h,w] = strength * np.random.poisson(lam=np.abs(self.processed_image_tensor[i,0,h,w]))

        else:
            for i in range(0, n_images):
                for h in range(0, height):
                    for w in range(0, width):
                        image_tensor_with_noise[i, h, w] = strength * np.random.poisson(lam=np.abs(self.processed_image_tensor[i,h,w]))

        if clip is not None:
            image_tensor_with_noise = np.clip(image_tensor_with_noise, 0, clip)

        self.__processed_image_tensor = image_tensor_with_noise

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def log_transform_images(self,
                             addition=1):
        """
        Log-transforms image tensor or any image-like array of size  :math:`(N, H, W)`
        or :math:`(N, 2, H, W)`.

        .. math::

            \mathbf{T}_{\\text{log}} = \log_{10} (\mathbf{T} + a)


        **Example:**

        .. code:: python

            from pykitPIV import Image, Postprocess

            # Upload saved images:
            image = Image()
            images_tensor_dic = image.upload_from_h5(filename='pykitPIV-dataset.h5')
            images_tensor = images_tensor_dic['I']

            # Initialize a postprocessing object:
            postprocess = Postprocess(image_tensor, random_seed=100)

            # Create a log-transformation of image intensities:
            postprocess.log_transform_images(addition=10000)

        :param addition: (optional)
            ``int`` or ``float`` specifying the added constant, :math:`a`,
            whose purpose is to eliminate zero elements in the images tensor.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(addition, int) and not isinstance(addition, float):
            raise ValueError("Parameter `addition` has to be of type `int` or `float`.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        log_transformed_image_tensor = np.zeros_like(self.image_tensor)

        n_images = self.image_tensor.shape[0]

        for i in range(0, n_images):

            if self.__image_pair:
                log_transformed_image_tensor[i, 0, :, :] = np.log10(self.processed_image_tensor[i, 0, :, :] + addition)
                log_transformed_image_tensor[i, 1, :, :] = np.log10(self.processed_image_tensor[i, 1, :, :] + addition)
            else:
                log_transformed_image_tensor[i, :, :] = np.log10(self.processed_image_tensor[i, :, :] + addition)
                log_transformed_image_tensor[i, :, :] = np.log10(self.processed_image_tensor[i, :, :] + addition)

        self.__processed_image_tensor = log_transformed_image_tensor

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def filter_images(self,
                      image_tensor,
                      filter='gaussian',
                      padding='zeros'):
        """
        Filters image tensor by applying a convolution with specified kernel properties.

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
             xticks=True,
             yticks=True,
             title=None,
             vmin=None,
             vmax=None,
             cmap='Greys_r',
             cbar=False,
             figsize=(5, 5),
             dpi=300,
             filename=None):
        """
        Plots a single, post-processed static image. For PIV images,
        the user can select between :math:`I_1` or :math:`I_2`.

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
        :param xticks: (optional)
            ``bool`` specifying if ticks along the :math:`x`-axis should be plotted.
        :param yticks: (optional)
            ``bool`` specifying if ticks along the :math:`y`-axis should be plotted.
        :param title: (optional)
            ``str`` specifying figure title.
        :param cmap: (optional)
            ``str`` or an object of `matplotlib.colors.ListedColormap <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html>`_ specifying the color map to use.
        :param cbar: (optional)
            ``bool`` specifying whether colorbar should be plotted.
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

        if not isinstance(xticks, bool):
            raise ValueError("Parameter `xticks` has to be of type 'bool'.")

        if not isinstance(yticks, bool):
            raise ValueError("Parameter `yticks` has to be of type 'bool'.")

        if (title is not None) and (not isinstance(title, str)):
            raise ValueError("Parameter `title` has to be of type 'str'.")

        if not isinstance(cbar, bool):
            raise ValueError("Parameter `cbar` has to be of type 'bool'.")

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

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if not xticks:
            plt.xticks([])

        if not yticks:
            plt.yticks([])

        if title is not None:
            plt.title(title)

        if cbar:
            plt.colorbar()

        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')

        return plt

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
