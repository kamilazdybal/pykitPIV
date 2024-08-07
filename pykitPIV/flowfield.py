import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import scipy
import warnings
from pykitPIV.checks import *

################################################################################
################################################################################
####
####    Class: FlowField
####
################################################################################
################################################################################

class FlowField:
    """
    Generates or uploads velocity field(s) to advect particles between two consecutive PIV images.

    Two-dimensional velocity field is supported for the moment: :math:`\\vec{V} = [u, v]`, where
    :math:`u` and :math:`v` velocity components are represented as two-dimensional arrays
    of shape (height [px] :math:`\\times` width [px]).

    Velocity magnitude is computed as :math:`|\\vec{V}| = \\sqrt{u^2 + v^2}` and is also represented as a two-dimensional array
    of shape (height [px] :math:`\\times` width [px]).

    **Example:**

    .. code:: python

        from pykitPIV import FlowField

        # We are going to generate 10 flow fields for 10 PIV image pairs:
        n_images = 10

        # Specify size in pixels for each image:
        image_size = (128,512)

        # Initialize a flow field object:
        flowfield = FlowField(n_images=n_images,
                              size=image_size,
                              size_buffer=10,
                              random_seed=100)

    :param n_images:
        ``int`` specifying the number of image pairs to create.
    :param size: (optional)
        ``tuple`` of two ``int`` elements specifying the size of each image in pixels. The first number is image height, :math:`h`, the second number is image width, :math:`w`.
    :param size_buffer: (optional)
        ``int`` specifying the buffer in pixels :math:`[\\text{px}]` to add to the image size in the width and height direction.
        This number should be approximately equal to the maximum displacement that particles are subject to in order to allow for new particles to arrive into the image area.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``. If specified, all image generation is reproducible.

    **Attributes:**

    - **n_images** - (read-only) as per user input.
    - **size** - (read-only) as per user input.
    - **size_buffer** - (read-only) as per user input.
    - **random_seed** - (read-only) as per user input.
    - **displacement** - (read-only) as per user input.
    - **displacement_per_image** - (read-only) ``numpy.ndarray`` specifying the template for the displacements per each image. Template diameters are random numbers between ``displacement[0]`` and ``displacement[1]``.
    - **gaussian_filters** - (read-only) as per user input.
    - **n_gaussian_filter_iter** - (read-only) as per user input.
    - **gaussian_filter_per_image** - (read-only) ``numpy.ndarray`` specifying the template for the Gaussian filter sizes in pixels :math:`[\\text{px}]` for each image. Template diameters are random numbers between ``gaussian_filters[0]`` and ``gaussian_filters[1]``.
    - **velocity_field** - (read-only) ``list`` of two-element ``tuple`` specifying the velocity field components per each image, :math:`u` and :math:`v`. The first element of each tuple is :math:`u` and the second element of each tuple is :math:`v`. Both elements have shapes :math:`(\\text{height} [\\text{px}] \\times \\text{width} [\\text{px}])]`.
    - **velocity_field_magnitude** - (read-only) ``list`` of ``numpy.ndarray`` specifying the velocity field magnitude per each image. Each array has shape :math:`(\\text{height} [\\text{px}] \\times \\text{width} [\\text{px}])]`.
    - **size_with_buffer** - (read-only) ``tuple`` specifying the size of each image in pixels with buffer added.
    """

    def __init__(self,
                 n_images,
                 size=(512, 512),
                 size_buffer=10,
                 random_seed=None):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if type(n_images) != int:
            raise ValueError("Parameter `n_images` has to be of type 'int'.")

        if n_images <= 0:
            raise ValueError("Parameter `n_images` has to be positive. At least one image has to be generated.")

        check_two_element_tuple(size, 'size')

        if not isinstance(size_buffer, int):
            raise ValueError("Parameter `size_buffer` has to be of type 'int'.")

        if size_buffer < 0:
            raise ValueError("Parameter `size_buffer` has to non-negative.")

        if random_seed is not None:
            if type(random_seed) != int:
                raise ValueError("Parameter `random_seed` has to be of type 'int'.")
            else:
                np.random.seed(seed=random_seed)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:
        self.__n_images = n_images
        self.__size = size
        self.__size_buffer = size_buffer
        self.__random_seed = random_seed
        self.__displacement = None
        self.__displacement_per_image = None
        self.__gaussian_filters = None
        self.__n_gaussian_filter_iter = None
        self.__gaussian_filter_per_image = None
        self.__velocity_field = None
        self.__velocity_field_magnitude = None

        # Compute the image outline that serves as a buffer:
        self.__height_with_buffer = self.size[0] + 2 * self.size_buffer
        self.__width_with_buffer = self.size[1] + 2 * self.size_buffer
        self.__size_with_buffer = (self.__height_with_buffer, self.__width_with_buffer)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Properties coming from user inputs:
    @property
    def n_images(self):
        return self.__n_images

    @property
    def size(self):
        return self.__size

    @property
    def size_buffer(self):
        return self.__size_buffer

    @property
    def random_seed(self):
        return self.__random_seed

    @property
    def displacement(self):
        return self.__displacement

    @property
    def displacement_per_image(self):
        return self.__displacement_per_image

    @property
    def gaussian_filters(self):
        return self.__gaussian_filters

    @property
    def n_gaussian_filter_iter(self):
        return self.__n_gaussian_filter_iter

    @property
    def gaussian_filter_per_image(self):
        return self.__gaussian_filter_per_image

    @property
    def velocity_field(self):
        return self.__velocity_field

    @property
    def velocity_field_magnitude(self):
        return self.__velocity_field_magnitude

    # Properties computed at class init:
    @property
    def size_with_buffer(self):
        return self.__size_with_buffer

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_random_velocity_field(self,
                                       displacement=(0, 10),
                                       gaussian_filters=(10,30),
                                       n_gaussian_filter_iter=6):
        """
        Generates random velocity field by smoothing a random initialization of pixels with a series of Gaussian filters.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128,512)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  random_seed=100)

            # Generate random velocity field:
            flowfield.generate_random_velocity_field(displacement=(0, 10),
                                                     gaussian_filters=(10,30),
                                                     n_gaussian_filter_iter=6)

        :param displacement: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) displacement
        :param gaussian_filters: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) Gaussian filter size (bandwidth) for smoothing out the random velocity fields to randomly sample from.
        :param n_gaussian_filter_iter: (optional)
            ``int`` specifying the number of iterations applying a Gaussian filter to the random velocity field to eventually arrive at a smoothed velocity map. With no iterations, each pixel attains a random velocity component value.
       """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        check_two_element_tuple(displacement, 'displacement')
        check_min_max_tuple(displacement, 'displacement')
        check_two_element_tuple(gaussian_filters, 'gaussian_filters')
        check_min_max_tuple(gaussian_filters, 'gaussian_filters')

        if type(n_gaussian_filter_iter) != int:
            raise ValueError("Parameter `n_gaussian_filter_iter` has to be of type 'int'.")

        if n_gaussian_filter_iter < 0:
            raise ValueError("Parameter `n_gaussian_filter_iter` has to be positive.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__displacement = displacement
        self.__gaussian_filters = gaussian_filters
        self.__n_gaussian_filter_iter = n_gaussian_filter_iter
        self.__velocity_field = []
        self.__velocity_field_magnitude = []

        self.__gaussian_filter_per_image = np.random.rand(self.__n_images) * (
                    self.__gaussian_filters[1] - self.__gaussian_filters[0]) + self.__gaussian_filters[0]
        self.__displacement_per_image = np.random.rand(self.__n_images) * (
                    self.__displacement[1] - self.__displacement[0]) + self.__displacement[0]

        for i in range(0, self.n_images):

            velocity_field_u = np.random.rand(self.size_with_buffer[0], self.size_with_buffer[1])
            velocity_field_v = np.random.rand(self.size_with_buffer[0], self.size_with_buffer[1])

            # Smooth out the random velocity field `n_gaussian_filter_iter` times:
            for _ in range(0, n_gaussian_filter_iter):
                velocity_field_u = scipy.ndimage.gaussian_filter(velocity_field_u, self.__gaussian_filter_per_image[i])
                velocity_field_v = scipy.ndimage.gaussian_filter(velocity_field_v, self.__gaussian_filter_per_image[i])

            velocity_field_u = velocity_field_u - np.mean(velocity_field_u)
            velocity_field_v = velocity_field_v - np.mean(velocity_field_v)

            velocity_magnitude = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            velocity_magnitude_scale = self.__displacement_per_image[i] / np.max(velocity_magnitude)

            velocity_field_u = velocity_magnitude_scale * velocity_field_u
            velocity_field_v = velocity_magnitude_scale * velocity_field_v

            self.__velocity_field_magnitude.append(np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2))
            self.__velocity_field.append((velocity_field_u, velocity_field_v))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_chebyshev_velocity_field(self,
                                          order=1):
        """
        Generates a velocity field using Chebyshev polynomials of the second kind. Each velocity component is computed as:

        .. math::

            u(h, w), v(h, w) = U_n(h) U_n(w) \\sin(n(h+w))

        where :math:`U_n` is the Chebyshev polynomial of the second kind and of order :math:`n`.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128,512)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  random_seed=100)

            # Generate Chebyshev velocity field:
            flowfield.generate_chebyshev_velocity_field(order=50)

        :param order: (optional)
            ``int`` specifying the order of the Chebyshev polynomial.
        """






        pass

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_spherical_harmonics_velocity_field(self,
                                                    degree=1,
                                                    order=1):
        """
        Generates a velocity field using spherical harmonics. Each velocity component is computed as:

        .. math::

            u(h, w), v(h, w) = Y_n^k \\sin(n(h+w))

        where :math:`Y_n^k` is the spherical harmonics function of order :math:`n` and degree :math:`k`.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128,512)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  random_seed=100)

            # Generate Chebyshev velocity field:
            flowfield.generate_spherical_harmonics_velocity_field(degree=1, order=1)

        """





        pass

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def upload_velocity_field(self, velocity_field_tuple):
        """
        Uploads a custom velocity field, e.g., generated from synthetic turbulence.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128,512)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  random_seed=100)

            # Importing external velocity field file:
            # ...
            # ...
            # ...

            # Upload velocity field:
            flowfield.upload_velocity_field(velocity_field_tuple)

        :param velocity_field_tuple:
            ``tuple`` of two ``numpy.ndarray`` specifying the velocity components.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(velocity_field_tuple, tuple):
            raise ValueError("Parameter `velocity_field_tuple` has to be of type 'tuple'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__velocity_field = []
        self.__velocity_field_magnitude = []

        for i in range(0, self.n_images):

            velocity_field_u = velocity_field_tuple[0][:,:,i]
            velocity_field_v = velocity_field_tuple[1][:,:,i]

            self.__velocity_field_magnitude.append(np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2))
            self.__velocity_field.append((velocity_field_u, velocity_field_v))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
