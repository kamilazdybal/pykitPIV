import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import chebyu, sph_harm
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
    Generates or uploads velocity field(s) for a set of ``n_images`` number of PIV image pairs.
    The velocity field(s) can later be used to advect particles between two consecutive PIV images.

    The velocity vector, :math:`\\vec{V} = [u, v]`, is represented as a four-dimensional tensor array
    of size :math:`(N, 2, H, W)`, where the second index corresponds to :math:`u` and :math:`v` velocity component, respectively.

    The velocity magnitude is computed as :math:`|\\vec{V}| = \\sqrt{u^2 + v^2}`
    and is also represented as a four-dimensional tensor array of size :math:`(N, 1, H, W)`.

    .. note::

        Only two-dimensional velocity fields are supported at the moment.

    **Example:**

    .. code:: python

        from pykitPIV import FlowField

        # We are going to generate 10 flow fields for 10 PIV image pairs:
        n_images = 10

        # Specify size in pixels for each image:
        image_size = (128, 512)

        # Initialize a flow field object:
        flowfield = FlowField(n_images=n_images,
                              size=image_size,
                              size_buffer=10,
                              random_seed=100)

    :param n_images:
        ``int`` specifying the number of PIV image pairs, :math:`N`, to create.
    :param size: (optional)
        ``tuple`` of two ``int`` elements specifying the size of each image in pixels. The first number is image height, :math:`h`, the second number is image width, :math:`w`.
    :param size_buffer: (optional)
        ``int`` specifying the buffer, :math:`b`, in pixels :math:`[\\text{px}]` to add to the image size in the width and height direction.
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
    - **velocity_field** - (read-only) ``numpy.ndarray`` specifying the velocity vector, :math:`\\vec{V} = [u, v]`, per each image. It has size :math:`(N, 2, H+2b, W+2b)`. The second index corresponds to :math:`u` and :math:`v` velocity component, respectively.
    - **velocity_field_magnitude** - (read-only) ``numpy.ndarray`` specifying the velocity field magnitude, :math:`|\\vec{V}| = \\sqrt{u^2 + v^2}`, per each image. It has size :math:`(N, 1, H+2b, W+2b)`.
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

            # Access the velocity components tensor:
            flowfield.velocity_field

            # Access the velocity field magnitude:
            flowfield.velocity_field_magnitude

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
        self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]))
        self.__velocity_field_magnitude = np.zeros((self.__n_images, 1, self.size_with_buffer[0], self.size_with_buffer[1]))

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

            self.__velocity_field_magnitude[i, 0, :, :] = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            self.__velocity_field[i, 0, :, :] = velocity_field_u
            self.__velocity_field[i, 1, :, :] = velocity_field_v

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_checkered_velocity_field(self,
                                          displacement=(0, 10),
                                          m=10,
                                          n=10,
                                          rotation=None):
        """
        Generates a checkered velocity field. Each velocity component is computed as:

        .. math::

            u(h, w), v(h, w) = \\sin(m \cdot h) \cdot \\cos(n \cdot w)

        where :math:`m` and :math:`n` free parameters.

        Optionally, the checkered pattern can be rotated with the additional rotation term:

        .. math::

            u(h, w), v(h, w) = \\sin(m \cdot h) \cdot \\cos(n \cdot w) \cdot \\sin(r \cdot (h + w))

        were :math:`r` is a free parameter.

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

            # Generate checkered velocity field:
            flowfield.generate_checkered_velocity_field(displacement=(0, 10),
                                                        m=10,
                                                        n=10,
                                                        rotation=10)

            # Access the velocity components tensor:
            flowfield.velocity_field

            # Access the velocity field magnitude:
            flowfield.velocity_field_magnitude

        :param displacement: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) displacement
        :param m: (optional)
            ``int`` or ``float`` specifying the scaling for the sine function, :math:`m`.
        :param n: (optional)
            ``int`` or ``float`` specifying the scaling for the cosine function, :math:`n`.
        :param rotation: (optional)
            ``int`` or ``float`` specifying the scaling for the rotation with the sine function, :math:`r`. If set to ``None``,
            the checkered pattern will not be rotated.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        check_two_element_tuple(displacement, 'displacement')
        check_min_max_tuple(displacement, 'displacement')

        if (not isinstance(m, int)) and (not isinstance(m, float)):
            raise ValueError("Parameter `m` has to be of type 'int' or 'float'.")

        if (not isinstance(n, int)) and (not isinstance(n, float)):
            raise ValueError("Parameter `n` has to be of type 'int' or 'float'.")

        if rotation is not None:
            if (not isinstance(rotation, int)) and (not isinstance(rotation, float)):
                raise ValueError("Parameter `rotation` has to be of type 'int' or 'float'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__displacement = displacement
        self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]))
        self.__velocity_field_magnitude = np.zeros((self.__n_images, 1, self.size_with_buffer[0], self.size_with_buffer[1]))

        self.__displacement_per_image = np.random.rand(self.__n_images) * (self.__displacement[1] - self.__displacement[0]) + self.__displacement[0]

        h = np.linspace(-1, 1, self.size_with_buffer[0])
        w = np.linspace(-1, 1, self.size_with_buffer[1])

        (grid_w, grid_h) = np.meshgrid(w, h)

        for i in range(0, self.n_images):

            if rotation is None:
                velocity_field_u = np.sin(m * grid_w) * np.cos(n * grid_h)
                velocity_field_v = np.sin(m * grid_w) * np.cos(n * grid_h)
            else:
                velocity_field_u = np.sin(m * grid_w) * np.cos(n * grid_h) * np.sin(rotation * (grid_w + grid_h))
                velocity_field_v = np.sin(m * grid_w) * np.cos(n * grid_h) * np.sin(rotation * (grid_w + grid_h))

            velocity_field_u = velocity_field_u - np.mean(velocity_field_u)
            velocity_field_v = velocity_field_v - np.mean(velocity_field_v)

            velocity_magnitude = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            velocity_magnitude_scale = self.__displacement_per_image[i] / np.max(velocity_magnitude)

            velocity_field_u = velocity_magnitude_scale * velocity_field_u
            velocity_field_v = velocity_magnitude_scale * velocity_field_v

            self.__velocity_field_magnitude[i, 0, :, :] = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            self.__velocity_field[i, 0, :, :] = velocity_field_u
            self.__velocity_field[i, 1, :, :] = velocity_field_v

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_chebyshev_velocity_field(self,
                                          displacement=(0, 10),
                                          start=0.3,
                                          stop=0.8,
                                          order=10):
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
            flowfield.generate_chebyshev_velocity_field(order=10)

            # Access the velocity components tensor:
            flowfield.velocity_field

            # Access the velocity field magnitude:
            flowfield.velocity_field_magnitude

        :param displacement: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) displacement.
        :param start: (optional)
            ``int`` or ``float`` specifying the start value for generating the coordinate grid.
        :param stop: (optional)
            ``int`` or ``float`` specifying the end value for generating the coordinate grid.
        :param order: (optional)
            ``int`` specifying the order of the Chebyshev polynomial.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        check_two_element_tuple(displacement, 'displacement')
        check_min_max_tuple(displacement, 'displacement')

        if (not isinstance(start, int)) and (not isinstance(start, float)):
            raise ValueError("Parameter `start` has to be of type 'int' or 'float'.")

        if (not isinstance(stop, int)) and (not isinstance(stop, float)):
            raise ValueError("Parameter `stop` has to be of type 'int' or 'float'.")

        if (not isinstance(order, int)):
            raise ValueError("Parameter `order` has to be of type 'int'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__displacement = displacement
        self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]))
        self.__velocity_field_magnitude = np.zeros((self.__n_images, 1, self.size_with_buffer[0], self.size_with_buffer[1]))
        self.__displacement_per_image = np.random.rand(self.__n_images) * (self.__displacement[1] - self.__displacement[0]) + self.__displacement[0]

        h = np.linspace(start, stop, self.size_with_buffer[0])
        w = np.linspace(start, stop, self.size_with_buffer[1])
        (grid_w, grid_h) = np.meshgrid(w, h)

        # Generate Chebyshev polynomials:
        un = chebyu(order)

        for i in range(0, self.n_images):

            # Generate a 2D field:
            velocity_field_u = un(grid_h) * un(grid_w) * np.sin((grid_h + grid_w) * order)
            velocity_field_v = un(grid_h) * un(grid_w) * np.sin((grid_h + grid_w) * order)

            velocity_magnitude = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            velocity_magnitude_scale = self.__displacement_per_image[i] / np.max(velocity_magnitude)

            velocity_field_u = velocity_magnitude_scale * velocity_field_u
            velocity_field_v = velocity_magnitude_scale * velocity_field_v

            self.__velocity_field_magnitude[i, 0, :, :] = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            self.__velocity_field[i, 0, :, :] = velocity_field_u
            self.__velocity_field[i, 1, :, :] = velocity_field_v

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_spherical_harmonics_velocity_field(self,
                                                    displacement=(0, 10),
                                                    start=0.3,
                                                    stop=0.8,
                                                    order=1,
                                                    degree=1):
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

            # Generate spherical harmonics velocity field:
            flowfield.generate_spherical_harmonics_velocity_field(degree=1, order=1)

            # Access the velocity components tensor:
            flowfield.velocity_field

            # Access the velocity field magnitude:
            flowfield.velocity_field_magnitude

        :param displacement: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) displacement.
        :param start: (optional)
            ``int`` or ``float`` specifying the start value for generating the azimuthal/polar coordinate grid.
        :param stop: (optional)
            ``int`` or ``float`` specifying the end value for generating the azimuthal/polar coordinate grid.
        :param order: (optional)
            ``int`` specifying the order, :math:`n`, of the spherical harmonics.
        :param degree: (optional)
            ``int`` specifying the degree, :math:`k`, of the spherical harmonics.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        check_two_element_tuple(displacement, 'displacement')
        check_min_max_tuple(displacement, 'displacement')

        if (not isinstance(start, int)) and (not isinstance(start, float)):
            raise ValueError("Parameter `start` has to be of type 'int' or 'float'.")

        if (not isinstance(stop, int)) and (not isinstance(stop, float)):
            raise ValueError("Parameter `stop` has to be of type 'int' or 'float'.")

        if (not isinstance(order, int)):
            raise ValueError("Parameter `order` has to be of type 'int'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__displacement = displacement
        self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]))
        self.__velocity_field_magnitude = np.zeros((self.__n_images, 1, self.size_with_buffer[0], self.size_with_buffer[1]))
        self.__displacement_per_image = np.random.rand(self.__n_images) * (self.__displacement[1] - self.__displacement[0]) + self.__displacement[0]

        h = np.linspace(start, stop, self.size_with_buffer[0])
        w = np.linspace(start, stop, self.size_with_buffer[1])

        # Rescale the azimuthal coordinate:
        h = (h - h.min()) / (h.max() - h.min()) * np.pi * 0.2 + np.pi * 0.4

        # Rescale the polar coordinate:
        w = (w - w.min()) / (w.max() - w.min()) * np.pi * 0.2 / 6 * 5 + (np.pi * 0.5 - np.pi * 0.2 / 6 * 5 / 2)

        (grid_w, grid_h) = np.meshgrid(w, h)

        for i in range(0, self.n_images):

            # Generate a 2D field:
            velocity_field_u = np.real(sph_harm(order, degree, grid_h, grid_w)) * np.sin((grid_h + grid_w) * degree)
            velocity_field_v = np.real(sph_harm(order, degree, grid_h, grid_w)) * np.sin((grid_h + grid_w) * degree)

            velocity_magnitude = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            velocity_magnitude_scale = self.__displacement_per_image[i] / np.max(velocity_magnitude)

            velocity_field_u = velocity_magnitude_scale * velocity_field_u
            velocity_field_v = velocity_magnitude_scale * velocity_field_v

            self.__velocity_field_magnitude[i, 0, :, :] = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            self.__velocity_field[i, 0, :, :] = velocity_field_u
            self.__velocity_field[i, 1, :, :] = velocity_field_v

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def upload_velocity_field(self, velocity_field):
        """
        Uploads a custom velocity field, *e.g.*, generated from synthetic turbulence.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to use 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128,512)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  random_seed=100)

            # Importing an external velocity field file:
            # ...
            # ...

            # Prepare the velocity_field variable that has shape (10, 2, 148, 532):
            # velocity_field = ...

            # Upload velocity field:
            flowfield.upload_velocity_field(velocity_field)

        :param velocity_field:
            ``numpy.ndarray`` specifying the velocity components. It should be of size :math:`(N, 2, H, W)`,
            where :math:`N` is the number PIV image pairs, :math:`2` refers to each velocity component,
            :math:`H` is the height and :math:`W` is the width of each PIV image. It can also be of size :math:`(1, 2, H, W)`,
            in which case the same velocity field will be applied to all PIV image pairs.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(velocity_field, np.ndarray):
            raise ValueError("Parameter `velocity_field` has to be of type 'numpy.ndarray'.")

        (N, n_velocity_components, H, W) = np.shape(velocity_field)

        if N == 1 and self.n_images != 1:
            print('The same velocity field will be applied to all PIV image pairs.')
        else:
            if N != self.n_images:
                raise ValueError("The number of PIV image pairs does not match the value at class initialization.")

        if n_velocity_components != 2:
            raise ValueError("The number of velocity components has to be equal to 2.")

        if self.size_buffer > 0:
            if H != self.__height_with_buffer:
                raise ValueError("Improper height of the velocity field. It should be equal to " + str(self.__height_with_buffer) + '.')
            if W != self.__width_with_buffer:
                raise ValueError("Improper width of the velocity field. It should be equal to " + str(self.__width_with_buffer) + '.')
        else:
            if H != self.size[0]:
                raise ValueError("Improper height of the velocity field. It should be equal to " + str(self.size[0]) + '.')
            if W != self.size[1]:
                raise ValueError("Improper width of the velocity field. It should be equal to " + str(self.size[1]) + '.')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if N == 1:

            self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]))

            for i in range(0, self.__n_images):
                self.__velocity_field[i, 0, :, :] = velocity_field[0, 0, :, :]
                self.__velocity_field[i, 1, :, :] = velocity_field[0, 1, :, :]

            self.__velocity_field_magnitude = np.sqrt(self.__velocity_field[:, 0:1, :, :] ** 2 + self.__velocity_field[:, 1:2, :, :] ** 2)

        else:

            self.__velocity_field = velocity_field
            self.__velocity_field_magnitude = np.sqrt(velocity_field[:, 0:1, :, :] ** 2 + velocity_field[:, 1:2, :, :] ** 2)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
