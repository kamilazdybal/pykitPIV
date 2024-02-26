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
    Generates velocity fields to advect the particles between two consecutive images.

    **Example:**

    .. code:: python

        import numpy as np
        import cmcrameri.cm as cmc
        from pykitPIV import FlowField, Image

        # Specify size in pixels for each image:
        image_size = (128,512)

        # Initialize a flow field object that generates a random velocity field for one image pair:
        flowfield = FlowField(1,
                              size=image_size,
                              size_buffer=10,
                              flow_mode='random',
                              gaussian_filters=(8,10),
                              n_gaussian_filter_iter=10,
                              sin_period=(30,300),
                              displacement=(0,10),
                              random_seed=100)

    We can now visualize the generated random velocity field using the ``Image`` class.

    .. code:: python

        # Initialize an image object:
        image = Image(random_seed=100)

        # Add the velocity field to the image:
        image.add_velocity_field(flowfield)

        # Visualize the velocity field:
        image.plot_velocity_field(0,
                                  with_buffer=True,
                                  xlabel='Width [px]',
                                  ylabel='Height [px]',
                                  title=('Example random velocity component $u$', 'Example random velocity component $v$'),
                                  cmap=cmc.oslo_r,
                                  figsize=(10,2),
                                  filename='example-random-velocity-field.png');

    The code above will return two figures showing the random velocity field components, :math:`u` and :math:`v`:

    .. image:: ../images/example-random-velocity-field-u.png
      :width: 700
      :align: center

    .. image:: ../images/example-random-velocity-field-v.png
      :width: 700
      :align: center

    We can also visualize velocity magnitude field. Optionally, a stream plot or a quiver plot can be added on top to visualize the velocity vector field:

    .. code:: python

        image.plot_velocity_field_magnitude(0,
                                            add_quiver=True,
                                            quiver_step=10,
                                            quiver_color='r',
                                            xlabel='Width [px]',
                                            ylabel='Height [px]',
                                            title='Example random velocity field magnitude',
                                            cmap=cmc.oslo_r,
                                            figsize=(10,2),
                                            filename='example-random-velocity-field-magnitud-quiver.png');

    .. image:: ../images/example-random-velocity-field-magnitude-quiver.png
      :width: 700
      :align: center

    .. code:: python

        image.plot_velocity_field_magnitude(0,
                                            add_streamplot=True,
                                            streamplot_density=1,
                                            streamplot_color='r',
                                            xlabel='Width [px]',
                                            ylabel='Height [px]',
                                            title='Example random velocity field magnitude',
                                            cmap=cmc.oslo_r,
                                            figsize=(10,2),
                                            filename='example-random-velocity-field-magnitude-streamplot.png');

    .. image:: ../images/example-random-velocity-field-magnitude-streamplot.png
      :width: 700
      :align: center

    :param n_images:
        ``int`` specifying the number of image pairs to create.
    :param size: (optional)
        ``tuple`` of two ``int`` elements specifying the size of each image in pixels. The first number is image height, the second number is image width.
    :param size_buffer: (optional)
        ``int`` specifying the buffer in pixels :math:`[\\text{px}]` to add to the image size in the width and height direction.
        This number should be approximately equal to the maximum displacement that particles are subject to in order to allow for new particles to arrive into the image area.
    :param flow_mode: (optional)
        ``str`` specifying the mode for the velocity field generation. It can be one of the following: ``'random'``, ``'random-sinusoidal'``, ``'quadrant'``, or ``'checkerboard'``.
    :param gaussian_filters: (optional)
        ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) Gaussian filter size (bandwidth) for smoothing out the random velocity fields to randomly sample from.
    :param n_gaussian_filter_iter: (optional)
        ``int`` specifying the number of iterations applying a Gaussian filter to the random velocity field to eventually arrive at a smoothed velocity map. With no iterations, each pixel attains a random velocity component value.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``. If specified, all image generation is reproducible.
    """

    def __init__(self,
                 n_images,
                 size=(512, 512),
                 size_buffer=10,
                 flow_mode='random',
                 displacement=(0, 10),
                 gaussian_filters=(10,30),
                 n_gaussian_filter_iter=6,
                 sin_period=(30,300),
                 random_seed=None):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if type(n_images) != int:
            raise ValueError("Parameter `n_images` has to of type 'int'.")

        if n_images < 0:
            raise ValueError("Parameter `n_images` has to be positive.")

        check_two_element_tuple(size, 'size')

        if not isinstance(size_buffer, int):
            raise ValueError("Parameter `size_buffer` has to be of type 'int'.")

        if size_buffer < 0:
            raise ValueError("Parameter `size_buffer` has to non-negative.")

        __flow_mode = ['random', 'random-sinusoidal', 'quadrant', 'checkerboard']
        if flow_mode not in __flow_mode:
            raise ValueError("Parameter `flow_mode` has to be 'random', 'random-sinusoidal', 'quadrant', or 'checkerboard'.")

        check_two_element_tuple(displacement, 'displacement')
        check_min_max_tuple(displacement, 'displacement')
        check_two_element_tuple(gaussian_filters, 'gaussian_filters')
        check_min_max_tuple(gaussian_filters, 'gaussian_filters')

        if type(n_gaussian_filter_iter) != int:
            raise ValueError("Parameter `n_gaussian_filter_iter` has to of type 'int'.")

        if n_gaussian_filter_iter < 0:
            raise ValueError("Parameter `n_gaussian_filter_iter` has to be positive.")

        check_two_element_tuple(sin_period, 'sin_period')
        check_min_max_tuple(sin_period, 'sin_period')

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
        self.__flow_mode = flow_mode
        self.__displacement = displacement
        self.__gaussian_filters = gaussian_filters
        self.__n_gaussian_filter_iter = n_gaussian_filter_iter
        self.__sin_period = sin_period
        self.__random_seed = random_seed

        # Compute the image outline that serves as a buffer:
        self.__height_with_buffer = self.size[0] + 2 * self.size_buffer
        self.__width_with_buffer = self.size[1] + 2 * self.size_buffer
        self.__size_with_buffer = (self.__height_with_buffer, self.__width_with_buffer)

        # Generate random velocity field:
        if flow_mode == 'random':

            self.__velocity_field = []
            self.__velocity_field_magnitude = []

            self.__gaussian_filter_per_image = np.random.rand(self.__n_images) * (self.__gaussian_filters[1] - self.__gaussian_filters[0]) + self.__gaussian_filters[0]
            self.__displacement_per_image = np.random.rand(self.__n_images) * (self.__displacement[1] - self.__displacement[0]) + self.__displacement[0]

            for i in range(0, self.n_images):

                velocity_field_u = np.random.rand(self.size_with_buffer[0], self.size_with_buffer[1])
                velocity_field_v = np.random.rand(self.size_with_buffer[0], self.size_with_buffer[1])

                # Smooth out the random velocity field `n_gaussian_filter_iter` times:
                for _ in range(0,n_gaussian_filter_iter):
                    velocity_field_u = scipy.ndimage.gaussian_filter(velocity_field_u, self.__gaussian_filter_per_image[i])
                    velocity_field_v = scipy.ndimage.gaussian_filter(velocity_field_v, self.__gaussian_filter_per_image[i])

                velocity_field_u = velocity_field_u - np.mean(velocity_field_u)
                velocity_field_v = velocity_field_v - np.mean(velocity_field_v)

                velocity_magnitude = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
                velocity_magnitude_scale = self.__displacement_per_image[i] / np.max(velocity_magnitude)

                velocity_field_u = velocity_magnitude_scale * velocity_field_u
                velocity_field_v = velocity_magnitude_scale * velocity_field_v

                self.__velocity_field_magnitude.append(np.sqrt(velocity_field_u**2 + velocity_field_v**2))
                self.__velocity_field.append((velocity_field_u, velocity_field_v))

        else:

            # Set to None for the moment for any other flow_mode:
            self.__velocity_field = None
            self.__velocity_field_magnitude = None
            self.__gaussian_filter_per_image = None
            self.__displacement_per_image = None

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
    def flow_mode(self):
        return self.__flow_mode

    @property
    def displacement(self):
        return self.__displacement

    @property
    def gaussian_filters(self):
        return self.__gaussian_filters

    @property
    def n_gaussian_filter_iter(self):
        return self.__n_gaussian_filter_iter

    @property
    def sin_period(self):
        return self.__sin_period

    @property
    def random_seed(self):
        return self.__random_seed

    # Properties computed at class init:
    @property
    def size_with_buffer(self):
        return self.__size_with_buffer

    @property
    def gaussian_filter_per_image(self):
        return self.__gaussian_filter_per_image

    @property
    def displacement_per_image(self):
        return self.__displacement_per_image

    @property
    def velocity_field(self):
        return self.__velocity_field

    @property
    def velocity_field_magnitude(self):
        return self.__velocity_field_magnitude