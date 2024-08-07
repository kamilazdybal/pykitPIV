import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import scipy
import warnings
from pykitPIV.checks import *

########################################################################################################################
########################################################################################################################
####
####    Class: Particle
####
########################################################################################################################
########################################################################################################################

class Particle:
    """
    Generates particles with specified properties for a set of ``n_images`` number of PIV image pairs.

    **Example:**

    .. code:: python

        from pykitPIV import Particle

        # We are going to generate 10 PIV image pairs:
        n_images = 10

        # Specify size in pixels for each image:
        image_size = (128,512)

        # Initialize a particle object:
        particles = Particle(n_images=n_images,
                             size=image_size,
                             size_buffer=10,
                             diameters=(2,4),
                             distances=(1,2),
                             densities=(0.01,0.05),
                             signal_to_noise=(5,20),
                             diameter_std=1,
                             seeding_mode='random',
                             random_seed=100)

    :param n_images:
        ``int`` specifying the number of image pairs to create.
    :param size: (optional)
        ``tuple`` of two ``int`` elements specifying the size of each image in pixels :math:`[\\text{px}]`. The first number is image height, :math:`h`, the second number is image width, :math:`w`.
    :param size_buffer: (optional)
        ``int`` specifying the buffer in pixels :math:`[\\text{px}]` to add to the image size in the width and height direction.
        This number should be approximately equal to the maximum displacement that particles are subject to in order to allow new particles to arrive into the image area
        and old particles to exit the image area.
    :param diameters: (optional)
        ``tuple`` of two ``int`` elements specifying the minimum (first element) and maximum (second element) particle diameter in pixels :math:`[\\text{px}]` to randomly sample from.
    :param distances: (optional)
        ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) particle distances in pixels :math:`[\\text{px}]` to randomly sample from. Only used when ``seeding_mode`` is ``'poisson'``.
    :param densities: (optional)
        ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) particle seeding density on an image in particle per pixel :math:`[\\text{ppp}]` to randomly sample from. Only used when ``seeding_mode`` is ``'random'``.
    :param diameter_std: (optional)
        ``float`` or ``int`` specifying the standard deviation in pixels :math:`[\\text{px}]` for the distribution of particle diameters.
    :param signal_to_noise: (optional)
        ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) signal-to-noise ratio for particle generation. [Kamila] I still wonder if this should rather be a property of Motion class. Maybe not, Motion can always access this class attribute.
    :param seeding_mode: (optional)
        ``str`` specifying the seeding mode for initializing particles in the image domain. It can be one of the following: ``'random'``, ``'poisson'``.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``. If specified, all image generation is reproducible.

    **Attributes:**

    - **n_images** - (read-only) as per user input.
    - **size** - (read-only) as per user input.
    - **size_buffer** - (read-only) as per user input.
    - **diameters** - (read-only) as per user input.
    - **distances** - (read-only) as per user input.
    - **densities** - (read-only) as per user input.
    - **diameter_std** - (read-only) as per user input.
    - **signal_to_noise** - (read-only) as per user input.
    - **seeding_mode** - (read-only) as per user input.
    - **random_seed** - (read-only) as per user input.
    - **size_with_buffer** - (read-only) ``tuple`` specifying the size of each image in pixels with buffer added.
    - **diameter_per_image** - (read-only) ``numpy.ndarray`` specifying the template for the particle diameters in pixels :math:`[\\text{px}]` for each image. Template diameters are random numbers between ``diameters[0]`` and ``diameters[1]``.
    - **distance_per_image** - (read-only) ``numpy.ndarray`` specifying the template for the particle distances in pixels :math:`[\\text{px}]` for each image. Template distances are random numbers between ``distances[0]`` and ``distances[1]``.
    - **density_per_image** - (read-only) ``numpy.ndarray`` specifying the template for the particle densities in particle per pixel :math:`[\\text{ppp}]` for each image. Template densities are random numbers between ``densities[0]`` and ``densities[1]``.
    - **SNR_per_image** - (read-only) ``numpy.ndarray`` specifying the template for the signal-to-noise ratio for each image. Template signal-to-noise are random numbers between ``signal_to_noise[0]`` and ``signal_to_noise[1]``.
    - **n_of_particles** - (read-only) ``list`` specifying the number of particles created for each image based on each template density.
    - **particle_coordinates** - (read-only) ``list`` specifying the absolute coordinates of all particle centers for each image. The posititions are computed based on the ``seeding_mode``. The first element in each tuple are the coordinates along the image height, and the second element are the coordinates along the image width.
    - **particle_positions** - (read-only) ``list`` specifying the position per pixel of all particle centers for each image. The posititions are computed based on the ``seeding_mode``.
    - **particle_diameters** - (read-only) ``list`` specifying the diameters of all seeded particles in pixels :math:`[\\text{px}]` for each image based on each template diameter.
   """

    def __init__(self,
                 n_images,
                 size=(512,512),
                 size_buffer=10,
                 diameters=(3,6),
                 distances=(0.5,2),
                 densities=(0.05,0.1),
                 signal_to_noise=(5,20),
                 diameter_std=0.1,
                 seeding_mode='random',
                 random_seed=None):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if type(n_images) != int:
            raise ValueError("Parameter `n_images` has to of type 'int'.")

        if n_images <= 0:
            raise ValueError("Parameter `n_images` has to be positive. At least one image has to be generated.")

        check_two_element_tuple(size, 'size')

        if not isinstance(size_buffer, int):
            raise ValueError("Parameter `size_buffer` has to be of type 'int'.")

        if size_buffer < 0:
            raise ValueError("Parameter `size_buffer` has to non-negative.")

        check_two_element_tuple(diameters, 'diameters')
        check_min_max_tuple(diameters, 'diameters')
        check_two_element_tuple(distances, 'distances')
        check_min_max_tuple(distances, 'distances')
        check_two_element_tuple(densities, 'densities')
        check_min_max_tuple(densities, 'densities')
        check_two_element_tuple(signal_to_noise, 'signal_to_noise')
        check_min_max_tuple(signal_to_noise, 'signal_to_noise')

        if (not isinstance(diameter_std, float)) and (not isinstance(diameter_std, int)):
            raise ValueError("Parameter `diameter_std` has to be of type 'float' or 'int'.")

        __seeding_mode = ['random', 'poisson']
        if seeding_mode not in __seeding_mode:
            raise ValueError("Parameter `seeding_mode` has to be 'random', or 'poisson'.")

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
        self.__diameters = diameters
        self.__distances = distances
        self.__densities = densities
        self.__signal_to_noise = signal_to_noise
        self.__diameter_std = diameter_std
        self.__seeding_mode = seeding_mode
        self.__random_seed = random_seed

        # Compute the image outline that serves as a buffer:
        self.__height_with_buffer = self.size[0] + 2 * self.size_buffer
        self.__width_with_buffer = self.size[1] + 2 * self.size_buffer
        self.__size_with_buffer = (self.__height_with_buffer, self.__width_with_buffer)

        # Initialize parameters for particle generation:
        self.__particle_diameter_per_image = np.random.rand(self.__n_images) * (self.__diameters[1] - self.__diameters[0]) + self.__diameters[0]
        self.__particle_distance_per_image = np.random.rand(self.__n_images) * (self.__distances[1] - self.__distances[0]) + self.__distances[0]
        self.__particle_SNR_per_image = np.random.rand(self.__n_images) * (self.__signal_to_noise[1] - self.__signal_to_noise[0]) + self.__signal_to_noise[0]

        # Compute the seeding density for each image:
        if seeding_mode == 'random':

            self.__particle_density_per_image = np.random.rand(self.__n_images) * (self.__densities[1] - self.__densities[0]) + self.__densities[0]

        elif seeding_mode == 'poisson':

            print('Poisson sampling is not supported yet.')

        # Compute the total number of particles for a given particle density on each image:
        n_of_particles = self.__size[0] * self.__size[1] * self.__particle_density_per_image
        self.__n_of_particles = [int(i) for i in n_of_particles]

        # Initialize particle positions and particle diameters on each of the ``n_image`` images:

        particle_coordinates = []
        particle_positions = []
        particle_diameters = []

        for i in range(0,self.n_images):

            if seeding_mode == 'random':

                # Generate absolute coordinates for particles' centers within the total available image area (drawn from random uniform distribution):
                self.__y_coordinates = self.__height_with_buffer * np.random.rand(self.n_of_particles[i])
                self.__x_coordinates = self.__width_with_buffer * np.random.rand(self.n_of_particles[i])

                particle_coordinates.append((self.__y_coordinates, self.__x_coordinates))

                # Populate a matrix that shows particle locations per pixel of the image area:
                seeded_array = np.zeros((self.__height_with_buffer, self.__width_with_buffer))
                for x, y in zip(np.floor(self.__x_coordinates).astype(int), np.floor(self.__y_coordinates).astype(int)):

                    seeded_array[y, x] += 1

                particle_positions.append(seeded_array)

            elif seeding_mode == 'poisson':

                print('Poisson sampling is not supported yet.')

            # Generate diameters for all particles in a current image:
            particle_diameters.append(np.random.normal(self.diameter_per_image[i], self.diameter_std, self.n_of_particles[i]))

        # Initialize particle coordinates:
        self.__particle_coordinates = particle_coordinates

        # Initialize particle positions:
        self.__particle_positions = particle_positions

        # Initialize particle diameters:
        self.__particle_diameters = particle_diameters

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
    def diameters(self):
        return self.__diameters

    @property
    def distances(self):
        return self.__distances

    @property
    def densities(self):
        return self.__densities

    @property
    def signal_to_noise(self):
        return self.__signal_to_noise

    @property
    def diameter_std(self):
        return self.__diameter_std

    @property
    def seeding_mode(self):
        return self.__seeding_mode

    @property
    def random_seed(self):
        return self.__random_seed

    # Properties computed at class init:
    @property
    def size_with_buffer(self):
        return self.__size_with_buffer

    @property
    def diameter_per_image(self):
        return self.__particle_diameter_per_image

    @property
    def distance_per_image(self):
        return self.__particle_distance_per_image

    @property
    def density_per_image(self):
        return self.__particle_density_per_image

    @property
    def SNR_per_image(self):
        return self.__particle_SNR_per_image

    @property
    def n_of_particles(self):
        return self.__n_of_particles

    @property
    def particle_coordinates(self):
        return self.__particle_coordinates

    @property
    def particle_positions(self):
        return self.__particle_positions

    @property
    def particle_diameters(self):
        return self.__particle_diameters

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # ##################################################################################################################

    # Plotting functions

    # ##################################################################################################################

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_properties_per_image(self):
        """
        Plots statistical properties of the generated particles on one selected image out of all ``n_images`` images.

        :return:
            - **plt** - ``matplotlib.pyplot`` image handle.
        """

        pass

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_properties_across_images(self):
        """
        Plots statistical properties of the generated particles across all ``n_images`` images.


        :return:
            - **plt** - ``matplotlib.pyplot`` image handle.
        """

        pass