import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import scipy
import warnings

def check_two_element_tuple(x, name):

    if not isinstance(x, tuple):
        raise ValueError("Parameter `" + name + "` has to of type 'tuple'.")

    if len(x) != 2:
        raise ValueError("Parameter `" + name + "` has to have two elements.")

########################################################################################################################
########################################################################################################################
####
####    Class: Particle
####
########################################################################################################################
########################################################################################################################

class Particle:
    """
    Generates particles for a set of ``n_images`` number of image pairs.

    :param n_images:
        ``int`` specifying the number of image pairs to create.
    :param size:
        ``tuple`` of two ``int`` elements specifying the size of each image in pixels. The first number is image height, the second number is image width.
    :param diameters:
        ``tuple`` of two ``int`` elements specifying the minimum and maximum particle diameter in pixels to randomly sample from.
    :param distances:
        ``tuple`` of two elements specifying the minimum and maximum particle distances to randomly sample from. Only used when ``seeding_mode`` is ``'poisson'``.
    :param densities:
        ``tuple`` of two elements specifying the minimum and maximum particle density on an image to randomly sample from. Only used when ``seeding_mode`` is ``'random'``.
    :param signal_to_noise:
        ``tuple`` of two elements specifying the minimum and maximum signal-to-noise ratio for particle generation. [Kamila] I still wonder if this should rather be a property of Motion class. Maybe not, Motion can allways access this class attribute.
    :param seeding_mode:
        ``str`` specifying the seeding mode for initializing particles in the image domain. It can be one of the following: ``'random'``, ``'poisson'``.
    :param random_seed:
        ``int`` specifying the random seed for random number generation in ``numpy``.
   """

    def __init__(self,
                 n_images,
                 size=(512,512),
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

        if n_images < 0:
            raise ValueError("Parameter `n_images` has to be positive.")

        check_two_element_tuple(size, 'size')
        check_two_element_tuple(diameters, 'diameters')
        check_two_element_tuple(distances, 'distances')
        check_two_element_tuple(densities, 'densities')
        check_two_element_tuple(signal_to_noise, 'signal_to_noise')

        if type(diameter_std) != float:
            raise ValueError("Parameter `diameter_std` has to of type 'float'.")

        __seeding_mode = ['random', 'poisson']
        if seeding_mode not in __seeding_mode:
            raise ValueError("Parameter `seeding_mode` has to be 'random', or 'poisson'.")

        if random_seed is not None:
            if type(random_seed) != int:
                raise ValueError("Parameter `random_seed` has to of type 'int'.")
            else:
                np.random.seed(seed=random_seed)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:
        self.__n_images = n_images
        self.__size = size
        self.__diameters = diameters
        self.__distances = distances
        self.__densities = densities
        self.__signal_to_noise = signal_to_noise
        self.__diameter_std = diameter_std
        self.__seeding_mode = seeding_mode
        self.__random_seed = random_seed

        # Initialize parameters for particle generation:
        self.__particle_diameter_per_image = np.random.rand(self.__n_images) * (self.__diameters[0] - self.__diameters[1]) + self.__diameters[0]
        self.__particle_distance_per_image = np.random.rand(self.__n_images) * (self.__distances[0] - self.__distances[1]) + self.__distances[0]
        self.__particle_SNR_per_image = np.random.rand(self.__n_images) * (self.__signal_to_noise[1] - self.__signal_to_noise[0]) + self.__signal_to_noise[0]

        # Compute the seeding density for each image:
        if seeding_mode == 'random':

            self.__particle_density_per_image = np.random.rand(self.__n_images) * (self.__densities[1] - self.__densities[0]) + self.__densities[0]

        elif seeding_mode == 'poisson':

            particle_densities = np.zeros((self.n_images,))

            for i in range(0, self.n_images):
                sx = np.arange(((self.__particle_diameter_per_image[i] + self.__particle_distance_per_image[i]) / 2), (self.size[0] - (self.__particle_diameter_per_image[i] + self.__particle_distance_per_image[i]) / 2),(self.__particle_diameter_per_image[i] + self.__particle_distance_per_image[i]))
                sy = np.arange(((self.__particle_diameter_per_image[i] + self.__particle_distance_per_image[i]) / 2), (self.size[1] - (self.__particle_diameter_per_image[i] + self.__particle_distance_per_image[i]) / 2),(self.__particle_diameter_per_image[i] + self.__particle_distance_per_image[i]))
                particle_densities[i] = len(sx) * len(sy) / self.__size[0] / self.__size[1]

            self.__particle_density_per_image = particle_densities

        # Compute the total number of particles for a given particle density on each image:
        n_of_particles = self.__size[0] * self.__size[1] * self.__particle_density_per_image
        self.__n_of_particles = [int(i) for i in n_of_particles]

        # Initialize particle positions:
        self.__particle_positions = None

        # Initialize particle radii:
        self.__particle_radii = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Properties coming from user inputs:
    @property
    def n_images(self):
        return self.__n_images

    @property
    def size(self):
        return self.__size

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
    def diameter_per_image(self):
        return self.__particle_diameter_per_image

    @property
    def distance_per_image(self):
        return self.__particle_distance_per_image

    @property
    def density_per_image(self):
        return self.__particle_density_per_image

    @property
    def n_of_particles(self):
        return self.__n_of_particles

    @property
    def particle_positions(self):
        return self.__particle_positions

    @property
    def particle_radii(self):
        return self.__particle_radii

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def seed_particles(self):
        """
        Initializes particle positions on each image.
        """

        np.random.seed(seed=self.random_seed)

        particle_positions = []
        particle_radii = []

        for i in range(0,self.n_images):

            if self.seeding_mode == 'random':

                random_particle_positions = list(np.random.choice(self.size[0] * self.size[1], size=self.n_of_particles[i], replace=False))
                seeded_array = np.zeros((self.size[0], self.size[1]))
                seeded_array.ravel()[random_particle_positions] = 1
                particle_positions.append(seeded_array)

            elif self.seeding_mode == 'poisson':

                print('Poisson sampling is not supported yet.')

            particle_radii.append(np.random.rand(self.n_of_particles[i]) * self.diameter_std/2/3 + self.diameters[i]/2)

        self.__particle_positions = particle_positions
        self.__particle_radii = particle_radii

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
