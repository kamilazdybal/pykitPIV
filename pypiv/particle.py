import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import scipy
import warnings

################################################################################
################################################################################
####
####    Class: Particle
####
################################################################################
################################################################################

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
        ``tuple`` of two elements specifying the minimum and maximum particle distances to randomly sample from. [Kamila] Ask Claudio what is this.
    :param densities:
        ``tuple`` of two elements specifying the minimum and maximum particle density on an image to randomly sample from.
    :param signal_to_noise:
        ``tuple`` of two elements specifying the minimum and maximum signal-to-noise ratio for particle generation. [Kamila] I still wonder if this should rather be a property of Motion class. Maybe not, Motion can allways access this class attribute.
    :param seeding_mode:
        ``str`` specifying the seeding mode for initializing particles in the image domain. It can be one of the following: ``'random'``, ``'poisson'``.
    """

    def __init__(self,
                 n_images,
                 size=(512,512),
                 diameters=(3,6),
                 distances=(0.5,2),
                 densities=(0.05,0.1),
                 signal_to_noise=(5,20),
                 seeding_mode='random'):

        # Input parameter check:
        if type(n_images) != int:
            raise ValueError("Parameter `n_images` has to of type 'int'.")

        if n_images > 0:
            raise ValueError("Parameter `n_images` has to be positive.")

        if type(size) != tuple:
            raise ValueError("Parameter `size` has to of type 'tuple'.")

        if len(size) != 2:
            raise ValueError("Parameter `size` has to have two elements.")

        if type(diameters) != tuple:
            raise ValueError("Parameter `diameters` has to of type 'tuple'.")

        if type(distances) != tuple:
            raise ValueError("Parameter `distances` has to of type 'tuple'.")

        if type(densities) != tuple:
            raise ValueError("Parameter `densities` has to of type 'tuple'.")

        if type(signal_to_noise) != tuple:
            raise ValueError("Parameter `signal_to_noise` has to of type 'tuple'.")

        __seeding_mode = ['random', 'poisson']
        if seeding_mode not in __seeding_mode:
            raise ValueError("Parameter `seeding_mode` has to be 'random', or 'poisson'.")

        # Class init:
        self.__n_images = n_images
        self.__size = size
        self.__diameters = diameters
        self.__distances = distances
        self.__densities = densities
        self.__signal_to_noise = signal_to_noise

        # Initialize parameters for particle generation:
        self.__particle_diameters_per_image = np.random.rand(self.__n_images) * (self.__diameter[0] - self.__diameter[1]) + self.__diameter[0]
        self.__particle_distances_per_image = np.random.rand(self.__n_images) * (self.__distance[0] - self.__distance[1]) + self.__distance[0]
        self.__particle_SNR_per_image = np.random.rand(self.__n_images) * (self.__signal_to_noise[1] - self.__signal_to_noise[0]) + self.__signal_to_noise[0]

        # Compute the seeding density for each image:
        if particle_seeding_mode == 'random':

            seeded_particle_density = np.random.rand(self.__n_images) * (self.particle_density[1] - self.particle_density[0]) + self.particle_density[0]

        elif particle_seeding_mode == 'poisson':

            seeded_particle_density = np.zeros((self.n_images,))

            for i in range(0, self.n_images):
                sx = np.arange(((self.D[i] + self.L[i]) / 2), (self.size[1] - (self.D[i] + self.L[i]) / 2),
                               (self.D[i] + self.L[i]))
                sy = np.arange(((self.D[i] + self.L[i]) / 2), (self.size[0] - (self.D[i] + self.L[i]) / 2),
                               (self.D[i] + self.L[i]))
                seeded_particle_density[i] = len(sx) * len(sy) / self.size[1] / self.size[0]

        self.__seeded_particle_density = seeded_particle_density

        # Compute the total number of particles for a given particle density on each image:
        n_of_particles = self.size[1] * self.size[0] * seeded_particle_density
        self.__n_of_particles = n_of_particles

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
    def std(self):
        return self.__std

    @property
    def signal_to_noise(self):
        return self.__signal_to_noise

    # Properties computed at class init:
    @property
    def particle_density(self):
        return self.__particle_density

    @property
    def seeded_particle_density(self):
        return self.__seeded_particle_density

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def seed_particles(self, particle_seeding_mode):
