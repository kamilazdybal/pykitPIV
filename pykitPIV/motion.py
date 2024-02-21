import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import scipy
import warnings
from scipy.interpolate import RegularGridInterpolator
from pykitPIV.checks import *
from pykitPIV.particle import Particle
from pykitPIV.flowfield import FlowField

################################################################################
################################################################################
####
####    Class: Motion
####
################################################################################
################################################################################

class Motion:
    """
    Applies velocity field defined by the ``FlowField`` class instance to particles defined by the ``Particle`` class instance
    and provides the position of particles at the next time instance, :math:`t + \\Delta t`.

    :param particles:
        ``Particle`` class instance specifying the properties and positions of particles.
    :param flowfield:
        ``FlowField`` class instance specifying the flow field.
    :param time_separation: (optional)
        ``float`` or ``int`` specifying the time separation in seconds :math:`[s]` between two consecutive PIV images.
    :param n_steps: (optional)
        ``int`` specifying the number of time steps that the numerical solver should take.
    :param particle_loss: (optional)
        ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) percentage of lost particles.
        Between two consecutive image pairs, this percentage of particles will be randomly removed and replaced due to movement of particles off the laser plane.
    :param warp_images:
        ``str`` specifying the method to warp PIV images.
    """

    def __init__(self,
                 particles,
                 flowfield,
                 time_separation=1,
                 n_steps=50,
                 particle_loss=(0, 2),
                 warp_images='forward',
                 method='SINC-8'):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(particles, Particle):
            raise ValueError("Parameter `particles` has to be an instance of `Particle` class.")

        if not isinstance(flowfield, FlowField):
            raise ValueError("Parameter `flowfield` has to be an instance of `FlowField` class.")

        # Check that the image sizes match between the Particle class object and the FlowField class object:
        if particles.size_with_buffer != flowfield.size_with_buffer:
            raise ValueError(f"Inconsistent PIV image sizes between `Particle` class instance {particles.size_with_buffer} and `FlowField` class instance {flowfield.size_with_buffer}.")

        # Check that the number of images matches between the Particle class object and the FlowField class object:
        if particles.n_images != flowfield.n_images:
            raise ValueError(f"Inconsistent number of PIV image pairs between `Particle` class instance ({particles.n_images}) and `FlowField` class instance ({flowfield.n_images}).")

        if (not isinstance(time_separation, float)) and (not isinstance(time_separation, int)):
            raise ValueError("Parameter `time_separation` has to be of type `float` or `int`.")

        if time_separation <= 0:
            raise ValueError("Parameter `time_separation` has to be a non-zero, positive number.")

        if not isinstance(n_steps, int):
            raise ValueError("Parameter `n_steps` has to be of type `int`.")

        if n_steps < 1:
            raise ValueError("Parameter `n_steps` has to be at least 1.")

        check_two_element_tuple(particle_loss, 'particle_loss')
        check_min_max_tuple(particle_loss, 'particle_loss')

        __warp_images = ['forward', 'symmetric', 'two-step-forward']
        __method = ['cubic-interpolation', 'SINC-8']

        if warp_images not in __warp_images:
            raise ValueError("Parameter `warp_images` has to be 'forward', 'random-symmetric', or 'two-step-forward'.")

        if method not in __method:
            raise ValueError("Parameter `method` has to be 'cubic-interpolation', or 'SINC-8''.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Integration time step:
        self.__delta_t = time_separation / n_steps

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:
        self.__particles = particles
        self.__flowfield = flowfield
        self.__time_separation = time_separation
        self.__n_steps = n_steps
        self.__particle_loss = particle_loss

        self.__warp_images = warp_images
        self.__method = method

        # Initialize particle coordinates:
        self.__particle_coordinates_I1 = self.__particles.particle_coordinates
        self.__particle_coordinates_I2 = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Properties coming from user inputs:
    @property
    def particle_loss(self):
        return self.__particle_loss

    @property
    def warp_images(self):
        return self.__warp_images

    @property
    def n_steps(self):
        return self.__n_steps

    @property
    def method(self):
        return self.__method

    # Properties computed at class init:
    @property
    def delta_t(self):
        return self.__delta_t

    @property
    def particle_coordinates_I1(self):
        return self.__particle_coordinates_I1

    @property
    def particle_coordinates_I2(self):
        return self.__particle_coordinates_I2

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def forward_euler(self):

        particle_coordinates_I2 = []

        for i in range(0,self.__particles.n_images):

            # Build interpolants for the velocity field components:
            grid = (np.linspace(0, self.__particles.size_with_buffer[0], self.__particles.size_with_buffer[0]),
                    np.linspace(0, self.__particles.size_with_buffer[1], self.__particles.size_with_buffer[1]))

            interpolate_u_component = RegularGridInterpolator(grid, self.__flowfield.velocity_field[i][0])
            interpolate_v_component = RegularGridInterpolator(grid, self.__flowfield.velocity_field[i][1])

            # Retrieve the old particle coordinates (at time t):
            particle_coordinates_I1 = np.hstack((self.__particles.particle_coordinates[i][0][:, None],
                                                 self.__particles.particle_coordinates[i][1][:, None]))

            # This method assumes that the velocity field does not change during the image separation time:
            for i in range(0,self.n_steps):

                # Compute the new coordinates at the next time step:
                y_coordinates_I2 = particle_coordinates_I1[:,0] + interpolate_v_component(particle_coordinates_I1) * self.__delta_t * (i+1)
                x_coordinates_I2 = particle_coordinates_I1[:,1] + interpolate_u_component(particle_coordinates_I1) * self.__delta_t * (i+1)

                particle_coordinates_I1 = np.hstack((y_coordinates_I2[:,None], x_coordinates_I2[:,None]))

                # Remove particles that have moved outside of the image area:
                idx_removed_y, = np.where((particle_coordinates_I1[:,0] < 0) | (particle_coordinates_I1[:,0] > self.__particles.size_with_buffer[0]))
                idx_removed_x, = np.where((particle_coordinates_I1[:,1] < 0) | (particle_coordinates_I1[:,1] > self.__particles.size_with_buffer[1]))

                idx_removed = np.unique(np.concatenate((idx_removed_y, idx_removed_x)))

                idx_retained = [i for i in range(0,particle_coordinates_I1.shape[0]) if i not in idx_removed]

                particle_coordinates_I1 = particle_coordinates_I1[idx_retained,:]

            particle_coordinates_I2.append((y_coordinates_I2, x_coordinates_I2))

        self.__particle_coordinates_I2 = particle_coordinates_I2

