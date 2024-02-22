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
    :param particle_loss: (optional)
        ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) percentage of lost particles.
        Between two consecutive image pairs, this percentage of particles will be randomly removed and replaced due to movement of particles off the laser plane.
    """

    def __init__(self,
                 particles,
                 flowfield,
                 time_separation=1,
                 particle_loss=(0, 2)):

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

        check_two_element_tuple(particle_loss, 'particle_loss')
        check_min_max_tuple(particle_loss, 'particle_loss')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:
        self.__particles = particles
        self.__flowfield = flowfield
        self.__time_separation = time_separation
        self.__particle_loss = particle_loss

        # Initialize particle coordinates:
        self.__particle_coordinates_I1 = self.__particles.particle_coordinates
        self.__particle_coordinates_I2 = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Properties coming from user inputs:
    @property
    def time_separation(self):
        return self.__time_separation

    @property
    def particle_loss(self):
        return self.__particle_loss

    # Properties computed at class init:
    @property
    def particle_coordinates_I1(self):
        return self.__particle_coordinates_I1

    @property
    def particle_coordinates_I2(self):
        return self.__particle_coordinates_I2

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def forward_euler(self,
                      n_steps):
        """
        Advects particles with a forward Euler numerical scheme according to the formula:

        .. math::

            x_{t + \Delta t} = x_{t} + u \cdot \Delta t

            y_{t + \Delta t} = y_{t} + v \cdot \Delta t

        where :math:`u` and :math:`v` are velocity components in the :math:`x` and :math:`y` direction respectively.
        Velocity components in-between the grid points are interpolated using ``scipy.interpolate.RegularGridInterpolator``.

        :math:`\Delta t` is computed as:

        .. math::

            \Delta t = T / n

        where :math:`T` is the time separation between two images specified as ``time_separation`` at class init and
        :math:`n` is the number of steps for the solver to take specified by the ``n_steps`` input parameter.
        The Euler scheme is applied :math:`n` times from :math:`t=0` to :math:`t=T`.

        :param n_steps:
            ``int`` specifying the number of time steps, :math:`n`, that the numerical solver should take.
        """

        if not isinstance(n_steps, int):
            raise ValueError("Parameter `n_steps` has to be of type `int`.")

        if n_steps < 1:
            raise ValueError("Parameter `n_steps` has to be at least 1.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Integration time step:
        __delta_t = self.time_separation / n_steps

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
            for i in range(0,n_steps):

                # Compute the new coordinates at the next time step:
                y_coordinates_I2 = particle_coordinates_I1[:,0] + interpolate_v_component(particle_coordinates_I1) * __delta_t * (i+1)
                x_coordinates_I2 = particle_coordinates_I1[:,1] + interpolate_u_component(particle_coordinates_I1) * __delta_t * (i+1)

                particle_coordinates_I1 = np.hstack((y_coordinates_I2[:,None], x_coordinates_I2[:,None]))

                # Remove particles that have moved outside of the image area:
                idx_removed_y, = np.where((particle_coordinates_I1[:,0] < 0) | (particle_coordinates_I1[:,0] > self.__particles.size_with_buffer[0]))
                idx_removed_x, = np.where((particle_coordinates_I1[:,1] < 0) | (particle_coordinates_I1[:,1] > self.__particles.size_with_buffer[1]))

                idx_removed = np.unique(np.concatenate((idx_removed_y, idx_removed_x)))

                idx_retained = [i for i in range(0,particle_coordinates_I1.shape[0]) if i not in idx_removed]

                particle_coordinates_I1 = particle_coordinates_I1[idx_retained,:]

            particle_coordinates_I2.append((y_coordinates_I2, x_coordinates_I2))

        self.__particle_coordinates_I2 = particle_coordinates_I2

