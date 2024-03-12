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
    Applies velocity field defined by the ``FlowField`` class instance to particles defined by the ``Particle`` class instance.
    The ``Motion`` class provides the position of particles at the next time instance, :math:`t + T`, where :math:`T`
    is the time separation for the PIV image pair :math:`\\mathbf{I} = (I_1, I_2)^{\\top}`.

    .. note::

        Particles that exit the image area as a result of their motion are removed from image :math:`I_2`.
        To ensure that motion of particles does not cause unphysical removal of particles near image boundaries, set an appropriately large
        image buffer when instantiating objects of ``Particle`` and ``FlowField`` class (see parameter ``size_buffer``).

    **Example:**

    .. code:: python

        from pykitPIV import Particle, Flowfield, Motion

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

        # Initialize a flow field object:
        flowfield = FlowField(n_images=n_images,
                              size=image_size,
                              size_buffer=10,
                              flow_mode='random',
                              gaussian_filters=(8,10),
                              n_gaussian_filter_iter=10,
                              sin_period=(30,300),
                              displacement=(0,10),
                              random_seed=100)

        # Initialize a motion object:
        motion = Motion(particles, flowfield)


    :param particles:
        ``Particle`` class instance specifying the properties and positions of particles.
    :param flowfield:
        ``FlowField`` class instance specifying the flow field.
    :param time_separation: (optional)
        ``float`` or ``int`` specifying the time separation, :math:`T`, in seconds :math:`[s]` between two consecutive PIV images.
    :param particle_loss: (optional)
        ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
        percentage of lost particles between two consecutive PIV images. This percentage of particles from image :math:`I_1` will be randomly
        removed and replaced in image :math:`I_2`. This parameter mimics the complete loss of luminosity of particles that move off the laser plane
        and a gain of luminosity for brand new particles that arrive into the laser plane.

    **Attributes:**

    - **time_separation** - (can be re-set) as per user input.
    - **particle_loss** - (read-only) as per user input.
    - **particle_coordinates_I1** - (read-only) coordinates of particles in image :math:`I_1`.
    - **particle_coordinates_I2** - (read-only) coordinates of particles in image :math:`I_2`.
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

        # Initialize updated particle diameters:
        self.__updated_particle_diameters = None

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

    @property
    def updated_particle_diameters(self):
        return self.__updated_particle_diameters

    # Setters:
    @time_separation.setter
    def time_separation(self, new_time_separation):
        if (not isinstance(new_time_separation, float)) and (not isinstance(new_time_separation, int)):
            raise ValueError("Parameter `time_separation` has to be of type `float` or `int`.")
        else:
            if new_time_separation <= 0:
                raise ValueError("Parameter `time_separation` has to be a non-zero, positive number.")
            else:
                self.__time_separation = new_time_separation

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def forward_euler(self,
                      n_steps):
        """
        Advects particles with the forward Euler numerical scheme according to the formula:

        .. math::

            x_{t + \Delta t} = x_{t} + u \cdot \Delta t

            y_{t + \Delta t} = y_{t} + v \cdot \Delta t

        where :math:`u` and :math:`v` are velocity components in the :math:`x` and :math:`y` direction respectively.
        Velocity components in-between the grid points are interpolated using `scipy.interpolate.RegularGridInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html>`_.

        :math:`\Delta t` is computed as:

        .. math::

            \Delta t = T / n

        where :math:`T` is the time separation between two images specified as ``time_separation`` at class init and
        :math:`n` is the number of steps for the solver to take specified by the ``n_steps`` input parameter.
        The Euler scheme is applied :math:`n` times from :math:`t=0` to :math:`t=T`.

        .. note::

            Note, that the central assumption for generating the kinematic relationship between two consecutive PIV images
            is that the velocity field defined by :math:`(u, v)` remains constant for the duration of time :math:`T`.

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

        self.__updated_particle_diameters = []

        for i in range(0,self.__particles.n_images):

            # Build interpolants for the velocity field components:
            grid = (np.linspace(0, self.__particles.size_with_buffer[0], self.__particles.size_with_buffer[0]),
                    np.linspace(0, self.__particles.size_with_buffer[1], self.__particles.size_with_buffer[1]))

            interpolate_u_component = RegularGridInterpolator(grid, self.__flowfield.velocity_field[i][0])
            interpolate_v_component = RegularGridInterpolator(grid, self.__flowfield.velocity_field[i][1])

            # Retrieve the old particle coordinates (at time t):
            particle_coordinates_old = np.hstack((self.__particles.particle_coordinates[i][0][:, None],
                                                 self.__particles.particle_coordinates[i][1][:, None]))

            updated_particle_diameters = self.__particles.particle_diameters[i]

            # This method assumes that the velocity field does not change during the image separation time:
            for i in range(0,n_steps):

                # Compute the new coordinates at the next time step:
                y_coordinates_I2 = particle_coordinates_old[:,0] + interpolate_v_component(particle_coordinates_old) * __delta_t * (i+1)
                x_coordinates_I2 = particle_coordinates_old[:,1] + interpolate_u_component(particle_coordinates_old) * __delta_t * (i+1)

                particle_coordinates_old = np.hstack((y_coordinates_I2[:,None], x_coordinates_I2[:,None]))

                # Remove particles that have moved outside the image area:
                idx_removed_y, = np.where((particle_coordinates_old[:,0] < 0) | (particle_coordinates_old[:,0] > self.__particles.size_with_buffer[0]))
                idx_removed_x, = np.where((particle_coordinates_old[:,1] < 0) | (particle_coordinates_old[:,1] > self.__particles.size_with_buffer[1]))
                idx_removed = np.unique(np.concatenate((idx_removed_y, idx_removed_x)))
                idx_retained = [i for i in range(0,particle_coordinates_old.shape[0]) if i not in idx_removed]

                particle_coordinates_old = particle_coordinates_old[idx_retained,:]

                updated_particle_diameters = updated_particle_diameters[idx_retained]

            particle_coordinates_I2.append((particle_coordinates_old[:,0], particle_coordinates_old[:,1]))
            self.__updated_particle_diameters.append(updated_particle_diameters)

        self.__particle_coordinates_I2 = particle_coordinates_I2

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def runge_kutta_4th(self,
                      n_steps):
        """
        Advects particles with the 4th order Runge-Kutta numerical scheme according to the formula:


        .. note::

            Note, that the central assumption for generating the kinematic relationship between two consecutive PIV images
            is that the velocity field defined by :math:`(u, v)` remains constant for the duration of time :math:`T`.

        """

        pass

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_particle_motion(self,
                             idx,
                             xlabel=None,
                             ylabel=None,
                             title=None,
                             figsize=(5,5),
                             dpi=300,
                             filename=None):
        """
        Plots the positions of particles on images :math:`I_1` and :math:`I_2`.

        :param idx:
            ``int`` specifying the index of the image to plot out of ``n_images`` number of images.
        :param xlabel: (optional)
            ``str`` specifying :math:`x`-label.
        :param ylabel: (optional)
            ``str`` specifying :math:`y`-label.
        :param title: (optional)
            ``str`` specifying figure title.
        :param figsize: (optional)
            ``tuple`` of two numerical elements specifying the figure size as per ``matplotlib.pyplot``.
        :param dpi: (optional)
            ``int`` specifying the dpi for the image.
        :param filename: (optional)
            ``str`` specifying the path and filename to save an image. If set to ``None``, the image will not be saved.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(idx, int):
            raise ValueError("Parameter `idx` has to be of type 'int'.")
        if idx < 0:
            raise ValueError("Parameter `idx` has to be non-negative.")

        if (xlabel is not None) and (not isinstance(xlabel, str)):
            raise ValueError("Parameter `xlabel` has to be of type 'str'.")

        if (ylabel is not None) and (not isinstance(ylabel, str)):
            raise ValueError("Parameter `ylabel` has to be of type 'str'.")

        if (title is not None) and (not isinstance(title, str)):
            raise ValueError("Parameter `title` has to be of type 'str'.")

        check_two_element_tuple(figsize, 'figsize')

        if not isinstance(dpi, int):
            raise ValueError("Parameter `dpi` has to be of type 'int'.")

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if self.particle_coordinates_I2 is None:

            print('Note: Particles have not been advected yet!\n\n')

        else:

            fig = plt.figure(figsize=figsize)

            plt.scatter(self.particle_coordinates_I1[idx][1], self.particle_coordinates_I1[idx][0], c='k', s=2, zorder=10)
            plt.scatter(self.particle_coordinates_I2[idx][1], self.particle_coordinates_I2[idx][0], c='#ee6c4d', s=1.5, zorder=20)

            if xlabel is not None:
                plt.xlabel(xlabel)

            if ylabel is not None:
                plt.ylabel(ylabel)

            if title is not None:
                plt.title(title)

            if filename is not None:
                plt.savefig(filename, dpi=dpi, bbox_inches='tight')

            return plt