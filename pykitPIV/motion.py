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
    The ``Motion`` class provides the position of particles at the next time instance, :math:`t_2 = t_1 + \Delta t`, where :math:`\Delta t`
    is the time separation for the PIV image pair :math:`\\mathbf{I} = (I_1, I_2)^{\\top}`.

    .. note::

        Particles that exit the image area as a result of their motion are removed from image :math:`I_2`.

        To ensure that motion of particles does not cause unphysical disappearance of particles near image boundaries, set an appropriately large
        image buffer, :math:`b`, when instantiating objects of ``Particle`` and ``FlowField`` class (see parameter ``size_buffer``).
        This allows new particles to enter the image area.

    **Example:**

    .. code:: python

        from pykitPIV import Particle, FlowField, Motion

        # We are going to generate 10 PIV image pairs:
        n_images = 10

        # Specify size in pixels for each image:
        image_size = (128, 512)

        # Initialize a particle object:
        particles = Particle(n_images=n_images,
                             size=image_size,
                             size_buffer=10,
                             diameters=(2, 4),
                             distances=(1, 2),
                             densities=(0.01, 0.05),
                             diameter_std=1,
                             seeding_mode='random',
                             random_seed=100)

        # Initialize a flow field object:
        flowfield = FlowField(n_images=n_images,
                              size=image_size,
                              size_buffer=10,
                              random_seed=100)

        # Generate random velocity field:
        flowfield.generate_random_velocity_field(gaussian_filters=(2, 10),
                                                 n_gaussian_filter_iter=10,
                                                 displacement=(2, 5))

        # Initialize a motion object:
        motion = Motion(particles, flowfield)


    :param particles:
        ``Particle`` class instance specifying the properties and positions of particles.
    :param flowfield:
        ``FlowField`` class instance specifying the flow field.
    :param time_separation: (optional)
        ``float`` or ``int`` specifying the time separation, :math:`\Delta t`, in seconds :math:`[s]` between two consecutive PIV images.
    :param particle_loss: (optional)
        ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
        percentage of lost particles between two consecutive PIV images. This percentage of particles from image :math:`I_1` will be randomly
        removed in image :math:`I_2`. This parameter mimics the complete loss of luminosity of particles that move perpendicular to the laser plane.
    :param particle_gain: (optional)
        ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
        percentage of lost particles between two consecutive PIV images. This percentage of particles from image :math:`I_1` will be randomly
        added in image :math:`I_2`. This parameter mimics the gain of luminosity for new particles that arrive into the laser plane.

    **Attributes:**

    - **time_separation** - (can be re-set) as per user input.
    - **particle_loss** - (can be re-set) as per user input.
    - **particle_coordinates_I1** - (read-only) ``list`` of ``tuple`` specifying the coordinates of particles in image :math:`I_1`. The first element in each tuple are the coordinates along the **image height**, and the second element are the coordinates along the **image width**.
    - **particle_coordinates_I2** - (read-only) ``list`` of ``tuple`` specifying the  coordinates of particles in image :math:`I_2`. The first element in each tuple are the coordinates along the **image height**, and the second element are the coordinates along the **image width**.
    - **updated_particle_diameters** - (read-only) ``list`` of ``numpy.ndarray`` specifying the updated particle diameters for each PIV image pair.
    - **displacement_field** - (read-only) ``numpy.ndarray`` specifying the displacement field, :math:`ds = [dx, dy]`, in the :math:`x` and :math:`y` direction. It is computed as the velocity component multiplied by time separation and has a unit of :math:`\\text{px}`. It has size :math:`(N, 2, H+2b, W+2b)`. The second index corresponds to :math:`dx` and :math:`dy` displacement, respectively.
    - **displacement_field_magnitude** - (read-only) ``numpy.ndarray`` specifying the displacement field magnitude, :math:`|ds| = \sqrt{dx^2 + dy^2}`. It has a unit of :math:`\\text{px}`. It has size :math:`(N, 1, H+2b, W+2b)`.
    """

    def __init__(self,
                 particles,
                 flowfield,
                 time_separation=1,
                 particle_loss=(0, 2),
                 particle_gain=(0, 2),
                 verbose=False,
                 random_seed=None):

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

        check_two_element_tuple(particle_gain, 'particle_gain')
        check_min_max_tuple(particle_gain, 'particle_gain')

        # Check that a velocity field is present in the FlowField class object:
        if flowfield.velocity_field is None:
            raise AttributeError("No velocity field is generated in the FlowField class object.")

        if random_seed is not None:
            if type(random_seed) != int:
                raise ValueError("Parameter `random_seed` has to be of type 'int'.")
            else:
                np.random.seed(seed=random_seed)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:
        self.__particles = particles
        self.__flowfield = flowfield
        self.__time_separation = time_separation
        self.__particle_loss = particle_loss
        self.__verbose = verbose
        self.__random_seed = random_seed

        # Initialize particle coordinates:
        self.__particle_coordinates_I1 = self.__particles.particle_coordinates
        self.__particle_coordinates_I2 = None

        # Initialize updated particle diameters:
        self.__updated_particle_diameters = None

        # Compute the displacement field:
        self.__displacement_field = np.zeros((self.__particles.n_images, 2, self.__particles.size_with_buffer[0], self.__particles.size_with_buffer[1]))
        self.__displacement_field[:, 0, :, :] = self.__flowfield.velocity_field[:, 0, :, :] * time_separation
        self.__displacement_field[:, 1, :, :] = self.__flowfield.velocity_field[:, 1, :, :] * time_separation

        self.__displacement_field_magnitude = np.sqrt(self.displacement_field[:, 0:1, :, :] ** 2 + self.displacement_field[:, 1:2, :, :] ** 2)

        # Check whether particles loss/gain will have to be modeled:
        if self.__particle_loss[1] > 0:
            self.__particles_lost = True
        else:
            self.__particles_lost = False

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

    @property
    def displacement_field(self):
        return self.__displacement_field

    @property
    def displacement_field_magnitude(self):
        return self.__displacement_field_magnitude

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

                # Re-compute the displacement field:
                self.__displacement_field = np.zeros((self.__particles.n_images, 2, self.__particles.size_with_buffer[0], self.__particles.size_with_buffer[1]))
                self.__displacement_field[:, 0, :, :] = self.__flowfield.velocity_field[:, 0, :, :] * new_time_separation
                self.__displacement_field[:, 1, :, :] = self.__flowfield.velocity_field[:, 1, :, :] * new_time_separation

                self.__displacement_field_magnitude = np.sqrt(self.displacement_field[:, 0:1, :, :] ** 2 + self.displacement_field[:, 1:2, :, :] ** 2)

    @particle_loss.setter
    def particle_loss(self, new_particle_loss):

        check_two_element_tuple(new_particle_loss, 'particle_loss')
        check_min_max_tuple(new_particle_loss, 'particle_loss')

        self.__particle_loss = new_particle_loss

        # Check whether particles loss/gain will have to be modeled:
        if self.__particle_loss[1] > 0:
            self.__particles_lost = True
        else:
            self.__particles_lost = False

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __lose_particles(self,
                         idx,
                         n_particles):

        current_loss_percentage = np.random.rand(1) * (self.__particle_loss[1] - self.__particle_loss[0]) + self.__particle_loss[0]
        current_loss_percentage = current_loss_percentage[0]

        idx_removed = np.random.choice(np.array([i for i in range(0,n_particles)]), int(current_loss_percentage*n_particles/100), replace=False)
        idx_retained = [ii for ii in range(0, n_particles) if ii not in idx_removed]

        if self.__verbose: print('Image ' + str(idx+1) + ':\t' + str(n_particles - len(idx_retained)) + ' particles lost')

        return idx_retained

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def forward_euler(self,
                      n_steps):
        """
        Advects particles with the forward Euler numerical scheme according to the formula:

        .. math::

            x_{t + \Delta t} = x_{t} + u(x_t, y_t) \cdot \Delta t

            y_{t + \Delta t} = y_{t} + v(x_t, y_t) \cdot \Delta t

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

        **Example:**

        .. code:: python

            # Advect particles with the forward Euler scheme:
            motion.forward_euler(n_steps=10)

        :param n_steps:
            ``int`` specifying the number of time steps, :math:`n`, that the numerical solver should take.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

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

            interpolate_u_component = RegularGridInterpolator(grid, self.__flowfield.velocity_field[i, 0, :, :])
            interpolate_v_component = RegularGridInterpolator(grid, self.__flowfield.velocity_field[i, 1, :, :])

            # Retrieve the old particle coordinates (at time t):
            particle_coordinates_old = np.hstack((self.__particles.particle_coordinates[i][0][:, None],
                                                 self.__particles.particle_coordinates[i][1][:, None]))

            updated_particle_diameters = self.__particles.particle_diameters[i]

            # This method assumes that the velocity field does not change during the image separation time:
            for _ in range(0,n_steps):

                # Compute the new coordinates at the next time step:
                y_coordinates_I2 = particle_coordinates_old[:,0] + interpolate_v_component(particle_coordinates_old) * __delta_t
                x_coordinates_I2 = particle_coordinates_old[:,1] + interpolate_u_component(particle_coordinates_old) * __delta_t

                particle_coordinates_old = np.hstack((y_coordinates_I2[:,None], x_coordinates_I2[:,None]))

                # Remove particles that have moved outside the image area:
                idx_removed_y, = np.where((particle_coordinates_old[:,0] < 0) | (particle_coordinates_old[:,0] > self.__particles.size_with_buffer[0]))
                idx_removed_x, = np.where((particle_coordinates_old[:,1] < 0) | (particle_coordinates_old[:,1] > self.__particles.size_with_buffer[1]))
                idx_removed = np.unique(np.concatenate((idx_removed_y, idx_removed_x)))
                idx_retained = [ii for ii in range(0,particle_coordinates_old.shape[0]) if ii not in idx_removed]

                particle_coordinates_old = particle_coordinates_old[idx_retained,:]
                updated_particle_diameters = updated_particle_diameters[idx_retained]

            if self.__particles_lost:

                idx_retained = self.__lose_particles(i, particle_coordinates_old.shape[0])

                particle_coordinates_old = particle_coordinates_old[idx_retained,:]
                updated_particle_diameters = updated_particle_diameters[idx_retained]

            particle_coordinates_I2.append((particle_coordinates_old[:,0], particle_coordinates_old[:,1]))
            self.__updated_particle_diameters.append(updated_particle_diameters)

        self.__particle_coordinates_I2 = particle_coordinates_I2

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def runge_kutta_4th(self,
                      n_steps):
        """
        Advects particles with the 4th order Runge-Kutta (RK4) numerical scheme according to the formula:

        .. math::

            x_{t + \Delta t} = x_{t} + R_x

            y_{t + \Delta t} = y_{t} + R_y

        where the residuals are computed as:

        .. math::

            R_x = \\frac{1}{6} \cdot \Big( R1_x + 2 \cdot R2_x + 2 \cdot R3_x + R4_x \Big)

            R_y = \\frac{1}{6} \cdot \Big( R1_y + 2 \cdot R2_y + 2 \cdot R3_y + R4_y \Big)

        with the coefficients:

        .. math::

            R1_x = \Delta t \cdot u(x_{t}, y_{t})

            R2_x = \Delta t \cdot u\Big( x_{t} + \\frac{1}{2} \cdot R1_x(t), y_{t} + \\frac{1}{2} \cdot R1_y(t)\Big)

            R3_x = \Delta t \cdot u\Big( x_{t} + \\frac{1}{2} \cdot R2_x(t), y_{t} + \\frac{1}{2} \cdot R2_y(t)\Big)

            R4_x = \Delta t \cdot u\Big( x_{t} + R3_x(t), y_{t} + R3_y(t)\Big)

        and:

        .. math::

            R1_y = \Delta t \cdot v(x_{t}, y_{t})

            R2_y = \Delta t \cdot v\Big( x_{t} + \\frac{1}{2} \cdot R1_x(t), y_{t} + \\frac{1}{2} \cdot R1_y(t)\Big)

            R3_y = \Delta t \cdot v\Big( x_{t} + \\frac{1}{2} \cdot R2_x(t), y_{t} + \\frac{1}{2} \cdot R2_y(t)\Big)

            R4_y = \Delta t \cdot v\Big( x_{t} + R3_x(t), y_{t} + R3_y(t)\Big)

        where :math:`u` and :math:`v` are velocity components in the :math:`x` and :math:`y` direction respectively.
        Velocity components in-between the grid points are interpolated using `scipy.interpolate.RegularGridInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html>`_.

        :math:`\Delta t` is computed as:

        .. math::

            \Delta t = T / n

        where :math:`T` is the time separation between two images specified as ``time_separation`` at class init and
        :math:`n` is the number of steps for the solver to take specified by the ``n_steps`` input parameter.
        The 4th order Runge-Kutta scheme is applied :math:`n` times from :math:`t=0` to :math:`t=T`.

        .. note::

            Note, that the central assumption for generating the kinematic relationship between two consecutive PIV images
            is that the velocity field defined by :math:`(u, v)` remains constant for the duration of time :math:`T`.

        **Example:**

        .. code:: python

            # Advect particles with the RK4 scheme:
            motion.runge_kutta_4th(n_steps=10)

        :param n_steps:
            ``int`` specifying the number of time steps, :math:`n`, that the numerical solver should take.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

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

            interpolate_u_component = RegularGridInterpolator(grid, self.__flowfield.velocity_field[i, 0, :, :], bounds_error=False, fill_value=None)
            interpolate_v_component = RegularGridInterpolator(grid, self.__flowfield.velocity_field[i, 1, :, :], bounds_error=False, fill_value=None)

            # Retrieve the old particle coordinates (at time t):
            particle_coordinates_old = np.hstack((self.__particles.particle_coordinates[i][0][:, None],
                                                 self.__particles.particle_coordinates[i][1][:, None]))

            updated_particle_diameters = self.__particles.particle_diameters[i]

            y_R1 = __delta_t * interpolate_v_component(particle_coordinates_old)
            y_R2 = __delta_t * interpolate_v_component(particle_coordinates_old)
            y_R3 = __delta_t * interpolate_v_component(particle_coordinates_old)
            y_R4 = __delta_t * interpolate_v_component(particle_coordinates_old)

            x_R1 = __delta_t * interpolate_u_component(particle_coordinates_old)
            x_R2 = __delta_t * interpolate_u_component(particle_coordinates_old)
            x_R3 = __delta_t * interpolate_u_component(particle_coordinates_old)
            x_R4 = __delta_t * interpolate_u_component(particle_coordinates_old)

            y_R = 1.0 / 6.0 * (y_R1 + 2 * y_R2 + 2 * y_R3 + y_R4)
            x_R = 1.0 / 6.0 * (x_R1 + 2 * x_R2 + 2 * x_R3 + x_R4)

            # This method assumes that the velocity field does not change during the image separation time:
            for _ in range(0,n_steps):

                x_R1_old = copy.deepcopy(x_R1)
                x_R2_old = copy.deepcopy(x_R2)
                x_R3_old = copy.deepcopy(x_R3)

                y_R1_old = copy.deepcopy(y_R1)
                y_R2_old = copy.deepcopy(y_R2)
                y_R3_old = copy.deepcopy(y_R3)

                # Compute the new coordinates at the next time step:
                y_coordinates_I2 = particle_coordinates_old[:,0] + y_R
                x_coordinates_I2 = particle_coordinates_old[:,1] + x_R

                particle_coordinates_old = np.hstack((y_coordinates_I2[:,None], x_coordinates_I2[:,None]))

                # Remove particles that have moved outside the image area:
                idx_removed_y, = np.where((particle_coordinates_old[:,0] < 0) | (particle_coordinates_old[:,0] > self.__particles.size_with_buffer[0]))
                idx_removed_x, = np.where((particle_coordinates_old[:,1] < 0) | (particle_coordinates_old[:,1] > self.__particles.size_with_buffer[1]))
                idx_removed = np.unique(np.concatenate((idx_removed_y, idx_removed_x)))
                idx_retained = [i for i in range(0,particle_coordinates_old.shape[0]) if i not in idx_removed]

                particle_coordinates_old = particle_coordinates_old[idx_retained,:]

                updated_particle_diameters = updated_particle_diameters[idx_retained]

                x_R1_old = x_R1_old[idx_retained]
                x_R2_old = x_R2_old[idx_retained]
                x_R3_old = x_R3_old[idx_retained]

                y_R1_old = y_R1_old[idx_retained]
                y_R2_old = y_R2_old[idx_retained]
                y_R3_old = y_R3_old[idx_retained]

                y_R1 = __delta_t * interpolate_v_component(particle_coordinates_old)
                y_R2 = __delta_t * interpolate_v_component(particle_coordinates_old + np.hstack((x_R1_old[:,None]/2, y_R1_old[:,None]/2)))
                y_R3 = __delta_t * interpolate_v_component(particle_coordinates_old + np.hstack((x_R2_old[:,None]/2, y_R2_old[:,None]/2)))
                y_R4 = __delta_t * interpolate_v_component(particle_coordinates_old + np.hstack((x_R3_old[:,None], y_R3_old[:,None])))

                x_R1 = __delta_t * interpolate_u_component(particle_coordinates_old)
                x_R2 = __delta_t * interpolate_u_component(particle_coordinates_old + np.hstack((x_R1_old[:,None]/2, y_R1_old[:,None]/2)))
                x_R3 = __delta_t * interpolate_u_component(particle_coordinates_old + np.hstack((x_R2_old[:,None]/2, y_R2_old[:,None]/2)))
                x_R4 = __delta_t * interpolate_u_component(particle_coordinates_old + np.hstack((x_R3_old[:,None], y_R3_old[:,None])))

                y_R = 1.0 / 6.0 * (y_R1 + 2 * y_R2 + 2 * y_R3 + y_R4)
                x_R = 1.0 / 6.0 * (x_R1 + 2 * x_R2 + 2 * x_R3 + x_R4)

            if self.__particles_lost:

                idx_retained = self.__lose_particles(i, particle_coordinates_old.shape[0])

                particle_coordinates_old = particle_coordinates_old[idx_retained,:]
                updated_particle_diameters = updated_particle_diameters[idx_retained]

            particle_coordinates_I2.append((particle_coordinates_old[:,0], particle_coordinates_old[:,1]))
            self.__updated_particle_diameters.append(updated_particle_diameters)

        self.__particle_coordinates_I2 = particle_coordinates_I2

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def compute_order_parameter(self):
        """
        Computes an order parameter as per
        `Vicsek et al. (1995) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.75.1226>`_:

        .. math::

            v_a = \\frac{1}{N} \\left| \\sum_{i=1}^N \\frac{\\vec{v}_i}{|\\vec{v}_i|} \\right|

        This is a number between 0 and 1, where a value approaching 0 indicates fully random motion of particles
        and a value approaching 1 indicates coherently moving particles (with ordered direction of velocities).
        """





        pass


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # ##################################################################################################################

    # Plotting functions

    # ##################################################################################################################

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_particle_motion(self,
                             idx,
                             s=10,
                             xlabel=None,
                             ylabel=None,
                             xticks=True,
                             yticks=True,
                             title=None,
                             color_I1='k',
                             color_I2='#ee6c4d',
                             figsize=(5,5),
                             dpi=300,
                             filename=None):
        """
        Plots the positions of particles on images :math:`I_1` and :math:`I_2`.

        **Example:**

        .. code:: python

            # Visualize the movement of particles:
            motion.plot_particle_motion(idx=0)

        :param idx:
            ``int`` specifying the index of the image to plot out of ``n_images`` number of images.
        :param s: (optional)
            ``int`` or ``float`` specifying the scatter point size.
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
        :param color_I1: (optional)
            ``str`` specifying the color for particles in image :math:`I_1`.
        :param color_I2: (optional)
            ``str`` specifying the color for particles in image :math:`I_2`.
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

        if not isinstance(idx, int):
            raise ValueError("Parameter `idx` has to be of type 'int'.")
        if idx < 0:
            raise ValueError("Parameter `idx` has to be non-negative.")

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

            plt.scatter(self.particle_coordinates_I1[idx][1], self.particle_coordinates_I1[idx][0], c=color_I1, s=s, zorder=10)
            plt.scatter(self.particle_coordinates_I2[idx][1], self.particle_coordinates_I2[idx][0], c=color_I2, s=s*2/3, zorder=20)

            plt.axis('equal')

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

            if filename is not None:
                plt.savefig(filename, dpi=dpi, bbox_inches='tight')

            return plt