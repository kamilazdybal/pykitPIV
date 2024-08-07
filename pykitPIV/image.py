import h5py
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from pykitPIV.checks import *
from pykitPIV.flowfield import FlowField
from pykitPIV.motion import Motion
from pykitPIV.particle import Particle

########################################################################################################################
########################################################################################################################
####
####    Class: Image
####
########################################################################################################################
########################################################################################################################

class Image:
    """
    Stores and plots synthetic PIV images and/or the associated flow fields at any stage of particle generation
    and movement.

    **Example:**

    .. code:: python

        from pykitPIV import Image

        # Initialize an image object:
        image = Image(random_seed=100)

    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``. If specified, all image generation is reproducible.

    **Attributes:**

    - **random_seed** - (read-only) as per user input.
    - **images_I1** - (read-only) ``list`` of ``numpy.ndarray``, where each element is the current version of a PIV image :math:`I_1` of a given size. Only available after ``Image.add_particles()`` has been called.
    - **images_I2** - (read-only) ``list`` of ``numpy.ndarray``, where each element is the current version of a PIV image :math:`I_2` of a given size. Only available after ``Image.add_motion()`` has been called.
    - **images_I1_no_buffer** - (read-only) ``list`` of ``numpy.ndarray``, where each element is the current version of a PIV image :math:`I_1` of a given size, without the buffer. Only available after ``Image.add_particles()`` and ``Image.remove_buffer()`` have been called.
    - **images_I2_no_buffer** - (read-only) ``list`` of ``numpy.ndarray``, where each element is the current version of a PIV image :math:`I_2` of a given size, without the buffer. Only available after ``Image.add_motion()`` and ``Image.remove_buffer()`` have been called.
    - **targets** - (read-only) ``list`` of ``tuple``, where each element contains the velocity field components, :math:`u` and :math:`v`, as ``numpy.ndarray``. Only available after ``Image.add_flowfield()`` has been called.
    - **targets_no_buffer** - (read-only) ``list`` of ``tuple``, where each element contains the velocity field components, :math:`u` and :math:`v`, as ``numpy.ndarray``, without the buffer. Only available after ``Image.add_flowfield()`` and ``Image.remove_buffer()`` have been called.
    - **exposures_per_image** - (read-only) ``numpy.ndarray`` specifying the template for the light exposure for each image. Only available after ``Image.add_reflected_light`` has been called.
    - **maximum_intensity** - (read-only) ``int`` specifying the maximum intensity that was used when adding reflected light to the image. Only available after ``Image.add_reflected_light`` has been called.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __init__(self,
                 random_seed=None):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if random_seed is not None:
            if type(random_seed) != int:
                raise ValueError("Parameter `random_seed` has to be of type 'int'.")
            else:
                np.random.seed(seed=random_seed)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:
        self.__random_seed = random_seed

        # Initialize images:
        self.__images_I1 = None
        self.__images_I2 = None

        # Initialize images without buffer:
        self.__images_I1_no_buffer = None
        self.__images_I2_no_buffer = None

        # Initialize flow targets:
        self.__targets = None

        # Initialize flow targets without buffer:
        self.__targets_no_buffer = None

        # Initialize particles:
        self.__particles = None

        # Initialize flow field:
        self.__flowfield = None

        # Initialize motion:
        self.__motion = None

        # Initialize exposures per image:
        self.__exposures_per_image = None

        # Initialize maximum intensity:
        self.__maximum_intensity = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Properties coming from user inputs:
    @property
    def random_seed(self):
        return self.__random_seed

    # Properties computed at class init:
    @property
    def images_I1(self):
        return self.__images_I1

    @property
    def images_I2(self):
        return self.__images_I2

    @property
    def images_I1_no_buffer(self):
        return self.__images_I1_no_buffer

    @property
    def images_I2_no_buffer(self):
        return self.__images_I2_no_buffer

    @property
    def targets(self):
        return self.__targets

    @property
    def targets_no_buffer(self):
        return self.__targets_no_buffer

    @property
    def exposures_per_image(self):
        return self.__exposures_per_image

    @property
    def maximum_intensity(self):
        return self.__maximum_intensity

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_particles(self,
                      particles):
        """
        Adds particles to the image. Particles should be defined using the ``Particle`` class.

        Calling this function populates the private ``image.__particles`` attribute and the ``image.images_I1`` attribute.

        **Example:**

        .. code:: python

            from pykitPIV import Particle, Image

            # Initialize a particle object:
            particles = Particle(1,
                                 size=(128,512),
                                 size_buffer=10,
                                 random_seed=100)

            # Initialize an image object:
            image = Image(random_seed=100)

            # Add particles to an image:
            image.add_particles(particles)

        :param particles:
            ``Particle`` class instance specifying the properties and positions of particles.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(particles, Particle):
            raise ValueError("Parameter `particles` has to be an instance of `Particle` class.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__particles = particles
        self.__images_I1 = self.__particles.particle_positions

        print('Particles added to the image.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_flowfield(self,
                      flowfield):
        """
        Adds the flow field to the image. The flow field should be defined using the ``FlowField`` class.

        Calling this function populates the private ``image.__flowfield`` attribute and the ``image.targets`` attribute.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField, Image

            # Initialize a flow field object:
            flowfield = FlowField(1,
                                  size=(128,512),
                                  size_buffer=10,
                                  random_seed=100)

            # Initialize an image object:
            image = Image(random_seed=100)

            # Add flow field to an image:
            image.add_flowfield(flowfield)

        :param flowfield:
            ``FlowField`` class instance specifying the flow field.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(flowfield, FlowField):
            raise ValueError("Parameter `flowfield` has to be an instance of `FlowField` class.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__flowfield = flowfield
        self.__targets = flowfield.velocity_field

        print('Velocity field added to the image.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_motion(self,
                   motion):
        """
        Adds particle movement to the image. The movement should be defined using the ``Motion`` class.

        Calling this function populates the private ``image.__motion`` attribute.

        **Example:**

        .. code:: python

            from pykitPIV import Particle, FlowField, Motion, Image

            # Initialize a particle object:
            particles = Particle(1,
                                 size=(128,512),
                                 size_buffer=10,
                                 random_seed=100)

            # Initialize a flow field object:
            flowfield = FlowField(1,
                                  size=(128,512),
                                  size_buffer=10,
                                  random_seed=100)

            # Initialize a motion object:
            motion = Motion(particles, flowfield)

            # Initialize an image object:
            image = Image(random_seed=100)

            # Add motion to an image:
            image.add_motion(motion)

        :param motion:
            ``Motion`` class instance specifying the movement of particles from one instance in time to the next.
            In general, the movement is defined by the ``FlowField`` class applied to particles defined by the ``Particle`` class.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(motion, Motion):
            raise ValueError("Parameter `motion` has to be an instance of `Motion` class.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__motion = motion

        print('Particle movement added to the image.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def compute_light_intensity_at_pixel(self,
                                         peak_intensity,
                                         particle_diameter,
                                         coordinate_height,
                                         coordinate_width,
                                         alpha=1/8):
        """
        Computes the intensity of light reflected from a particle at a requested pixel at position relative
        to the particle centroid.

        The reflected light follows a Gaussian distribution, i.e., the light intensity value, :math:`i_p`, at the
        request pixel, :math:`p`, is computed as:

        .. math::

            i_p =  i_{\\text{peak}} \\cdot \\exp \Big(- \\frac{h_p^2 + w_p^2}{\\alpha \\cdot d_p^2} \Big)

        where:

        - :math:`i_{\\text{peak}}` is the peak intensity applied at the particle centroid.
        - :math:`h_p` is the pixel coordinate in the image height direction relative to the particle centroid.
        - :math:`w_p` is the pixel coordinate in the image width direction relative to the particle centroid.
        - :math:`\\alpha` is a custom multiplier, :math:`\\alpha`. The default value is :math:`\\alpha = 1/8`.
        - :math:`d_p` is the particle diameter.

        :param peak_intensity:
            ``float`` specifying the peak intensity, :math:`i_{\\text{peak}}`, to apply at the particle centroid.
        :param particle_diameter:
            ``float`` specifying the particle diameter :math:`d_p`.
        :param coordinate_height:
            ``float`` specifying the pixel coordinate in the image height direction, :math:`h_p`, relative to the particle centroid.
        :param coordinate_width:
            ``float`` specifying the pixel coordinate in the image width direction, :math:`w_p`, relative to the particle centroid.
        :param alpha: (optional):
            ``float`` specifying the custom multiplier, :math:`\\alpha`, for the squared particle radius. The default value is :math:`1/8` as per `Rabault et al. (2017) <https://iopscience.iop.org/article/10.1088/1361-6501/aa8b87/meta>`_ and `Manickathan et al. (2022) <https://iopscience.iop.org/article/10.1088/1361-6501/ac8fae>`_.

        :return:
            - **pixel_value** - ``float`` specifying the light intensity value at the requested pixel.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not (isinstance(peak_intensity, float)):
            raise ValueError("Parameter `peak_intensity` has to be of type `float`.")

        if not (isinstance(particle_diameter, float)):
            raise ValueError("Parameter `particle_diameter` has to be of type `float`.")

        if not (isinstance(coordinate_height, float)):
            raise ValueError("Parameter `coordinate_height` has to be of type `float`.")

        if not (isinstance(coordinate_width, float)):
            raise ValueError("Parameter `coordinate_width` has to be of type `float`.")

        if not (isinstance(alpha, float)):
            raise ValueError("Parameter `alpha` has to be of type `float`.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        pixel_value = peak_intensity * np.exp(-(coordinate_height**2 + coordinate_width**2) / (alpha * particle_diameter**2))

        return pixel_value

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_reflected_light(self,
                            exposures=(0.5,0.9),
                            maximum_intensity=2**16-1,
                            laser_beam_thickness=2,
                            laser_over_exposure=1,
                            laser_beam_shape=0.85,
                            alpha=1/8):
        """
        Creates particle sizes and adds laser light reflected from particles.

        The reflected light follows a Gaussian distribution and is computed using
        the ``Image.compute_light_intensity_at_pixel()`` method.

        **Example:**

        .. code:: python

            from pykitPIV import Particle, Image

            # Initialize a particle object:
            particles = Particle(1,
                                 size=(128,512),
                                 size_buffer=10)

            # Initialize an image object:
            image = Image(random_seed=100)

            # Add particles to an image:
            image.add_particles(particles)

            # Add reflected light to an image:
            image.add_reflected_light(exposures=(0.5, 0.9),
                                      maximum_intensity=2**16-1,
                                      laser_beam_thickness=1,
                                      laser_over_exposure=1,
                                      laser_beam_shape=0.95,
                                      alpha=1/20)

        :param exposures: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) light exposure.
        :param maximum_intensity: (optional)
            ``int`` specifying the maximum light intensity.
        :param laser_beam_thickness: (optional)
            ``int`` or ``float`` specifying the thickness of the laser beam. With a small thickness, particles that are e
            ven slightly off-plane will appear darker. Note, that the thicker the laser plane, the larger the distance
            that the particle can travel away from the laser plane to lose its luminosity.
        :param laser_over_exposure: (optional)
            ``int`` or ``float`` specifying the overexposure of the laser beam.
        :param laser_beam_shape: (optional)
            ``int`` or ``float`` specifying the spread of the Gaussian shape of the laser beam. The larger this number
            is, the wider the Gaussian light distribution from the laser and more particles will be illuminated.
        :param alpha: (optional):
            ``float`` specifying the custom multiplier, :math:`\\alpha`, for the squared particle radius as per the ``Particle.compute_light_intensity_at_pixel()`` method.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        check_two_element_tuple(exposures, 'exposures')
        check_min_max_tuple(exposures, 'exposures')

        if not(isinstance(maximum_intensity, int)):
            raise ValueError("Parameter `maximum_intensity` has to be of type `int`.")

        if not(isinstance(laser_beam_thickness, int)) and not(isinstance(laser_beam_thickness, float)):
            raise ValueError("Parameter `laser_beam_thickness` has to be of type `int` or `float`.")

        if not(isinstance(laser_over_exposure, int)) and not(isinstance(laser_over_exposure, float)):
            raise ValueError("Parameter `laser_over_exposure` has to be of type `int` or `float`.")

        if not(isinstance(laser_beam_shape, int)) and not(isinstance(laser_beam_shape, float)):
            raise ValueError("Parameter `laser_beam_shape` has to be of type `int` or `float`.")

        if not(isinstance(alpha, float)):
            raise ValueError("Parameter `alpha` has to be of type `float`.")

        if self.random_seed is not None:
            np.random.seed(seed=self.random_seed)

        if self.__particles is None:
            raise NameError("Particles have not been added to the image yet! Use the `Image.add_particles()` method first.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Save the maximum intensity:
        self.__maximum_intensity = maximum_intensity

        # Randomize image exposure:
        self.__exposures_per_image = np.random.rand(self.__particles.n_images) * (exposures[1] - exposures[0]) + exposures[0]

        # Function for adding Gaussian light distribution to image I1 or image I2:
        def __gaussian_light(idx,
                             particle_height_coordinate,
                             particle_width_coordinate,
                             image_instance):

            # Note, that this number can be different between image I1 and I2 due to removal of particles from image area.
            # That's why we do not set number_of_particles = self.__particles.n_of_particles[idx] -- this would only work for I1.
            number_of_particles = particle_height_coordinate.shape[0]

            # Initialize an empty image:
            particles_with_gaussian_light = np.zeros((self.__particles.size_with_buffer[0], self.__particles.size_with_buffer[1]))

            # Establish the peak intensity for each particle depending on its position with respect to the laser beam plane:
            particle_positions_off_laser_plane = laser_beam_thickness * np.random.rand(number_of_particles) - laser_beam_thickness / 2
            particle_position_relative_to_laser_centerline = np.abs(particle_positions_off_laser_plane) / (laser_beam_thickness / 2)
            particle_peak_intensities = self.exposures_per_image[idx] * maximum_intensity * np.exp(-0.5 * (particle_position_relative_to_laser_centerline ** 2 / laser_beam_shape ** 2))

            # Add Gaussian blur to each particle location that mimics the light reflect from a particle of a given size:
            for p in range(0, number_of_particles):

                if image_instance == 1:
                    particle_diameter_on_image = self.__particles.particle_diameters[idx][p]
                elif image_instance == 2:
                    particle_diameter_on_image = self.__motion.updated_particle_diameters[idx][p]

                px_c_height = np.floor(particle_height_coordinate[p]).astype(int)
                px_c_width = np.floor(particle_width_coordinate[p]).astype(int)
                ceil_of_particle_radius = np.ceil(self.__particles.particle_diameters[idx][p] / 2).astype(int)

                # We only apply the Gaussian blur in the square neighborhood of the particle center:
                for h in range(px_c_height - ceil_of_particle_radius, px_c_height + ceil_of_particle_radius + 1):
                    for w in range(px_c_width - ceil_of_particle_radius, px_c_width + ceil_of_particle_radius + 1):

                        # Only change the value of pixels that are within the image area:
                        if (h >= 0 and h < self.__particles.size_with_buffer[0]) and (w >= 0 and w < self.__particles.size_with_buffer[1]):
                            # 0.5 is added because we are computing the distance from particle center to the center of each considered pixel:
                            coordinate_height = particle_height_coordinate[p] - (h + 0.5)
                            coordinate_width = particle_width_coordinate[p] - (w + 0.5)

                            particles_with_gaussian_light[h, w] = particles_with_gaussian_light[h, w] + self.compute_light_intensity_at_pixel(particle_peak_intensities[p],
                                                                                                                                              particle_diameter_on_image,
                                                                                                                                              coordinate_height,
                                                                                                                                              coordinate_width,
                                                                                                                                              alpha=alpha)

            return particles_with_gaussian_light

        def __clip_intensities(image, maximum_intensity):
            return np.clip(image, a_min=0, a_max=maximum_intensity)

        # Add light to image I1:
        if self.__particles is not None:

            images_I1 = []

            for i in range(0,self.__particles.n_images):

                particles_with_gaussian_light = __gaussian_light(i,
                                                                 self.__particles.particle_coordinates[i][0],
                                                                 self.__particles.particle_coordinates[i][1],
                                                                 image_instance=1)

                images_I1.append(__clip_intensities(particles_with_gaussian_light, maximum_intensity))

                self.__images_I1 = images_I1

            print('Reflected light added to images I1.')

        # Add light to image I2:
        if self.__motion is not None:

            images_I2 = []

            for i in range(0,self.__particles.n_images):

                particles_with_gaussian_light = __gaussian_light(i,
                                                                 self.__motion.particle_coordinates_I2[i][0],
                                                                 self.__motion.particle_coordinates_I2[i][1],
                                                                 image_instance=2)

                images_I2.append(__clip_intensities(particles_with_gaussian_light, maximum_intensity))

                self.__images_I2 = images_I2

            print('Reflected light added to images I2.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def remove_buffers(self):
        """
        Removes buffers from the generated PIV image pairs and from the associated targets (velocity fields).
        Executing this function populates the class attributes ``Image.images_I1_no_buffer``,
        ``Image.images_I2_no_buffer`` with copies of ``Image.images_I1``, ``Image.images_I2`` but with the buffer
        removed, and it also populates the class attribute  ``Image.targets_no_buffer``,
        with a copy of ``Image.targets`` but with the buffer removed.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if self.__particles.size_buffer > 0:

            # Remove buffer from PIV image pairs:
            if self.images_I1 is None:

                raise ValueError("Images have not been initialized yet!")

            else:

                self.__images_I1_no_buffer = []

                for i in range(0,self.__particles.n_images):

                    self.__images_I1_no_buffer.append(self.images_I1[i][self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer])

                print('Buffers removed from images I1.')

            if self.images_I2 is None:

                pass

            else:

                self.__images_I2_no_buffer = []

                for i in range(0,self.__particles.n_images):

                    self.__images_I2_no_buffer.append(self.images_I2[i][self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer])

                print('Buffers removed from images I2.')

            # Remove buffer from PIV image targets (velocity field):
            if self.__flowfield is not None:

                self.__targets_no_buffer = []

                for i in range(0,self.__particles.n_images):

                    target_tuple = (self.__flowfield.velocity_field[i][0][self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer],
                                    self.__flowfield.velocity_field[i][1][self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer])

                    self.__targets_no_buffer.append(target_tuple)

                print('Buffers removed from the velocity field.')

        else:

            print('Images do not have a buffer to remove!')

            self.__images_I1_no_buffer = self.images_I1
            self.__images_I2_no_buffer = self.images_I2

            if self.__flowfield is not None:

                self.__targets_no_buffer = self.targets

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def measure_counts(self, image):
        """
        Measures the number of light intensity counts in a given image.

        :param image:
            ``numpy.ndarray`` specifying the single PIV image.

        :return:
            - **counts_dictionary** - ``dict`` specifying the measured counts. Keys are pixel intensities and values are the number of occurrence of the given intensity in the image.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(image, np.ndarray):
            raise ValueError("Parameter `image` has to be of type `numpy.ndarray`.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        __flat_image = list(image.ravel())

        counts_dictionary = {}

        unique_counts = list(np.unique(__flat_image))

        for value in unique_counts:
            counts_dictionary[value] = __flat_image.count(value)

        return counts_dictionary

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def image_pairs_to_tensor(self):
        """
        Prepares a 4-dimensional array with dimensions: ``(n_images, 2, image_height, image_width)`` that stores
        the image pairs.

        The second dimension represents the image pair, :math:`I_1` and :math:`I_2`.

        :return:
            - **images_tensor** - ``numpy.ndarray`` specifying the PIV image pairs tensor.
        """

        images_tensor = np.zeros((self.__particles.n_images, 2, self.__particles.size[0], self.__particles.size[1]))

        for i in range(0, self.__particles.n_images):

            images_tensor[i, 0, :, :] = self.images_I1_no_buffer[i]
            images_tensor[i, 1, :, :] = self.images_I2_no_buffer[i]

        return images_tensor

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def targets_to_tensor(self):
        """
        Prepares a 4-dimensional array with dimensions: ``(n_images, 2, image_height, image_width)`` that stores
        the flow targets.

        The second dimension represents the velocity field components, :math:`u` and :math:`v`.

        :return:
            - **targets_tensor** - ``numpy.ndarray`` specifying the flow targets tensor.
        """

        targets_tensor = np.zeros((self.__particles.n_images, 2, self.__particles.size[0], self.__particles.size[1]))

        for i in range(0, self.__particles.n_images):

            targets_tensor[i, 0, :, :] = self.targets_no_buffer[i][0]
            targets_tensor[i, 1, :, :] = self.targets_no_buffer[i][1]

        return targets_tensor

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def save_to_h5(self,
                   tensors_dictionary,
                   save_individually=False,
                   filename=None,
                   verbose=False):
        """
        Saves the image pairs tensor and the associated flow targets tensor to ``.h5`` data format.

        :param tensors_dictionary:
            ``dict`` specifying the tensors to save.
        :param save_individually: (optional)
            ``bool`` specifying if each image pair and the associated targets should be saved to a separate file.
            It is recommended to save individually for large datasets that will be uploaded by **PyTorch**, since at
            any iteration of a machine learning algorithm, only a small batch of samples is uploaded to memory.
        :param filename: (optional)
            ``str`` specifying the path and filename to save the ``.h5`` data. Note that ``'-pair-#'`` will be added
            automatically to your filename for each saved image pair.
            If set to ``None``, a default name ``'PIV-dataset-pair-#.h5'`` will be used.
        :param verbose: (optional)
            ``bool`` for printing verbose details.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(tensors_dictionary, dict):
            raise ValueError("Parameter `tensors_dict` has to be of type 'dict'.")

        if not isinstance(save_individually, bool):
            raise ValueError("Parameter `save_individually` has to be of type 'bool'.")

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        if filename is None:
            filename = 'PIV-dataset.h5'

        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be of type 'bool'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if save_individually:

            dictionary_keys = list(tensors_dictionary.keys())

            n_images, _, _ ,_ = tensors_dictionary[dictionary_keys[0]].shape

            for i in range(0,n_images):

                individual_filename = filename.split('.')[0] + '-sample-' + str(i) + '.h5'

                with h5py.File(individual_filename, 'w', libver='latest') as f:
                    for name_tag, data_item in tensors_dictionary.items():
                        dataset = f.create_dataset(name_tag, data=data_item[i], compression='gzip', compression_opts=9)
                    f.close()

                if verbose: print(individual_filename + '\tsaved.')

        else:

            with h5py.File(filename, 'w', libver='latest') as f:
                for name_tag, data_item in tensors_dictionary.items():
                    dataset = f.create_dataset(name_tag, data=data_item, compression='gzip', compression_opts=9)
                f.close()

            if verbose: print('Dataset saved.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def upload_from_h5(self,
                       filename=None):
        """
        Upload image pairs tensor and the associated flow targets tensor from ``.h5`` data format.

        :param filename: (optional)
            ``str`` specifying the path and filename to save the ``.h5`` data. Note that ``'-pair-#'`` will be added
            automatically to your filename for each saved image pair.
            If set to ``None``, a default name ``'PIV-dataset-pair-#.h5'`` will be used.

        :return:
            - **tensors_dictionary** - ``dict`` specifying the dataset tensors.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        if filename is None:
            filename = 'PIV-dataset.h5'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        tensors_dictionary = {}

        f = h5py.File(filename, 'r')

        for key in f.keys():
            tensors_dictionary[key] = np.array(f[key])

        f.close()

        return tensors_dictionary

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # ##################################################################################################################

    # Plotting functions

    # ##################################################################################################################

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot(self,
             idx,
             instance=1,
             with_buffer=False,
             xlabel=None,
             ylabel=None,
             title=None,
             cmap='Greys_r',
             figsize=(5,5),
             dpi=300,
             filename=None):
        """
        Plots a single, static PIV image, :math:`I_1` or :math:`I_2`.

        :param idx:
            ``int`` specifying the index of the image to plot out of ``n_images`` number of images.
        :param instance: (optional)
            ``int`` specifying whether :math:`I_1` (``instance=1``) or :math:`I_2` (``instance=2``) should be plotted.
        :param with_buffer: (optional)
            ``bool`` specifying whether the buffer for the image size should be visualized. If set to ``False``, the true PIV image size is visualized. If set to ``True``, the PIV image with a buffer is visualized and buffer outline is marked with a red rectangle.
        :param xlabel: (optional)
            ``str`` specifying :math:`x`-label.
        :param ylabel: (optional)
            ``str`` specifying :math:`y`-label.
        :param title: (optional)
            ``str`` specifying figure title.
        :param cmap: (optional)
            ``str`` or an object of `matplotlib.colors.ListedColormap <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html>`_ specifying the color map to use.
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

        if not isinstance(instance, int):
            raise ValueError("Parameter `instance` has to be of type 'int'.")
        if instance not in [1,2]:
            raise ValueError("Parameter `instance` has to be 1 (for image I1) or 2 (for image I2).")

        if not isinstance(with_buffer, bool):
            raise ValueError("Parameter `with_buffer` has to be of type 'bool'.")

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

        if self.images_I1 is None:

            print('Note: Particles have not been added to the image yet!\n\n')

        elif instance==2 and self.__images_I2 is None:

            print('Note: Particles have not been advected yet!\n\n')

        else:

            if instance==1:

                image_to_plot = self.__images_I1[idx]

            elif instance==2:

                image_to_plot = self.__images_I2[idx]

            fig = plt.figure(figsize=figsize)

            # Check if particles were generated with a buffer:
            if self.__particles.size_buffer == 0:

                plt.imshow(image_to_plot, cmap=cmap, origin='lower', vmin=0, vmax=self.maximum_intensity)

            else:

                if with_buffer:

                    im = plt.imshow(image_to_plot, cmap=cmap, origin='lower', vmin=0, vmax=self.maximum_intensity)

                    # Extend the imshow area with the buffer:
                    f = lambda pixel: pixel - self.__particles.size_buffer
                    im.set_extent([f(x) for x in im.get_extent()])

                    # Visualize a rectangle that separates the proper PIV image area and the artificial buffer outline:
                    rect = patches.Rectangle((-0.5, -0.5), self.__particles.size[1], self.__particles.size[0], linewidth=1, edgecolor='r', facecolor='none')
                    ax = plt.gca()
                    ax.add_patch(rect)

                else:

                    plt.imshow(image_to_plot[self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer], cmap=cmap, origin='lower', vmin=0, vmax=self.maximum_intensity)

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)

        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')

        return plt

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_image_pair(self,
                        idx,
                        with_buffer=False,
                        xlabel=None,
                        ylabel=None,
                        title=None,
                        cmap='Greys_r',
                        figsize=(5,5),
                        dpi=300,
                        filename=None):
        """
        Plots an animated PIV image pair, :math:`\mathbf{I} = (I_1, I_2)^{\\top}`, at time :math:`t`
        and :math:`t + \\Delta t` respectively.

        :param idx:
            ``int`` specifying the index of the image to plot out of ``n_images`` number of images.
        :param with_buffer:
            ``bool`` specifying whether the buffer for the image size should be visualized. If set to ``False``, the true PIV image size is visualized. If set to ``True``, the PIV image with a buffer is visualized and buffer outline is marked with a red rectangle.
        :param xlabel: (optional)
            ``str`` specifying :math:`x`-label.
        :param ylabel: (optional)
            ``str`` specifying :math:`y`-label.
        :param title: (optional)
            ``str`` specifying figure title.
        :param cmap: (optional)
            ``str`` or an object of `matplotlib.colors.ListedColormap <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html>`_ specifying the color map to use.
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

        if not isinstance(with_buffer, bool):
            raise ValueError("Parameter `with_buffer` has to be of type 'bool'.")

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

        imagelist = [self.images_I1[idx], self.images_I2[idx]]

        if self.__motion is None:

            print('Note: Movement of particles has not been added to the image yet!\n\n')

        else:

            fig = plt.figure(figsize=figsize)

            # Check if particles were generated with a buffer:
            if self.__particles.size_buffer == 0:

                im = plt.imshow(imagelist[0], cmap=cmap, origin='lower', animated=True)

            else:

                if with_buffer:

                    im = plt.imshow(imagelist[0], cmap=cmap, origin='lower', animated=True)

                    # Extend the imshow area with the buffer:
                    f = lambda pixel: pixel - self.__particles.size_buffer
                    im.set_extent([f(x) for x in im.get_extent()])

                    # Visualize a rectangle that separates the proper PIV image area and the artificial buffer outline:
                    rect = patches.Rectangle((-0.5, -0.5), self.__particles.size[1], self.__particles.size[0], linewidth=1, edgecolor='r', facecolor='none')
                    ax = plt.gca()
                    ax.add_patch(rect)

                else:

                    im = plt.imshow(imagelist[0][self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer], cmap=cmap, origin='lower', animated=True)

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)

        def updatefig(j):

            if self.__particles.size_buffer == 0:
                im.set_array(imagelist[j])
            else:
                if with_buffer:
                    im.set_array(imagelist[j])
                else:
                    im.set_array(imagelist[j][self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer])

            return [im]

        anim = animation.FuncAnimation(fig, updatefig, frames=range(2), interval=50, blit=True)

        anim.save(filename, fps=2, bitrate=-1, dpi=dpi, savefig_kwargs={'bbox_inches' : 'tight'})

        return anim

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_velocity_field(self,
                            idx,
                            with_buffer=False,
                            xlabel=None,
                            ylabel=None,
                            title=None,
                            cmap='viridis',
                            vmin_vmax=None,
                            figsize=(5,5),
                            dpi=300,
                            filename=None):
        """
        Plots each component of a velocity field.

        :param idx:
            ``int`` specifying the index of the velocity field to plot out of ``n_images`` number of images.
        :param with_buffer:
            ``bool`` specifying whether the buffer for the image size should be visualized. If set to ``False``, the true PIV image size is visualized. If set to ``True``, the PIV image with a buffer is visualized and buffer outline is marked with a red rectangle.
        :param xlabel: (optional)
            ``str`` specifying :math:`x`-label.
        :param ylabel: (optional)
            ``str`` specifying :math:`y`-label.
        :param title: (optional)
            ``tuple`` of two ``str`` elements specifying figure titles for each velocity field component.
        :param cmap: (optional)
            ``str`` or an object of `matplotlib.colors.ListedColormap <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html>`_ specifying the color map to use.
        :param vmin_vmax: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) fixed bounds for the colorbar.
        :param figsize: (optional)
            ``tuple`` of two numerical elements specifying the figure size as per ``matplotlib.pyplot``.
        :param dpi: (optional)
            ``int`` specifying the dpi for the image.
        :param filename: (optional)
            ``str`` specifying the path and filename to save an image. If set to ``None``, the image will not be saved. An appendix ``-u`` or ``-v`` will be added to each filename to differentiate between velocity field components.

        :return:
            - **plt** - ``matplotlib.pyplot`` image handle.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(idx, int):
            raise ValueError("Parameter `idx` has to be of type 'int'.")
        if idx < 0:
            raise ValueError("Parameter `idx` has to be non-negative.")

        if not isinstance(with_buffer, bool):
            raise ValueError("Parameter `with_buffer` has to be of type 'bool'.")

        if (xlabel is not None) and (not isinstance(xlabel, str)):
            raise ValueError("Parameter `xlabel` has to be of type 'str'.")

        if (ylabel is not None) and (not isinstance(ylabel, str)):
            raise ValueError("Parameter `ylabel` has to be of type 'str'.")

        if (title is not None) and (not isinstance(title, tuple)):
            raise ValueError("Parameter `title` has to be of type 'tuple'.")

        if vmin_vmax is not None:
            check_two_element_tuple(vmin_vmax, 'vmin_vmax')
            check_min_max_tuple(vmin_vmax)

        check_two_element_tuple(figsize, 'figsize')

        if not isinstance(dpi, int):
            raise ValueError("Parameter `dpi` has to be of type 'int'.")

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if self.__flowfield is None:

            print('Note: Flow field has not been added to the image yet!\n\n')

        else:

            # Plot u-component of velocity:

            fig1 = plt.figure(figsize=figsize)

            # Check if flowfield has been generated with a buffer:
            if self.__flowfield.size_buffer == 0:

                if vmin_vmax is not None:
                    plt.imshow(self.__flowfield.velocity_field[idx][0], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin='lower')
                else:
                    plt.imshow(self.__flowfield.velocity_field[idx][0], cmap=cmap, origin='lower')

            else:

                if with_buffer:

                    if vmin_vmax is not None:
                        im = plt.imshow(self.__flowfield.velocity_field[idx][0], cmap=cmap, vmin=vmin_vmax[0],
                                   vmax=vmin_vmax[1], origin='lower')
                    else:
                        im = plt.imshow(self.__flowfield.velocity_field[idx][0], cmap=cmap, origin='lower')

                    # Extend the imshow area with the buffer:
                    f = lambda pixel: pixel - self.__flowfield.size_buffer
                    im.set_extent([f(x) for x in im.get_extent()])

                    # Visualize a rectangle that separates the proper PIV image area and the artificial buffer outline:
                    rect = patches.Rectangle((-0.5, -0.5), self.__flowfield.size[1], self.__flowfield.size[0], linewidth=1, edgecolor='r', facecolor='none')
                    ax = plt.gca()
                    ax.add_patch(rect)

                else:

                    if vmin_vmax is not None:
                        plt.imshow(self.__flowfield.velocity_field[idx][0][self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin='lower')
                    else:
                        plt.imshow(self.__flowfield.velocity_field[idx][0][self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, origin='lower')

            if xlabel is not None:
                plt.xlabel(xlabel)

            if ylabel is not None:
                plt.ylabel(ylabel)

            if title is not None:
                plt.title(title[0])

            plt.colorbar()

            if filename is not None:

                plt.savefig(filename.split('.')[0] + '-u.' + filename.split('.')[1], dpi=dpi, bbox_inches='tight')

            # Plot v-component of velocity:

            fig2 = plt.figure(figsize=figsize)

            # Check if flowfield has been generated with a buffer:
            if self.__flowfield.size_buffer == 0:

                if vmin_vmax is not None:
                    plt.imshow(self.__flowfield.velocity_field[idx][1], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin='lower')
                else:
                    plt.imshow(self.__flowfield.velocity_field[idx][1], cmap=cmap, origin='lower')

            else:

                if with_buffer:

                    if vmin_vmax is not None:
                        im = plt.imshow(self.__flowfield.velocity_field[idx][1], cmap=cmap, vmin=vmin_vmax[0],
                                   vmax=vmin_vmax[1], origin='lower')
                    else:
                        im = plt.imshow(self.__flowfield.velocity_field[idx][1], cmap=cmap, origin='lower')

                    # Extend the imshow area with the buffer:
                    f = lambda pixel: pixel - self.__flowfield.size_buffer
                    im.set_extent([f(x) for x in im.get_extent()])

                    # Visualize a rectangle that separates the proper PIV image area and the artificial buffer outline:
                    rect = patches.Rectangle((-0.5, -0.5), self.__flowfield.size[1], self.__flowfield.size[0], linewidth=1, edgecolor='r', facecolor='none')
                    ax = plt.gca()
                    ax.add_patch(rect)

                else:

                    if vmin_vmax is not None:
                        plt.imshow(self.__flowfield.velocity_field[idx][1][self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin='lower')
                    else:
                        plt.imshow(self.__flowfield.velocity_field[idx][1][self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, origin='lower')

            if xlabel is not None:
                plt.xlabel(xlabel)

            if ylabel is not None:
                plt.ylabel(ylabel)

            if title is not None:
                plt.title(title[1])

            plt.colorbar()

            if filename is not None:
                plt.savefig(filename.split('.')[0] + '-v.' + filename.split('.')[1], dpi=dpi, bbox_inches='tight')

        return fig1, fig2

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_velocity_field_magnitude(self,
                                      idx,
                                      with_buffer=False,
                                      xlabel=None,
                                      ylabel=None,
                                      title=None,
                                      cmap='viridis',
                                      vmin_vmax=None,
                                      add_quiver=False,
                                      quiver_step=10,
                                      quiver_color='k',
                                      add_streamplot=False,
                                      streamplot_density=1,
                                      streamplot_color='k',
                                      figsize=(5,5),
                                      dpi=300,
                                      filename=None):
        """
        Plots a velocity field magnitude.
        In addition, velocity vectors can be visualized by setting ``add_quiver=True``,
        and velocity streamlines can be visualized by setting ``add_streamplot=True``.

        :param idx:
            ``int`` specifying the index of the velocity field to plot out of ``n_images`` number of images.
        :param with_buffer:
            ``bool`` specifying whether the buffer for the image size should be visualized. If set to ``False``, the true PIV image size is visualized. If set to ``True``, the PIV image with a buffer is visualized and buffer outline is marked with a red rectangle.
        :param xlabel: (optional)
            ``str`` specifying :math:`x`-label.
        :param ylabel: (optional)
            ``str`` specifying :math:`y`-label.
        :param title: (optional)
            ``str`` specifying figure title.
        :param cmap: (optional)
            ``str`` or an object of `matplotlib.colors.ListedColormap <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html>`_ specifying the color map to use.
        :param vmin_vmax: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) fixed bounds for the colorbar.
        :param add_quiver: (optional)
            ``bool`` specifying if vector field should be plotted on top of the scalar magnitude field.
        :param quiver_step: (optional)
            ``int`` specifying the step on the pixel grid to attach a vector to. The higher this number is, the less dense the vector field is.
        :param quiver_color: (optional)
            ``str`` specifying the color of velocity vectors.
        :param add_streamplot: (optional)
            ``bool`` specifying if streamlines should be plotted on top of the scalar magnitude field.
        :param streamplot_density: (optional)
            ``float`` or ``int`` specifying the streamplot density.
        :param streamplot_color: (optional)
            ``str`` specifying the streamlines color.
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

        if not isinstance(with_buffer, bool):
            raise ValueError("Parameter `with_buffer` has to be of type 'bool'.")

        if (xlabel is not None) and (not isinstance(xlabel, str)):
            raise ValueError("Parameter `xlabel` has to be of type 'str'.")

        if (ylabel is not None) and (not isinstance(ylabel, str)):
            raise ValueError("Parameter `ylabel` has to be of type 'str'.")

        if (title is not None) and (not isinstance(title, str)):
            raise ValueError("Parameter `title` has to be of type 'str'.")

        if vmin_vmax is not None:
            check_two_element_tuple(vmin_vmax, 'vmin_vmax')
            check_min_max_tuple(vmin_vmax)

        if not isinstance(add_quiver, bool):
            raise ValueError("Parameter `add_quiver` has to be of type 'bool'.")

        if not isinstance(quiver_step, int):
            raise ValueError("Parameter `quiver_step` has to be of type 'int'.")

        if not isinstance(quiver_color, str):
            raise ValueError("Parameter `quiver_color` has to be of type 'str'.")

        if not isinstance(add_streamplot, bool):
            raise ValueError("Parameter `add_streamplot` has to be of type 'bool'.")

        if (not isinstance(streamplot_density, float)) and (not isinstance(streamplot_density, int)):
            raise ValueError("Parameter `streamplot_density` has to be of type 'float' or 'int'.")

        if not isinstance(streamplot_color, str):
            raise ValueError("Parameter `streamplot_color` has to be of type 'str'.")

        check_two_element_tuple(figsize, 'figsize')

        if not isinstance(dpi, int):
            raise ValueError("Parameter `dpi` has to be of type 'int'.")

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if self.__flowfield is None:

            print('Note: Flow field has not been added to the image yet!\n\n')

        else:

            fig = plt.figure(figsize=figsize)

            # Check if flowfield has been generated with a buffer:
            if self.__flowfield.size_buffer == 0:

                if vmin_vmax is not None:
                    plt.imshow(self.__flowfield.velocity_field_magnitude[idx], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin='lower')
                else:
                    plt.imshow(self.__flowfield.velocity_field_magnitude[idx], cmap=cmap, origin='lower')

            else:

                if with_buffer:

                    if vmin_vmax is not None:
                        im = plt.imshow(self.__flowfield.velocity_field_magnitude[idx], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin='lower')
                    else:
                        im = plt.imshow(self.__flowfield.velocity_field_magnitude[idx], cmap=cmap, origin='lower')

                    # Extend the imshow area with the buffer:
                    f = lambda pixel: pixel - self.__flowfield.size_buffer
                    im.set_extent([f(x) for x in im.get_extent()])

                    # Visualize a rectangle that separates the proper PIV image area and the artificial buffer outline:
                    rect = patches.Rectangle((-0.5, -0.5), self.__flowfield.size[1], self.__flowfield.size[0], linewidth=1, edgecolor='r', facecolor='none')
                    ax = plt.gca()
                    ax.add_patch(rect)

                else:

                    if vmin_vmax is not None:
                        plt.imshow(self.__flowfield.velocity_field_magnitude[idx][self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin='lower')
                    else:
                        plt.imshow(self.__flowfield.velocity_field_magnitude[idx][self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, origin='lower')

            if xlabel is not None:
                plt.xlabel(xlabel)

            if ylabel is not None:
                plt.ylabel(ylabel)

            if title is not None:
                plt.title(title)

            plt.colorbar()

            # Check if flowfield has been generated with a buffer:
            if self.__flowfield.size_buffer == 0:

                if add_quiver:
                    X = np.arange(0,self.__flowfield.size[1],quiver_step)
                    Y = np.arange(0,self.__flowfield.size[0],quiver_step)
                    plt.quiver(X, Y, self.__flowfield.velocity_field[idx][0][::quiver_step,::quiver_step], self.__flowfield.velocity_field[idx][1][::quiver_step,::quiver_step], color=quiver_color)

                if add_streamplot:
                    X = np.arange(0,self.__flowfield.size[1],1)
                    Y = np.arange(0,self.__flowfield.size[0],1)
                    plt.streamplot(X, Y, self.__flowfield.velocity_field[idx][0], self.__flowfield.velocity_field[idx][1], density=streamplot_density, color=streamplot_color)

            else:

                if with_buffer:

                    if add_quiver:
                        X = np.arange(-self.__flowfield.size_buffer, self.__flowfield.size[1] + self.__flowfield.size_buffer, quiver_step)
                        Y = np.arange(-self.__flowfield.size_buffer, self.__flowfield.size[0] + self.__flowfield.size_buffer, quiver_step)

                        plt.quiver(X, Y, self.__flowfield.velocity_field[idx][0][::quiver_step, ::quiver_step], self.__flowfield.velocity_field[idx][1][::quiver_step, ::quiver_step], color=quiver_color)

                        plt.xlim([-self.__flowfield.size_buffer, self.__flowfield.size[1] + self.__flowfield.size_buffer])
                        plt.ylim([-self.__flowfield.size_buffer, self.__flowfield.size[0] + self.__flowfield.size_buffer])

                    if add_streamplot:
                        X = np.arange(-self.__flowfield.size_buffer, self.__flowfield.size[1] + self.__flowfield.size_buffer, 1)
                        Y = np.arange(-self.__flowfield.size_buffer, self.__flowfield.size[0] + self.__flowfield.size_buffer, 1)
                        plt.streamplot(X, Y, self.__flowfield.velocity_field[idx][0], self.__flowfield.velocity_field[idx][1], density=streamplot_density, color=streamplot_color)

                        plt.xlim([-self.__flowfield.size_buffer, self.__flowfield.size[1] + self.__flowfield.size_buffer])
                        plt.ylim([-self.__flowfield.size_buffer, self.__flowfield.size[0] + self.__flowfield.size_buffer])

                else:

                    if add_quiver:
                        X = np.arange(0, self.__flowfield.size[1], quiver_step)
                        Y = np.arange(0, self.__flowfield.size[0], quiver_step)

                        velocity_field_u_subset = self.__flowfield.velocity_field[idx][0][self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer]
                        velocity_field_v_subset = self.__flowfield.velocity_field[idx][1][self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer]

                        plt.quiver(X, Y, velocity_field_u_subset[::quiver_step, ::quiver_step], velocity_field_v_subset[::quiver_step, ::quiver_step], color=quiver_color)

                    if add_streamplot:
                        X = np.arange(0, self.__flowfield.size[1], 1)
                        Y = np.arange(0, self.__flowfield.size[0], 1)

                        plt.streamplot(X, Y, self.__flowfield.velocity_field[idx][0][self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer],
                                       self.__flowfield.velocity_field[idx][1][self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], density=streamplot_density, color=streamplot_color)

            if filename is not None:

                plt.savefig(filename, dpi=dpi, bbox_inches='tight')

        return plt

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_image_histogram(self,
                             image,
                             logscale=False,
                             xlabel=None,
                             ylabel=None,
                             title=None,
                             color='k',
                             figsize=(5, 5),
                             dpi=300,
                             filename=None):
        """
        Plots a historgram of the image intensity.

        :param image:
            ``numpy.ndarray`` specifying the single PIV image.
        :param logscale:
            ``bool`` specifying whether the y-axis should be plotted in the logscale.
        :param xlabel: (optional)
            ``str`` specifying :math:`x`-label.
        :param ylabel: (optional)
            ``str`` specifying :math:`y`-label.
        :param title: (optional)
            ``str`` specifying figure title.
        :param color: (optional)
            ``str`` specifying the color of the bars.
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

        if not isinstance(image, np.ndarray):
            raise ValueError("Parameter `image` has to be of type 'numpy.ndarray'.")

        if not isinstance(logscale, bool):
            raise ValueError("Parameter `logscale` has to be of type 'bool'.")

        if (xlabel is not None) and (not isinstance(xlabel, str)):
            raise ValueError("Parameter `xlabel` has to be of type 'str'.")

        if (ylabel is not None) and (not isinstance(ylabel, str)):
            raise ValueError("Parameter `ylabel` has to be of type 'str'.")

        if (title is not None) and (not isinstance(title, str)):
            raise ValueError("Parameter `title` has to be of type 'str'.")

        if not isinstance(color, str):
            raise ValueError("Parameter `color` has to be of type 'str'.")

        check_two_element_tuple(figsize, 'figsize')

        if not isinstance(dpi, int):
            raise ValueError("Parameter `dpi` has to be of type 'int'.")

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        fig = plt.figure(figsize=figsize)

        plt.hist(image.ravel(), color=color)
        plt.xlim([0,self.maximum_intensity])

        if logscale:
            plt.yscale('log')

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)

        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')

        return plt

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
