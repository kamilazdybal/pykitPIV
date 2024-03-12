import matplotlib.pyplot as plt
import h5py
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from pykitPIV.checks import *
from pykitPIV.flowfield import FlowField
from pykitPIV.motion import Motion
from pykitPIV.particle import Particle


# self.__LEF = self.__light_enhancement_factor[0] + np.random.rand(self.__n_images) * (
#             self.__light_enhancement_factor[1] - self.__light_enhancement_factor[0])
# self.__noise_s = self.__LEF / self.__SNR  # Not sure yet what that is...

################################################################################
################################################################################
####
####    Class: Image
####
################################################################################
################################################################################

class Image:
    """
    Stores and plots synthetic PIV images and/or the associated flow fields at any stage of particle generation and movement.

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
                 random_seed=None,
                 ):

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

    def add_velocity_field(self,
                           flowfield):
        """
        Adds velocity field to the image. The velocity field should be defined using the ``FlowField`` class.

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
        Computes the intensity of light reflected from a particle at a requested pixel at position relative to the particle centroid.
        The reflected light follows a Gaussian distribution, i.e., the light intensity value, :math:`i_p`, at the request pixel, :math:`p`, is computed as:

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
        The reflected light follows a Gaussian distribution and is computed using the ``Image.compute_light_intensity_at_pixel()`` method.

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

        if self.__particles is None:
            raise NameError("Particles have not been added to the image yet! Use the `Image.add_particles()` method first.")

        if self.random_seed is not None:
            np.random.seed(seed=self.random_seed)

        if not(isinstance(maximum_intensity, int)):
            raise ValueError("Parameter `maximum_intensity` has to be an instance of `int`.")

        self.__maximum_intensity = maximum_intensity

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Randomize image exposure:
        self.__exposures_per_image = np.random.rand(self.__particles.n_images) * (exposures[1] - exposures[0]) + exposures[0]

        # Function for adding Gaussian light distribution to image I1 and image I2:
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
        ``Image.images_I2_no_buffer`` with copies of ``Image.images_I1``, ``Image.images_I2`` but with the buffer removed,
        and it also populates the class attribute  ``Image.targets_no_buffer``,
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

    def image_pairs_to_tensor(self):
        """
        Prepares a 4-dimensional array with dimensions: ``(n_images, 2, image_height, image_width)`` that stores the image pairs.

        The second dimension represents the image pair, :math:`I_1` and :math:`I_2`.
        """

        images_tensor = np.zeros((self.__particles.n_images, 2, self.__particles.size[0], self.__particles.size[1]))

        for i in range(0, self.__particles.n_images):

            images_tensor[i, 0, :, :] = self.images_I1_no_buffer[i]
            images_tensor[i, 1, :, :] = self.images_I2_no_buffer[i]

        return images_tensor

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def targets_to_tensor(self):
        """
        Prepares a 4-dimensional array with dimensions: ``(n_images, 2, image_height, image_width)`` that stores the flow targets.

        The second dimension represents the velocity field components, :math:`u` and :math:`v`.
        """

        targets_tensor = np.zeros((self.__particles.n_images, 2, self.__particles.size[0], self.__particles.size[1]))

        for i in range(0, self.__particles.n_images):

            targets_tensor[i, 0, :, :] = self.targets_no_buffer[i][0]
            targets_tensor[i, 1, :, :] = self.targets_no_buffer[i][1]

        return targets_tensor

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def save_to_h5(self,
                   with_buffer=False,
                   save_individually=False,
                   filename=None):
        """
        Saves the image pairs, :math:`\\mathbf{I} = (I_1, I_2)`, to ``.h5`` data format.

        :param with_buffer: (optional)
            ``bool`` specifying whether the buffer for the image size should be saved. If set to ``False``, the true PIV image size is saved. If set to ``True``, the PIV image with a buffer is saved.
        :param save_individually: (optional)
            ``bool`` specifying whether each image pair should be saved into a separate ``.h5`` file.
            If set to ``False``, all pairs are saved into one ``.h5`` file;
            the even indices refer to :math:`I_1` and the odd indices refer to :math:`I_2`.
        :param filename: (optional)
            ``str`` specifying the path and filename to save the ``.h5`` data. Note that ``'-pair-#'`` will be added
            automatically to your filename for each saved image pair.
            If set to ``None``, a default name ``'PIV-dataset-pair-#.h5'`` will be used.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(with_buffer, bool):
            raise ValueError("Parameter `with_buffer` has to be of type 'bool'.")

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        if filename is None:
            filename = 'PIV-dataset.h5'

        if self.images_I1 is None:
            raise ValueError("Particles have not been added to the image yet!")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if self.images_I1 is not None and self.images_I2 is None:

            print('Note: Particles have not been advected yet! Saving only initial images, `I_1`.\n\n')

            if save_individually:

                n_zero_pad = int(np.ceil(np.log10(self.__particles.n_images)))

                for i in range(0,self.__particles.n_images):

                    if with_buffer:
                        images_to_save = self.__images_I1[i]
                    else:
                        images_to_save = self.__images_I1[i][self.__particles.size_buffer: -self.__particles.size_buffer,self.__particles.size_buffer: -self.__particles.size_buffer]

                    filename_split = filename.split('.')
                    save_filename = filename_split[0] + '-I1-' + str(i+1).zfill(n_zero_pad) + '.h5'

                    print('Saving ' + save_filename + ' ...')

                    with h5py.File(save_filename, 'w', libver='latest') as f:
                        for idx, image in enumerate(images_to_save):
                            dataset = f.create_dataset(str(idx), data=image, compression='gzip', compression_opts=9)
                        f.close()

                print('\nAll datasets saved.\n')

            else:

                images_to_save = []

                for i in range(0,self.__particles.n_images):

                    if with_buffer:
                        images_to_save.append(self.__images_I1[i])
                    else:
                        images_to_save.append(self.__images_I1[i][self.__particles.size_buffer: -self.__particles.size_buffer, self.__particles.size_buffer: -self.__particles.size_buffer])

                print('Saving ' + filename + ' ...')

                with h5py.File(filename, 'w', libver='latest') as f:
                    for idx, image in enumerate(images_to_save):
                        dataset = f.create_dataset(str(idx), data=image, compression='gzip', compression_opts=9)
                    f.close()

                print('\nDataset saved.\n')

        else:

            if save_individually:

                n_zero_pad = int(np.ceil(np.log10(self.__particles.n_images)))

                for i in range(0,self.__particles.n_images):

                    if with_buffer:
                        images_to_save = (self.__images_I1[i], self.__images_I2[i])
                    else:
                        images_to_save = (self.__images_I1[i][self.__particles.size_buffer: -self.__particles.size_buffer,self.__particles.size_buffer: -self.__particles.size_buffer],
                                          self.__images_I2[i][self.__particles.size_buffer: -self.__particles.size_buffer,self.__particles.size_buffer: -self.__particles.size_buffer])

                    filename_split = filename.split('.')
                    save_filename = filename_split[0] + '-pair-' + str(i+1).zfill(n_zero_pad) + '.h5'

                    print('Saving ' + save_filename + ' ...')

                    with h5py.File(save_filename, 'w', libver='latest') as f:
                        for idx, image in enumerate(images_to_save):
                            dataset = f.create_dataset(str(idx), data=image, compression='gzip', compression_opts=9)
                        f.close()

                print('\nAll datasets saved.\n')

            else:

                images_to_save = []

                for i in range(0,self.__particles.n_images):

                    if with_buffer:
                        images_to_save.append(self.__images_I1[i])
                        images_to_save.append(self.__images_I2[i])
                    else:
                        images_to_save.append(self.__images_I1[i][self.__particles.size_buffer: -self.__particles.size_buffer, self.__particles.size_buffer: -self.__particles.size_buffer])
                        images_to_save.append(self.__images_I2[i][self.__particles.size_buffer: -self.__particles.size_buffer, self.__particles.size_buffer: -self.__particles.size_buffer])

                print('Saving ' + filename + ' ...')

                with h5py.File(filename, 'w', libver='latest') as f:
                    for idx, image in enumerate(images_to_save):
                        dataset = f.create_dataset(str(idx), data=image, compression='gzip', compression_opts=9)
                    f.close()

                print('\nDataset saved.\n')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def upload_from_h5(self,
                       filename=None):
        """
        Upload image pairs, :math:`\\mathbf{I} = (I_1, I_2)`, from ``.h5`` data format.

        :param filename: (optional)
            ``str`` specifying the path and filename to save the ``.h5`` data. Note that ``'-pair-#'`` will be added
            automatically to your filename for each saved image pair.
            If set to ``None``, a default name ``'PIV-dataset-pair-#.h5'`` will be used.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        if filename is None:
            filename = 'PIV-dataset.h5'
















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
                    rect = patches.Rectangle((0, 0), self.__particles.size[1], self.__particles.size[0], linewidth=1, edgecolor='r', facecolor='none')
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
        Plots an animated PIV image pair, :math:`\mathbf{I} = (I_1, I_2)^{\\top}`, at time :math:`t` and :math:`t + \\Delta t` respectively.

        :param idx:
            ``int`` specifying the index of the image to plot out of ``n_images`` number of images.
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
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if not isinstance(idx, int):
            raise ValueError("Parameter `idx` has to be of type 'int'.")
        if idx < 0:
            raise ValueError("Parameter `idx` has to be non-negative.")

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
                    rect = patches.Rectangle((0, 0), self.__particles.size[1], self.__particles.size[0], linewidth=1, edgecolor='r', facecolor='none')
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

        anim.save(filename, fps=2, bitrate=-1, dpi=200)

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
            ``tuple`` of two numerical elements specifying the fixed minimum (first element) and maximum (second element) bounds for the colorbar.
        :param figsize: (optional)
            ``tuple`` of two numerical elements specifying the figure size as per ``matplotlib.pyplot``.
        :param dpi: (optional)
            ``int`` specifying the dpi for the image.
        :param filename: (optional)
            ``str`` specifying the path and filename to save an image. If set to ``None``, the image will not be saved. An appendix ``-u`` or ``-v`` will be added to each filename to differentiate between velocity field components.
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
                    rect = patches.Rectangle((0, 0), self.__flowfield.size[1], self.__flowfield.size[0], linewidth=1, edgecolor='r', facecolor='none')
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
                    rect = patches.Rectangle((0, 0), self.__flowfield.size[1], self.__flowfield.size[0], linewidth=1, edgecolor='r', facecolor='none')
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
            ``tuple`` of two numerical elements specifying the fixed minimum (first element) and maximum (second element) bounds for the colorbar.
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
                    rect = patches.Rectangle((0, 0), self.__flowfield.size[1], self.__flowfield.size[0], linewidth=1, edgecolor='r', facecolor='none')
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
