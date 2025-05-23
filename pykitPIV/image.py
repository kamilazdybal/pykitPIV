import h5py
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from pykitPIV.checks import *
from pykitPIV.flowfield import FlowField
from pykitPIV.motion import Motion
from pykitPIV.particle import Particle
from scipy.ndimage import map_coordinates

########################################################################################################################
########################################################################################################################
####
####    Class: ImageSpecs
####
########################################################################################################################
########################################################################################################################

class ImageSpecs:
    """
    Configuration object for the ``Image`` class.

    **Example:**

    .. code:: python

        from pykitPIV import ImageSpecs

        # Instantiate an object of ImageSpecs class:
        image_spec = ImageSpecs()

        # Change one field of motion_spec:
        image_spec.exposures = 0.95

        # You can print the current values of all attributes:
        print(image_spec)
    """

    def __init__(self,
                 n_images=1,
                 size=(512, 512),
                 size_buffer=10,
                 random_seed=None,
                 exposures=(0.5, 1),
                 maximum_intensity=2**16 - 1,
                 laser_beam_thickness=1,
                 laser_over_exposure=1,
                 laser_beam_shape=0.95,
                 no_laser_plane=False,
                 alpha=1/8,
                 extend_gaussian=1,
                 covariance_matrix=None,
                 clip_intensities=True,
                 normalize_intensities=False,
                 dtype=np.float64):

        self.n_images = n_images
        self.size = size
        self.size_buffer = size_buffer
        self.random_seed = random_seed
        self.exposures = exposures
        self.maximum_intensity = maximum_intensity
        self.laser_beam_thickness = laser_beam_thickness
        self.laser_over_exposure = laser_over_exposure
        self.laser_beam_shape = laser_beam_shape
        self.no_laser_plane = no_laser_plane
        self.alpha = alpha
        self.extend_gaussian = extend_gaussian
        self.covariance_matrix = covariance_matrix
        self.clip_intensities = clip_intensities
        self.normalize_intensities = normalize_intensities
        self.dtype = dtype

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"(n_images={self.n_images},\n"
                f"size={self.size},\n"
                f"size_buffer={self.size_buffer},\n"
                f"random_seed={self.random_seed},\n"
                f"exposures={self.exposures},\n"
                f"maximum_intensity={self.maximum_intensity},\n"
                f"laser_beam_thickness={self.laser_beam_thickness},\n"
                f"laser_over_exposure={self.laser_over_exposure},\n"
                f"laser_beam_shape={self.laser_beam_shape},\n"
                f"no_laser_plane={self.no_laser_plane},\n"
                f"alpha={self.alpha},\n"
                f"extend_gaussian={self.extend_gaussian},\n"
                f"covariance_matrix={self.covariance_matrix},\n"
                f"clip_intensities={self.clip_intensities},\n"
                f"normalize_intensities={self.normalize_intensities},\n"
                f"dtype={self.dtype})"
                )

########################################################################################################################
########################################################################################################################
####
####    Class: Image
####
########################################################################################################################
########################################################################################################################

class Image:
    """
    Stores and plots synthetic PIV images and/or the associated flow targets at any stage of particle generation
    and movement.

    **Example:**

    .. code:: python

        from pykitPIV import Image
        import numpy as np

        # Initialize an image object:
        image = Image(verbose=False,
                      dtype=np.float32,
                      random_seed=100)

    :param verbose: (optional)
        ``bool`` specifying if the verbose print statements should be displayed.
    :param dtype: (optional)
        ``numpy.dtype`` specifying the data type for image intensities. To reduce memory, you can switch from the
        default ``numpy.float64`` to ``numpy.float32``.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``.
        If specified, all operations are reproducible.

    **Attributes:**

    - **verbose** - (read-only) as per user input.
    - **dtype** - (read-only) as per user input.
    - **random_seed** - (read-only) as per user input.
    - **images_I1** - (read-only) ``numpy.ndarray`` of size :math:`(N, C_{in}, H+2b, W+2b)`, where :math:`N` is the
      number PIV image pairs, :math:`C_{in}` is the number of channels (one channel, greyscale, is supported at the moment),
      :math:`H` is the height and :math:`W` is the width of each PIV image, :math:`I_1`, and
      :math:`b` is the optional buffer.
      Only available after ``Image.add_particles()`` has been called.
    - **images_I2** - (read-only) ``numpy.ndarray`` of size :math:`(N, C_{in}, H+2b, W+2b)`, where :math:`N` is the
      number PIV image pairs, :math:`C_{in}` is the number of channels (one channel, greyscale, is supported at the moment),
      :math:`H` is the height and :math:`W` is the width of each PIV image, :math:`I_2`, and
      :math:`b` is the optional buffer.
      Only available after ``Image.add_motion()`` has been called.
    - **exposures_per_image** - (read-only) ``numpy.ndarray`` specifying the template for the light exposure for each image.
      Only available after ``Image.add_reflected_light`` has been called.
    - **maximum_intensity** - (read-only) ``int`` specifying the maximum intensity that was used when adding reflected
      light to the image. Only available after ``Image.add_reflected_light`` has been called.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __init__(self,
                 verbose=False,
                 dtype=np.float64,
                 random_seed=None):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:
        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be of type 'bool'.")

        if not isinstance(dtype, type):
            raise ValueError("Parameter `dtype` has to be of type 'type'.")

        if random_seed is not None:
            if type(random_seed) != int:
                raise ValueError("Parameter `random_seed` has to be of type 'int'.")
            else:
                np.random.seed(seed=random_seed)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:
        self.__verbose = verbose
        self.__dtype = dtype
        self.__random_seed = random_seed

        # Initialize images:
        self.__images_I1 = None
        self.__images_I2 = None

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
    def verbose(self):
        return self.__verbose

    @property
    def dtype(self):
        return self.__dtype

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

        Calling this function populates the ``image.images_I1`` attribute and the private attribute ``image.__particles``
        which gives the user access to attributes from the ``Particle`` object.

        **Example:**

        .. code:: python

            from pykitPIV import Particle, Image

            # Initialize a particle object:
            particles = Particle(1,
                                 size=(128, 512),
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
        self.__images_I2 = self.__particles.particle_positions

        if self.__verbose: print('Particles added to the image.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_flowfield(self,
                      flowfield):
        """
        Adds the flow field to the image. The flow field should be defined using the ``FlowField`` class.

        Calling this function populates the private attribute ``image.__flowfield``
        which gives the user access to attributes from the ``FlowField`` object.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField, Image

            # Initialize a flow field object:
            flowfield = FlowField(1,
                                  size=(128, 512),
                                  size_buffer=10,
                                  time_separation=1,
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

        if self.__verbose: print('Velocity field added to the image.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_motion(self,
                   motion):
        """
        Adds particle movement to the image. The movement should be defined using the ``Motion`` class.

        Calling this function populates the private attribute ``image.__motion``
        which gives the user access to attributes from the ``Motion`` object.

        **Example:**

        .. code:: python

            from pykitPIV import Particle, FlowField, Motion, Image

            # Initialize a particle object:
            particles = Particle(1,
                                 size=(128, 512),
                                 size_buffer=10,
                                 random_seed=100)

            # Initialize a flow field object:
            flowfield = FlowField(1,
                                  size=(128, 512),
                                  size_buffer=10,
                                  time_separation=1,
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

        if self.__verbose: print('Particle movement added to the image.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def get_velocity_field(self):
        """
        Returns the velocity field from the object of the ``FlowField`` class.

        **Example:**

        .. code:: python

            from pykitPIV import Particle, FlowField, Image

            # Initialize a particle object:
            particles = Particle(1,
                                 size=(128, 512),
                                 size_buffer=10,
                                 random_seed=100)

            # Initialize a flow field object:
            flowfield = FlowField(1,
                                  size=(128, 512),
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Generate random velocity field:
            flowfield.generate_random_velocity_field(displacement=(0, 10),
                                                     gaussian_filters=(10, 30),
                                                     n_gaussian_filter_iter=6)

            # Initialize an image object:
            image = Image(random_seed=100)

            # Add flow field to an image:
            image.add_flowfield(flowfield)

            # Access the velocity field:
            V = image.get_velocity_field()

        :return:
            - **velocity_field** - as per ``FlowField`` class.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if self.__flowfield is None:
            raise NameError("Flow field has not been added to the image yet! Use the `Image.add_flowfield()` method first.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        return self.__flowfield.velocity_field

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def get_velocity_field_magnitude(self):
        """
        Returns the velocity field magnitude from the object of the ``FlowField`` class.

        **Example:**

        .. code:: python

            from pykitPIV import Particle, FlowField, Image

            # Initialize a particle object:
            particles = Particle(1,
                                 size=(128, 512),
                                 size_buffer=10,
                                 random_seed=100)

            # Initialize a flow field object:
            flowfield = FlowField(1,
                                  size=(128, 512),
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Generate random velocity field:
            flowfield.generate_random_velocity_field(displacement=(0, 10),
                                                     gaussian_filters=(10, 30),
                                                     n_gaussian_filter_iter=6)

            # Initialize an image object:
            image = Image(random_seed=100)

            # Add flow field to an image:
            image.add_flowfield(flowfield)

            # Access the velocity field magnitude:
            V_mag = image.get_velocity_field_magnitude()

        :return:
            - **velocity_field_magnitude** - as per ``FlowField`` class.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if self.__flowfield is None:
            raise NameError("Flow field has not been added to the image yet! Use the `Image.add_flowfield()` method first.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        return self.__flowfield.velocity_field_magnitude

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def get_displacement_field(self):
        """
        Returns the displacement field from the object of the ``FlowField`` class.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField, Image

            # Initialize a flow field object:
            flowfield = FlowField(1,
                                  size=(128, 512),
                                  size_buffer=10,
                                  time_separation=0.1,
                                  random_seed=100)

            # Generate random velocity field:
            flowfield.generate_random_velocity_field(displacement=(0, 10),
                                                     gaussian_filters=(10, 30),
                                                     n_gaussian_filter_iter=6)

            # Initialize an image object:
            image = Image(random_seed=100)

            # Add motion to an image:
            image.add_flowfield(flowfield)

            # Access the displacement field:
            ds = image.get_displacement_field()

        :return:
            - **displacement_field** - as per ``FlowField`` class.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if self.__flowfield is None:
            raise NameError("Flow field has not been added to the image yet! Use the `Image.add_flowfield()` method first.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__flowfield.compute_displacement_field()

        return self.__flowfield.displacement_field

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def get_displacement_field_magnitude(self):
        """
        Returns the displacement field magnitude from the object of the ``FlowField`` class.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField, Image

            # Initialize a flow field object:
            flowfield = FlowField(1,
                                  size=(128, 512),
                                  size_buffer=10,
                                  time_separation=0.1,
                                  random_seed=100)

            # Generate random velocity field:
            flowfield.generate_random_velocity_field(displacement=(0, 10),
                                                     gaussian_filters=(10, 30),
                                                     n_gaussian_filter_iter=6)

            # Initialize an image object:
            image = Image(random_seed=100)

            # Add motion to an image:
            image.add_flowfield(flowfield)

            # Access the displacement field magnitude:
            ds_mag = image.get_displacement_field_magnitude()

        :return:
            - **displacement_field_magnitude** - as per ``FlowField`` class.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if self.__flowfield is None:
            raise NameError("Flow field has not been added to the image yet! Use the `Image.add_flowfield()` method first.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__flowfield.compute_displacement_field()

        return self.__flowfield.displacement_field_magnitude

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def compute_light_intensity_at_pixel(self,
                                         peak_intensity,
                                         particle_diameter,
                                         coordinate_height,
                                         coordinate_width,
                                         alpha=1/8,
                                         covariance_matrix=None):
        """
        Computes the intensity of light reflected from a particle at a requested pixel at position relative
        to the particle centroid.

        The reflected light follows a Gaussian distribution which can be either univariate
        (in which case particles are spherical Gaussians)
        or covariate (in which case particles have an elongated shape).

        For a univariate distrubution, the light intensity value, :math:`i_p`, at the
        requested pixel, :math:`p`, is computed as:

        .. math::

            i_p =  i_{\\text{peak}} \\cdot \\exp \Big(- \\frac{h_p^2 + w_p^2}{\\alpha \\cdot d_p^2} \Big)

        where:

        - :math:`i_{\\text{peak}}` is the peak intensity applied at the particle centroid.
        - :math:`h_p` is the pixel coordinate in the image height direction relative to the particle centroid.
        - :math:`w_p` is the pixel coordinate in the image width direction relative to the particle centroid.
        - :math:`\\alpha` is a custom multiplier, :math:`\\alpha`. The default value is :math:`\\alpha = 1/8`.
        - :math:`d_p` is the particle diameter.

        For a covariate distrubution, the light intensity value, :math:`i_p`, at the requested pixel, :math:`p`, is computed as:

        .. math::

            i_p =  i_{\\text{peak}} \\cdot \\exp \Big(- \\frac{\mathbf{r} \cdot \\mathbf{C}^{-1} \cdot \\mathbf{r}^{\\top}}{\\alpha \\cdot d_p^2} \Big)

        where:

        - :math:`\\mathbf{C}` is the covariance matrix that is positive semi-definite and :math:`\\mathbf{C}^{-1}` is its inverse.
        - :math:`\\mathbf{r}` is the position vector, :math:`\\mathbf{r} = [w_p, h_p]`.

        :param peak_intensity:
            ``float`` specifying the peak intensity, :math:`i_{\\text{peak}}`, to apply at the particle centroid.
        :param particle_diameter:
            ``float`` specifying the particle diameter :math:`d_p`.
        :param coordinate_height:
            ``float`` specifying the pixel coordinate in the image height direction, :math:`h_p`, relative to the particle centroid.
        :param coordinate_width:
            ``float`` specifying the pixel coordinate in the image width direction, :math:`w_p`, relative to the particle centroid.
        :param alpha: (optional):
            ``float`` specifying the custom multiplier, :math:`\\alpha`, for the squared particle radius.
            The default and recommended value is :math:`1/8` as per
            `Raffel et al. (2018) <https://link.springer.com/book/10.1007/978-3-319-68852-7>`_,
            `Rabault et al. (2017) <https://iopscience.iop.org/article/10.1088/1361-6501/aa8b87/meta>`_
            and `Manickathan et al. (2022) <https://iopscience.iop.org/article/10.1088/1361-6501/ac8fae>`_.
        :param covariance_matrix: (optional):
            ``numpy.ndarray`` specifying the covariance matrix that has to be positive semi-definite.
            If set to ``None``, the particle images will be generated from a univariate distrubtion
            (as spherical Gaussians). It has to have size ``(2, 2)``.

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

        if covariance_matrix is not None:
            if not (isinstance(covariance_matrix, np.ndarray)):
                raise ValueError("Parameter `covariance_matrix` has to be of type `numpy.ndarray`.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if covariance_matrix is None:

            pixel_value = peak_intensity * np.exp(-(coordinate_height**2 + coordinate_width**2) / (alpha * particle_diameter**2))

        else:

            covariance_matrix_inv = np.linalg.inv(covariance_matrix)

            r = np.array([coordinate_width, coordinate_height])

            pixel_value = peak_intensity * np.exp(-(r.dot(covariance_matrix_inv.dot(r))) / (alpha * particle_diameter ** 2))

        return pixel_value

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_reflected_light(self,
                            exposures=(0.5, 1),
                            maximum_intensity=2**16-1,
                            laser_beam_thickness=2,
                            laser_over_exposure=1,
                            laser_beam_shape=0.85,
                            no_laser_plane=False,
                            alpha=1/8,
                            extend_gaussian=1,
                            covariance_matrix=None,
                            clip_intensities=True,
                            normalize_intensities=False):
        """
        Creates particle shapes and adds laser light reflected from particles.

        The reflected light follows a Gaussian distribution and is computed using
        the ``Image.compute_light_intensity_at_pixel()`` method.

        The particle shapes can be spherical Gaussians or can have a non-zero covariance. In the latter case,
        the user has to provide a full covariance matrix that is positive semi-definite.

        **Example:**

        .. code:: python

            from pykitPIV import Particle, Image

            # Initialize a particle object:
            particles = Particle(1,
                                 size=(128, 512),
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
                                      no_laser_plane=False,
                                      alpha=1/8,
                                      extend_gaussian=1,
                                      covariance_matrix=None,
                                      clip_intensities=True,
                                      normalize_intensities=False)

        Alternatively, we can model astigmatic PIV by providing a coviariance matrix:

        **Example:**

        .. code:: python

            # Define a full, positive semi-definite covariance matrix:
            covariance_matrix = np.array([[3.0, -2],
                                          [-2, 2.0]])

            # Add reflected light to an image:
            image.add_reflected_light(exposures=(0.5, 0.9),
                                      maximum_intensity=2**16-1,
                                      laser_beam_thickness=1,
                                      laser_over_exposure=1,
                                      laser_beam_shape=0.95,
                                      no_laser_plane=False,
                                      alpha=1/8,
                                      extend_gaussian=1,
                                      covariance_matrix=covariance_matrix,
                                      clip_intensities=True,
                                      normalize_intensities=False)

        .. image:: ../images/Image-setting-spectrum.png
            :width: 800

        :param exposures: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
            light exposure.
            It can also be set to ``int`` or ``float`` to generate a fixed exposure value across all :math:`N` image pairs.
        :param maximum_intensity: (optional)
            ``int`` specifying the maximum light intensity. This will be the brightest possible pixel, which will
            only happen if the particle is located at the center of that pixel and at the center of the laser plane.
            All particles with an offset with respect to the laser plane will have a lower intensity than the maximum.
        :param laser_beam_thickness: (optional)
            ``int`` or ``float`` specifying the thickness of the laser beam. With a small thickness, particles that are
            even slightly off-plane will appear darker. Note, that the thicker the laser plane, the larger the distance
            that the particle can travel away from the laser plane to lose its luminosity.
        :param laser_over_exposure: (optional)
            ``int`` or ``float`` specifying the overexposure of the laser beam.
        :param laser_beam_shape: (optional)
            ``int`` or ``float`` specifying the spread of the Gaussian shape of the laser beam. The larger this number
            is, the wider the Gaussian light distribution from the laser and more particles will be illuminated.
        :param no_laser_plane: (optional)
            ``bool`` specifying whether the laser plane is used to illuminate particles. For PIV images, set to ``False``.
            For BOS images set to ``True``.
        :param alpha: (optional):
            ``float`` specifying the custom multiplier, :math:`\\alpha`, for the squared particle radius as per the
            ``Particle.compute_light_intensity_at_pixel()`` method.
            The default and recommended value is :math:`1/8` as per
            `Raffel et al. (2018) <https://link.springer.com/book/10.1007/978-3-319-68852-7>`_,
            `Rabault et al. (2017) <https://iopscience.iop.org/article/10.1088/1361-6501/aa8b87/meta>`_
            and `Manickathan et al. (2022) <https://iopscience.iop.org/article/10.1088/1361-6501/ac8fae>`_.
        :param extend_gaussian: (optional):
            ``int`` specifying the multiple of the particle radius to be filled with the Gaussian blur. For BOS images,
            it is recommended to increase this number to, say, 2. For PIV images, it can be equal to 1. Note that
            a higher number will increase the computation time for adding light intensity to images.
        :param covariance_matrix: (optional):
            ``numpy.ndarray`` specifying the covariance matrix that has to be positive semi-definite.
            If set to ``None``, the particle images will be generated from a univariate distrubtion
            (as spherical Gaussians). It has to have size ``(2, 2)``.
        :param clip_intensities: (optional):
            ``bool`` specifying whether the image intensities should be clipped if any pixel exceeds
            the maximum light intensity. Only one of ``clip_intensities``, ``normalize_intensities`` can be ``True``.
        :param normalize_intensities: (optional):
            ``bool`` specifying whether the image intensities should be normalized if any pixel exceeds
            the maximum light intensity. Only one of ``clip_intensities``, ``normalize_intensities`` can be ``True``.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if isinstance(exposures, tuple):
            check_two_element_tuple(exposures, 'exposures')
            check_min_max_tuple(exposures, 'exposures')
        elif isinstance(exposures, int) or isinstance(exposures, float):
            check_non_negative_int_or_float(exposures, 'exposures')
        else:
            raise ValueError("Parameter `exposures` has to be of type 'tuple' or 'int' or 'float'.")

        if not(isinstance(maximum_intensity, int)):
            raise ValueError("Parameter `maximum_intensity` has to be of type `int`.")

        if not(isinstance(laser_beam_thickness, int)) and not(isinstance(laser_beam_thickness, float)):
            raise ValueError("Parameter `laser_beam_thickness` has to be of type `int` or `float`.")

        if not(isinstance(laser_over_exposure, int)) and not(isinstance(laser_over_exposure, float)):
            raise ValueError("Parameter `laser_over_exposure` has to be of type `int` or `float`.")

        if not(isinstance(laser_beam_shape, int)) and not(isinstance(laser_beam_shape, float)):
            raise ValueError("Parameter `laser_beam_shape` has to be of type `int` or `float`.")

        if not(isinstance(no_laser_plane, bool)):
            raise ValueError("Parameter `no_laser_plane` has to be of type `bool`.")

        if not(isinstance(alpha, float)):
            raise ValueError("Parameter `alpha` has to be of type `float`.")

        if not(isinstance(extend_gaussian, int)):
            raise ValueError("Parameter `extend_gaussian` has to be of type `int`.")

        if covariance_matrix is not None:
            if not (isinstance(covariance_matrix, np.ndarray)):
                raise ValueError("Parameter `covariance_matrix` has to be of type `numpy.ndarray`.")

        if not(isinstance(clip_intensities, bool)):
            raise ValueError("Parameter `clip_intensities` has to be of type `bool`.")

        if not(isinstance(normalize_intensities, bool)):
            raise ValueError("Parameter `normalize_intensities` has to be of type `bool`.")

        if clip_intensities and normalize_intensities:
            raise ValueError("Only one of `clip_intensities`, `normalize_intensities` can be set to True.")

        if self.random_seed is not None:
            np.random.seed(seed=self.random_seed)

        if self.__particles is None:
            raise NameError("Particles have not been added to the image yet! Use the `Image.add_particles()` method first.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if isinstance(exposures, tuple):
            __exposures = exposures
        else:
            __exposures = (exposures, exposures)

        # Save the maximum intensity:
        self.__maximum_intensity = maximum_intensity

        # Randomize image exposure:
        self.__exposures_per_image = np.random.rand(self.__particles.n_images) * (__exposures[1] - __exposures[0]) + __exposures[0]

        # Function for adding Gaussian light distribution to image I1 or image I2:
        def __gaussian_light(idx,
                             particle_height_coordinate,
                             particle_width_coordinate,
                             image_instance,
                             extend_gaussian=1,
                             no_laser_plane=False):

            # Note, that this number can be different between image I1 and I2 due to removal of particles from image area.
            # That's why we do not set number_of_particles = self.__particles.n_of_particles[idx] -- this would only work for I1.
            number_of_particles = particle_height_coordinate.shape[0]

            # Initialize an empty image:
            particles_with_gaussian_light = np.zeros((self.__particles.size_with_buffer[0], self.__particles.size_with_buffer[1]), dtype=self.dtype)

            # Establish the peak intensity for each particle depending on its position with respect to the laser beam plane:
            if no_laser_plane:
                particle_position_relative_to_laser_centerline = np.zeros((number_of_particles,))
            else:
                particle_positions_off_laser_plane = laser_beam_thickness * np.random.rand(number_of_particles) - laser_beam_thickness / 2
                particle_position_relative_to_laser_centerline = np.abs(particle_positions_off_laser_plane) / (laser_beam_thickness / 2)

            particle_peak_intensities = self.exposures_per_image[idx] * maximum_intensity * np.exp(-0.5 * (particle_position_relative_to_laser_centerline ** 2 / laser_beam_shape ** 2))

            # Add Gaussian blur to each particle location that mimics the light reflect from a particle of a given size:
            for p in range(0, number_of_particles):

                if image_instance == 1:
                    particle_diameter_on_image = self.__particles.particle_diameters[idx][p]
                    ceil_of_particle_radius = np.ceil(self.__particles.particle_diameters[idx][p] / 2).astype(int)
                elif image_instance == 2:
                    particle_diameter_on_image = self.__motion.updated_particle_diameters[idx][p]
                    ceil_of_particle_radius = np.ceil(self.__motion.updated_particle_diameters[idx][p] / 2).astype(int)

                px_c_height = np.floor(particle_height_coordinate[p]).astype(int)
                px_c_width = np.floor(particle_width_coordinate[p]).astype(int)

                # We only apply the Gaussian blur in the square neighborhood of the particle center:
                for h in range(px_c_height - extend_gaussian * ceil_of_particle_radius, px_c_height + extend_gaussian * ceil_of_particle_radius + 1):
                    for w in range(px_c_width - extend_gaussian * ceil_of_particle_radius, px_c_width + extend_gaussian * ceil_of_particle_radius + 1):

                        # Only change the value of pixels that are within the image area:
                        if (h >= 0 and h < self.__particles.size_with_buffer[0]) and (w >= 0 and w < self.__particles.size_with_buffer[1]):
                            # 0.5 is added because we are computing the distance from particle center to the center of each considered pixel:
                            coordinate_height = particle_height_coordinate[p] - (h + 0.5)
                            coordinate_width = particle_width_coordinate[p] - (w + 0.5)

                            particles_with_gaussian_light[h, w] = particles_with_gaussian_light[h, w] + self.compute_light_intensity_at_pixel(particle_peak_intensities[p],
                                                                                                                                              particle_diameter_on_image,
                                                                                                                                              coordinate_height,
                                                                                                                                              coordinate_width,
                                                                                                                                              alpha=alpha,
                                                                                                                                              covariance_matrix=covariance_matrix)

            return particles_with_gaussian_light

        # Different possible ways to handle pixel intensities exceeding the maximum requested intensity:
        def __clip_intensities(image, maximum_intensity):
            return np.clip(image, a_min=0, a_max=maximum_intensity)

        def __normalize_intensities(image, maximum_intensity):
            return (image / np.max(image)) * maximum_intensity

        __user_warned_I1 = False
        __user_warned_I2 = False
        warning_message = 'Some of pixel values in images I1 exceed the requested maximum pixel intensity. Consider clipping or normalizing the image intensities.'

        # Add light to image I1:
        if self.__particles is not None:

            images_I1 = np.zeros((self.__particles.n_images, 1, self.__particles.size_with_buffer[0], self.__particles.size_with_buffer[1]), dtype=self.dtype)

            for i in range(0,self.__particles.n_images):

                particles_with_gaussian_light = __gaussian_light(i,
                                                                 self.__particles.particle_coordinates[i][0],
                                                                 self.__particles.particle_coordinates[i][1],
                                                                 image_instance=1,
                                                                 extend_gaussian=extend_gaussian,
                                                                 no_laser_plane=no_laser_plane)

                if clip_intensities:
                    images_I1[i, 0, :, :] = __clip_intensities(particles_with_gaussian_light, maximum_intensity)

                if normalize_intensities:
                    images_I1[i, 0, :, :] = __normalize_intensities(particles_with_gaussian_light, maximum_intensity)

                # If the user didn't choose to either clip or normalize image intensities:
                if (not clip_intensities) and (not normalize_intensities):
                    images_I1[i, 0, :, :] = particles_with_gaussian_light

                    if not __user_warned_I1:
                        if np.any(particles_with_gaussian_light > maximum_intensity):
                            print(warning_message)
                            __user_warned_I1 = True

                self.__images_I1 = images_I1

            if self.__verbose: print('Reflected light added to images I1.')

        # Add light to image I2:
        if self.__motion is not None:

            images_I2 = np.zeros((self.__particles.n_images, 1, self.__particles.size_with_buffer[0], self.__particles.size_with_buffer[1]), dtype=self.dtype)

            for i in range(0,self.__particles.n_images):

                particles_with_gaussian_light = __gaussian_light(i,
                                                                 self.__motion.particle_coordinates_I2[i][0],
                                                                 self.__motion.particle_coordinates_I2[i][1],
                                                                 image_instance=2,
                                                                 extend_gaussian=extend_gaussian,
                                                                 no_laser_plane=no_laser_plane)

                if clip_intensities:
                    images_I2[i, 0, :, :] = __clip_intensities(particles_with_gaussian_light, maximum_intensity)

                if normalize_intensities:
                    images_I2[i, 0, :, :] = __normalize_intensities(particles_with_gaussian_light, maximum_intensity)

                # If the user didn't choose to either clip or normalize image intensities:
                if (not clip_intensities) and (not normalize_intensities):
                    images_I2[i, 0, :, :] = particles_with_gaussian_light

                    if not __user_warned_I2:
                        if np.any(particles_with_gaussian_light > maximum_intensity):
                            print(warning_message)
                            __user_warned_I2 = True

                self.__images_I2 = images_I2

            if self.__verbose: print('Reflected light added to images I2.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def warp_images(self,
                    time_separation=1,
                    order=1,
                    mode='constant'):
        """
        Warps images according to the velocity field.
        This function can be used to produce :math:`I_2` in background-oriented Schlieren (BOS) by warping :math:`I_1`.

        .. note::

            This function uses ``scipy.ndimage.map_coordinates`` to warp images.

        **Example:**

        .. code:: python

            from pykitPIV import Particle, FlowField, Image

            # We are going to generate 10 BOS image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (512, 512)

            # Initialize a particle object:
            particles = Particle(n_images,
                                 size=image_size,
                                 size_buffer=10,
                                 diameters=10,
                                 distances=1,
                                 densities=1.2,
                                 diameter_std=0,
                                 seeding_mode='poisson',
                                 random_seed=100)

            # Initialize an image object:
            image = Image(random_seed=100)
            image.add_particles(particles)

            image.add_reflected_light(exposures=0.99,
                                      maximum_intensity=2**16-1,
                                      no_laser_plane=True,
                                      alpha=1/8,
                                      extend_gaussian=2)

            # Initialize a flow field object:
            flowfield = FlowField(n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            flowfield.generate_potential_velocity_field(imposed_origin=None,
                                                        displacement=(2, 2))


            image.add_flowfield(flowfield)

            # Warp images:
            image.warp_images(time_separation=10)

        :param time_separation:
            ``float`` or ``int`` specifying the time separation, :math:`\\Delta t`, that will scale the velocity field to obtain a displacement field.
        :param order:
            ``int`` specifying the order of the spline interpolation as per ``scipy.ndimage.map_coordinates``.
            It has to be a number between 0 and 5.
        :param mode:
            ``str`` specifying how the input array is extended beyond its boundaries.
            With sufficient buffer size this should not affect the proper image area.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not(isinstance(time_separation, int)) and not(isinstance(time_separation, float)):
            raise ValueError("Parameter `time_separation` has to be of type `float` or `int`.")

        if not(isinstance(order, int)):
            raise ValueError("Parameter `order` has to be of type `int`.")

        if order < 0 or order > 5:
            raise ValueError("Parameter `order` has to be a number between 0 and 5.")

        if not(isinstance(mode, str)):
            raise ValueError("Parameter `mode` has to be of type `str`.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if self.__flowfield is not None:

            images_I2 = np.zeros((self.__particles.n_images, 1, self.__particles.size_with_buffer[0], self.__particles.size_with_buffer[1]), dtype=self.dtype)

            for i in range(0,self.__particles.n_images):

                yy, xx = np.meshgrid(np.arange(0, self.__particles.size_with_buffer[0]), np.arange(0, self.__particles.size_with_buffer[1]), indexing="ij")
                x_src = xx - time_separation * self.__flowfield.velocity_field[i, 0, :, :]
                y_src = yy - time_separation * self.__flowfield.velocity_field[i, 1, :, :]

                coords = np.zeros((2, self.__particles.size_with_buffer[0], self.__particles.size_with_buffer[1]))
                coords[0, :, :] = y_src
                coords[1, :, :] = x_src

                warped_I1 = map_coordinates(self.images_I1[i,0,:,:],
                                            coords,
                                            order=order,
                                            mode=mode).reshape(self.__particles.size_with_buffer[0], self.__particles.size_with_buffer[1])

                images_I2[i, 0, :, :] = warped_I1

            self.__images_I2 = images_I2

            if self.__verbose: print('Images have been warped to produce I2.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def remove_buffers(self, input_tensor):
        """
        Removes image buffers from the input tensor. If the input tensor is a four-dimensional array of size
        :math:`(N, \_, H+2b, W+2b)`, then the output is a four-dimensional tensor array of size :math:`(N, \_, H, W)`.

        **Example:**

        .. code:: python

            from pykitPIV import Particle, Image

            # Initialize a particle object:
            particles = Particle(10,
                                 size=(128, 512),
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
                                      alpha=1/8,
                                      covariance_matrix=None,
                                      clip_intensities=True,
                                      normalize_intensities=False)

            # Remove buffers from the first image frame:
            images_I1 = image.remove_buffers(image.images_I1)

        :param input_tensor:
            ``numpy.ndarray`` specifying the input tensor. It has to have size :math:`(N, C_{in}, H+2b, W+2b)`.

        :return:
            - **input_tensor** - ``numpy.ndarray`` specifying the input tensor with buffers removed.
              It has size :math:`(N, C_{in}, H, W)`.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        try:
            (N, _, H, W) = np.shape(input_tensor)
        except:
            raise ValueError("Parameter `input_tensor` has to be a four-dimensional tensor array.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if self.__particles.size_buffer > 0:

            return input_tensor[:, :, self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer]

        else:

            print('The buffer size was set to 0. Images do not have a buffer to remove!')

            return input_tensor

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def measure_counts(self, image):
        """
        Measures the number of light intensity counts in a given image.

        **Example:**

        .. code:: python

            from pykitPIV import Particle, Image

            # Initialize a particle object:
            particles = Particle(10,
                                 size=(20, 20),
                                 size_buffer=0)

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
                                      alpha=1/8,
                                      covariance_matrix=None,
                                      clip_intensities=True,
                                      normalize_intensities=False)

            # Measure light intensity counts:
            counts_dictionary = image.measure_counts(image.images_I1[0,:,:])

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

    def concatenate_tensors(self, input_tensor_tuple):
        """
        Concatenates multiple four-dimensional tensor arrays along the second dimension (the "channel" dimension).

        **Example:**

        .. code:: python

            from pykitPIV import Particle, FlowField, Motion, Image

            # Initialize a particle object:
            particles = Particle(1,
                                 size=(128, 512),
                                 size_buffer=10,
                                 random_seed=100)

            # Initialize a flow field object:
            flowfield = FlowField(1,
                                  size=(128, 512),
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Generate random velocity field:
            flowfield.generate_random_velocity_field(displacement=(0, 10),
                                                     gaussian_filters=(10, 30),
                                                     n_gaussian_filter_iter=6)

            # Initialize a motion object:
            motion = Motion(particles, flowfield)
            motion.runge_kutta_4th(n_steps=10)

            # Initialize an image object:
            image = Image(random_seed=100)

            # Add motion to an image:
            image.add_motion(motion)

            # Add particles to an image:
            image.add_particles(particles)

            # Add light reflected from particles:
            image.add_reflected_light(exposures=(0.6, 0.65),
                                      maximum_intensity=2**16-1,
                                      laser_beam_thickness=1,
                                      laser_over_exposure=1,
                                      laser_beam_shape=0.95,
                                      alpha=1/10)

            # Concatenate image frames:
            I = image.concatenate_tensors((image.images_I1,
                                           image.images_I2))


        :return:
            - **input_tensor_tuple** - ``tuple`` of ``numpy.ndarray`` specifying the four-dimensional tensors to concatenate.
              Each ``numpy.ndarray`` should have size :math:`(N, \_, H, W)`, where :math:`N` is the number of PIV image pairs,
              :math:`H` is the height and :math:`W` is the width of each PIV image.
              The second dimension refers to items being concatenated.
        """

        return np.concatenate(input_tensor_tuple, axis=1)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def save_to_h5(self,
                   tensors_dictionary,
                   save_individually=False,
                   filename=None,
                   verbose=False):
        """
        Saves the image pairs tensor and/or the associated flow targets tensor to ``.h5`` data format.

        **Example:**

        .. code:: python

            from pykitPIV import Particle, FlowField, Motion, Image

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
                                 diameter_std=(0.1, 1),
                                 seeding_mode='random',
                                 random_seed=100)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Generate random velocity field:
            flowfield.generate_random_velocity_field(gaussian_filters=(2, 10),
                                                     n_gaussian_filter_iter=10,
                                                     displacement=(2, 5))

            # Initialize a motion object:
            motion = Motion(particles,
                            flowfield,
                            particle_loss=(0, 2),
                            particle_gain='matching',
                            verbose=False,
                            random_seed=None)

            # Advect particles:
            motion.forward_euler(n_steps=10)

            # Initialize an image object:
            image = Image(random_seed=100)

            # Add particles to an image:
            image.add_particles(particles)

            # Add flow field to an image:
            image.add_flowfield(flowfield)

            # Add motion to an image:
            image.add_motion(motion)

            # Add reflected light to an image:
            image.add_reflected_light(exposures=(0.5, 0.9),
                                      maximum_intensity=2**16-1,
                                      laser_beam_thickness=1,
                                      laser_over_exposure=1,
                                      laser_beam_shape=0.95,
                                      alpha=1/8,
                                      covariance_matrix=None,
                                      clip_intensities=True,
                                      normalize_intensities=False)

            # Remove buffers from images:
            images_I1 = image.remove_buffers(image.images_I1)
            images_I2 = image.remove_buffers(image.images_I2)

            # Remove buffers from targets:
            velocity_field = image.remove_buffers(image.get_velocity_field())
            displacement_field = image.remove_buffers(image.get_displacement_field())

            # Prepare a tensors dictionary to save:
            images_intensities = image.concatenate_tensors((images_I1, images_I2))
            flow_targets = image.concatenate_tensors((velocity_field, displacement_field))

            tensors_dictionary = {"I"      : images_intensities,
                                  "target" : flow_targets}

            # Save tensors to h5:
            image.save_to_h5(tensors_dictionary,
                             save_individually=False,
                             filename='dataset.h5',
                             verbose=True)

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
        Upload image pairs tensor and/or the associated flow targets tensor from ``.h5`` data format.

        **Example:**

        .. code:: python

            from pykitPIV import Image

            # Instantiate an empty image object:
            image = Image()

            # Upload the saved dataset to the empty image object:
            tensors_dictionary_uploaded = image.upload_from_h5(filename='dataset.h5')

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
             xticks=True,
             yticks=True,
             title=None,
             cmap='Greys_r',
             cbar=False,
             origin='lower',
             figsize=(5, 5),
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
        :param xticks: (optional)
            ``bool`` specifying if ticks along the :math:`x`-axis should be plotted.
        :param yticks: (optional)
            ``bool`` specifying if ticks along the :math:`y`-axis should be plotted.
        :param title: (optional)
            ``str`` specifying figure title.
        :param cmap: (optional)
            ``str`` or an object of `matplotlib.colors.ListedColormap <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html>`_ specifying the color map to use.
        :param cbar: (optional)
            ``bool`` specifying whether colorbar should be plotted.
        :param origin: (optional)
            ``str`` specifying the origin location. It can be ``'upper'`` or ``'lower'``.
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

        if not isinstance(xticks, bool):
            raise ValueError("Parameter `xticks` has to be of type 'bool'.")

        if not isinstance(yticks, bool):
            raise ValueError("Parameter `yticks` has to be of type 'bool'.")

        if (title is not None) and (not isinstance(title, str)):
            raise ValueError("Parameter `title` has to be of type 'str'.")

        if not isinstance(cbar, bool):
            raise ValueError("Parameter `cbar` has to be of type 'bool'.")

        if not isinstance(origin, str):
            raise ValueError("Parameter `origin` has to be of type 'str'.")

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

                image_to_plot = self.__images_I1[idx, 0, :, :]

            elif instance==2:

                image_to_plot = self.__images_I2[idx, 0, :, :]

            fig = plt.figure(figsize=figsize)

            # Check if particles were generated with a buffer:
            if self.__particles.size_buffer == 0:

                plt.imshow(image_to_plot, cmap=cmap, origin=origin, vmin=0, vmax=self.maximum_intensity)

            else:

                if with_buffer:

                    im = plt.imshow(image_to_plot, cmap=cmap, origin=origin, vmin=0, vmax=self.maximum_intensity)

                    # Extend the imshow area with the buffer:
                    f = lambda pixel: pixel - self.__particles.size_buffer
                    im.set_extent([f(x) for x in im.get_extent()])

                    # Visualize a rectangle that separates the proper PIV image area and the artificial buffer outline:
                    rect = patches.Rectangle((-0.5, -0.5), self.__particles.size[1], self.__particles.size[0], linewidth=1, edgecolor='r', facecolor='none')
                    ax = plt.gca()
                    ax.add_patch(rect)

                else:

                    plt.imshow(image_to_plot[self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer], cmap=cmap, origin=origin, vmin=0, vmax=self.maximum_intensity)

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

        if cbar:
            plt.colorbar()

        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')

        return plt

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_image_pair(self,
                        idx,
                        with_buffer=False,
                        xlabel=None,
                        ylabel=None,
                        xticks=True,
                        yticks=True,
                        title=None,
                        cmap='Greys_r',
                        cbar=False,
                        origin='lower',
                        figsize=(5, 5),
                        dpi=300,
                        filename=None):
        """
        Plots a PIV image pair on single, static PIV image by superimposing :math:`I_1 - I_2`.

        **Example:**

        Given the following velocity field magnitude:

        .. image:: ../images/Image_plot_velocity_field_magnitude.png
            :width: 300

        This function will produce the following image:

        .. image:: ../images/Image_plot_image_pair.png
            :width: 300

        :param idx:
            ``int`` specifying the index of the image to plot out of ``n_images`` number of images.
        :param with_buffer: (optional)
            ``bool`` specifying whether the buffer for the image size should be visualized. If set to ``False``, the true PIV image size is visualized. If set to ``True``, the PIV image with a buffer is visualized and buffer outline is marked with a red rectangle.
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
        :param cmap: (optional)
            ``str`` or an object of `matplotlib.colors.ListedColormap <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html>`_ specifying the color map to use.
        :param cbar: (optional)
            ``bool`` specifying whether colorbar should be plotted.
        :param origin: (optional)
            ``str`` specifying the origin location. It can be ``'upper'`` or ``'lower'``.
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

        if not isinstance(xticks, bool):
            raise ValueError("Parameter `xticks` has to be of type 'bool'.")

        if not isinstance(yticks, bool):
            raise ValueError("Parameter `yticks` has to be of type 'bool'.")

        if (title is not None) and (not isinstance(title, str)):
            raise ValueError("Parameter `title` has to be of type 'str'.")

        if not isinstance(cbar, bool):
            raise ValueError("Parameter `cbar` has to be of type 'bool'.")

        if not isinstance(origin, str):
            raise ValueError("Parameter `origin` has to be of type 'str'.")

        check_two_element_tuple(figsize, 'figsize')

        if not isinstance(dpi, int):
            raise ValueError("Parameter `dpi` has to be of type 'int'.")

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if self.images_I1 is None:

            print('Note: Particles have not been added to the image yet!\n\n')

        if self.__images_I2 is None:

            print('Note: Particles have not been advected yet!\n\n')

        image_to_plot = self.__images_I1[idx, 0, :, :] - self.__images_I2[idx, 0, :, :]

        fig = plt.figure(figsize=figsize)

        # Check if particles were generated with a buffer:
        if self.__particles.size_buffer == 0:

            plt.imshow(image_to_plot, cmap=cmap, origin=origin, vmin=0, vmax=self.maximum_intensity)

        else:

            if with_buffer:

                im = plt.imshow(image_to_plot, cmap=cmap, origin=origin, vmin=0, vmax=self.maximum_intensity)

                # Extend the imshow area with the buffer:
                f = lambda pixel: pixel - self.__particles.size_buffer
                im.set_extent([f(x) for x in im.get_extent()])

                # Visualize a rectangle that separates the proper PIV image area and the artificial buffer outline:
                rect = patches.Rectangle((-0.5, -0.5), self.__particles.size[1], self.__particles.size[0], linewidth=1, edgecolor='r', facecolor='none')
                ax = plt.gca()
                ax.add_patch(rect)

            else:

                plt.imshow(image_to_plot[self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer], cmap=cmap, origin=origin, vmin=0, vmax=self.maximum_intensity)


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

        if cbar:
            plt.colorbar()

        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')

        return plt

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def animate_image_pair(self,
                           idx,
                           with_buffer=False,
                           xlabel=None,
                           ylabel=None,
                           title=None,
                           cmap='Greys_r',
                           cbar=False,
                           origin='lower',
                           figsize=(5, 5),
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
        :param cbar: (optional)
            ``bool`` specifying whether colorbar should be plotted.
        :param origin: (optional)
            ``str`` specifying the origin location. It can be ``'upper'`` or ``'lower'``.
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

        if not isinstance(cbar, bool):
            raise ValueError("Parameter `cbar` has to be of type 'bool'.")

        if not isinstance(origin, str):
            raise ValueError("Parameter `origin` has to be of type 'str'.")

        check_two_element_tuple(figsize, 'figsize')

        if not isinstance(dpi, int):
            raise ValueError("Parameter `dpi` has to be of type 'int'.")

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        imagelist = [self.images_I1[idx,0,:,:], self.images_I2[idx,0,:,:]]

        if self.__motion is None:

            print('Note: Motion has not been added to the image yet!\n\n')

        else:

            fig = plt.figure(figsize=figsize)

            # Check if particles were generated with a buffer:
            if self.__particles.size_buffer == 0:

                im = plt.imshow(imagelist[0], cmap=cmap, origin=origin, animated=True)

            else:

                if with_buffer:

                    im = plt.imshow(imagelist[0], cmap=cmap, origin=origin, animated=True)

                    # Extend the imshow area with the buffer:
                    f = lambda pixel: pixel - self.__particles.size_buffer
                    im.set_extent([f(x) for x in im.get_extent()])

                    # Visualize a rectangle that separates the proper PIV image area and the artificial buffer outline:
                    rect = patches.Rectangle((-0.5, -0.5), self.__particles.size[1], self.__particles.size[0], linewidth=1, edgecolor='r', facecolor='none')
                    ax = plt.gca()
                    ax.add_patch(rect)

                else:

                    im = plt.imshow(imagelist[0][self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer], cmap=cmap, origin=origin, animated=True)

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)

        if cbar:
            plt.colorbar()

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

        if filename is not None:
            anim.save(filename, fps=2, bitrate=-1, dpi=dpi, savefig_kwargs={'bbox_inches' : 'tight'})

        return anim

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_field(self,
                   idx,
                   field='velocity',
                   with_buffer=False,
                   xlabel=None,
                   ylabel=None,
                   title=None,
                   cmap='viridis',
                   cbar=False,
                   vmin_vmax=None,
                   origin='lower',
                   figsize=(5, 5),
                   dpi=300,
                   filename=None):
        """
        Plots each component of the velocity or displacement field.

        :param idx:
            ``int`` specifying the index of the velocity/displacement field to plot out of ``n_images`` number of images.
        :param field:
            ``str`` specifying which field should be plotted. It can be ``'velocity'`` or ``'displacement'``.
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
        :param cbar: (optional)
            ``bool`` specifying whether colorbar should be plotted.
        :param vmin_vmax: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) fixed bounds for the colorbar.
        :param origin: (optional)
            ``str`` specifying the origin location. It can be ``'upper'`` or ``'lower'``.
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

        if not isinstance(field, str):
            raise ValueError("Parameter `field` has to be of type 'str'.")

        if field not in ['velocity', 'displacement']:
            raise ValueError("Parameter `field` has to be 'velocity' or 'displacement'.")

        if not isinstance(with_buffer, bool):
            raise ValueError("Parameter `with_buffer` has to be of type 'bool'.")

        if (xlabel is not None) and (not isinstance(xlabel, str)):
            raise ValueError("Parameter `xlabel` has to be of type 'str'.")

        if (ylabel is not None) and (not isinstance(ylabel, str)):
            raise ValueError("Parameter `ylabel` has to be of type 'str'.")

        if (title is not None) and (not isinstance(title, tuple)):
            raise ValueError("Parameter `title` has to be of type 'tuple'.")

        if not isinstance(cbar, bool):
            raise ValueError("Parameter `cbar` has to be of type 'bool'.")

        if vmin_vmax is not None:
            check_two_element_tuple(vmin_vmax, 'vmin_vmax')
            check_min_max_tuple(vmin_vmax)

        if not isinstance(origin, str):
            raise ValueError("Parameter `origin` has to be of type 'str'.")

        check_two_element_tuple(figsize, 'figsize')

        if not isinstance(dpi, int):
            raise ValueError("Parameter `dpi` has to be of type 'int'.")

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        if self.__flowfield is None:
            raise AttributeError("Flow field has not been added to the image yet! Use the `Image.add_flowfield()` method first.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if field == 'velocity':

            quantity_to_plot = self.__flowfield.velocity_field

        elif field == 'displacement':

            self.__flowfield.compute_displacement_field()

            quantity_to_plot = self.__flowfield.displacement_field

        # Plot u-component of velocity:

        fig1 = plt.figure(figsize=figsize)

        # Check if flowfield has been generated with a buffer:
        if self.__flowfield.size_buffer == 0:

            if vmin_vmax is not None:
                plt.imshow(quantity_to_plot[idx, 0, :, :], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin=origin)
            else:
                plt.imshow(quantity_to_plot[idx, 0, :, :], cmap=cmap, origin=origin)

        else:

            if with_buffer:

                if vmin_vmax is not None:
                    im = plt.imshow(quantity_to_plot[idx, 0, :, :], cmap=cmap, vmin=vmin_vmax[0],
                               vmax=vmin_vmax[1], origin=origin)
                else:
                    im = plt.imshow(quantity_to_plot[idx, 0, :, :], cmap=cmap, origin=origin)

                # Extend the imshow area with the buffer:
                f = lambda pixel: pixel - self.__flowfield.size_buffer
                im.set_extent([f(x) for x in im.get_extent()])

                # Visualize a rectangle that separates the proper PIV image area and the artificial buffer outline:
                rect = patches.Rectangle((-0.5, -0.5), self.__flowfield.size[1], self.__flowfield.size[0], linewidth=1, edgecolor='r', facecolor='none')
                ax = plt.gca()
                ax.add_patch(rect)

            else:

                if vmin_vmax is not None:
                    plt.imshow(quantity_to_plot[idx, 0, self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin=origin)
                else:
                    plt.imshow(quantity_to_plot[idx, 0, self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, origin=origin)

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if title is not None:
            plt.title(title[0])

        if cbar:
            plt.colorbar()

        if filename is not None:

            plt.savefig(filename.split('.')[0] + '-u.' + filename.split('.')[1], dpi=dpi, bbox_inches='tight')

        # Plot v-component of velocity:

        fig2 = plt.figure(figsize=figsize)

        # Check if flowfield has been generated with a buffer:
        if self.__flowfield.size_buffer == 0:

            if origin == 'upper':
                if vmin_vmax is not None:
                    plt.imshow(-quantity_to_plot[idx, 1, :, :], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin=origin)
                else:
                    plt.imshow(-quantity_to_plot[idx, 1, :, :], cmap=cmap, origin=origin)
            elif origin == 'lower':
                if vmin_vmax is not None:
                    plt.imshow(quantity_to_plot[idx, 1, :, :], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin=origin)
                else:
                    plt.imshow(quantity_to_plot[idx, 1, :, :], cmap=cmap, origin=origin)

        else:

            if with_buffer:

                if origin == 'upper':

                    if vmin_vmax is not None:
                        im = plt.imshow(-quantity_to_plot[idx, 1, :, :], cmap=cmap, vmin=vmin_vmax[0],
                                   vmax=vmin_vmax[1], origin=origin)
                    else:
                        im = plt.imshow(-quantity_to_plot[idx, 1, :, :], cmap=cmap, origin=origin)

                elif origin == 'lower':

                    if vmin_vmax is not None:
                        im = plt.imshow(quantity_to_plot[idx, 1, :, :], cmap=cmap, vmin=vmin_vmax[0],
                                        vmax=vmin_vmax[1], origin=origin)
                    else:
                        im = plt.imshow(quantity_to_plot[idx, 1, :, :], cmap=cmap, origin=origin)

                # Extend the imshow area with the buffer:
                f = lambda pixel: pixel - self.__flowfield.size_buffer
                im.set_extent([f(x) for x in im.get_extent()])

                # Visualize a rectangle that separates the proper PIV image area and the artificial buffer outline:
                rect = patches.Rectangle((-0.5, -0.5), self.__flowfield.size[1], self.__flowfield.size[0], linewidth=1, edgecolor='r', facecolor='none')
                ax = plt.gca()
                ax.add_patch(rect)

            else:

                if origin == 'upper':

                    if vmin_vmax is not None:
                        plt.imshow(-quantity_to_plot[idx, 1, self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin=origin)
                    else:
                        plt.imshow(-quantity_to_plot[idx, 1, self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, origin=origin)

                elif origin == 'lower':

                    if vmin_vmax is not None:
                        plt.imshow(quantity_to_plot[idx, 1, self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin=origin)
                    else:
                        plt.imshow(quantity_to_plot[idx, 1, self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, origin=origin)

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if title is not None:
            plt.title(title[1])

        if cbar:
            plt.colorbar()

        if filename is not None:
            plt.savefig(filename.split('.')[0] + '-v.' + filename.split('.')[1], dpi=dpi, bbox_inches='tight')

        return fig1, fig2

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_field_magnitude(self,
                             idx,
                             field='velocity',
                             with_buffer=False,
                             xlabel=None,
                             ylabel=None,
                             xticks=True,
                             yticks=True,
                             title=None,
                             cmap='viridis',
                             vmin_vmax=None,
                             cbar=True,
                             cbar_fontsize=14,
                             add_quiver=False,
                             quiver_step=10,
                             quiver_color='k',
                             add_streamplot=False,
                             streamplot_density=1,
                             streamplot_color='k',
                             figsize=(5, 5),
                             dpi=300,
                             filename=None):
        """
        Plots a magnitude of the velocity or displacement field.
        In addition, a vector field can be visualized by setting ``add_quiver=True``,
        and streamlines can be visualized by setting ``add_streamplot=True``.

        :param idx:
            ``int`` specifying the index of the velocity field to plot out of ``n_images`` number of images.
        :param field:
            ``str`` specifying which field should be plotted. It can be ``'velocity'`` or ``'displacement'``.
        :param with_buffer:
            ``bool`` specifying whether the buffer for the image size should be visualized. If set to ``False``, the true PIV image size is visualized. If set to ``True``, the PIV image with a buffer is visualized and buffer outline is marked with a red rectangle.
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
        :param cmap: (optional)
            ``str`` or an object of `matplotlib.colors.ListedColormap <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html>`_ specifying the color map to use.
        :param vmin_vmax: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) fixed bounds for the colorbar.
        :param cbar: (optional)
            ``bool`` specifying whether colorbar should be plotted.
        :param cbar_fontsize: (optional)
            ``int`` specifying the fontsize for the colorbar.
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

        if not isinstance(field, str):
            raise ValueError("Parameter `field` has to be of type 'str'.")

        if field not in ['velocity', 'displacement']:
            raise ValueError("Parameter `field` has to be 'velocity' or 'displacement'.")

        if not isinstance(with_buffer, bool):
            raise ValueError("Parameter `with_buffer` has to be of type 'bool'.")

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

        if vmin_vmax is not None:
            check_two_element_tuple(vmin_vmax, 'vmin_vmax')
            check_min_max_tuple(vmin_vmax)

        if not isinstance(cbar, bool):
            raise ValueError("Parameter `cbar` has to be of type 'bool'.")

        if not isinstance(cbar_fontsize, int):
            raise ValueError("Parameter `cbar_fontsize` has to be of type 'int'.")

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

        if self.__flowfield is None:
            raise AttributeError("Flow field has not been added to the image yet! Use the `Image.add_flowfield()` method first.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if field == 'velocity':

            quantity_to_plot = self.__flowfield.velocity_field_magnitude

        elif field == 'displacement':

            self.__flowfield.compute_displacement_field()

            quantity_to_plot = self.__flowfield.displacement_field_magnitude

        if add_quiver or add_streamplot:

            if field == 'velocity':

                vector_to_plot = self.__flowfield.velocity_field

            elif field == 'displacement':

                vector_to_plot = self.__flowfield.displacement_field

        fig = plt.figure(figsize=figsize)

        # Check if flowfield has been generated with a buffer:
        if self.__flowfield.size_buffer == 0:

            if vmin_vmax is not None:
                plt.imshow(quantity_to_plot[idx, 0, :, :], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin='lower')
            else:
                plt.imshow(quantity_to_plot[idx, 0, :, :], cmap=cmap, origin='lower')

        else:

            if with_buffer:

                if vmin_vmax is not None:
                    im = plt.imshow(quantity_to_plot[idx, 0, :, :], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin='lower')
                else:
                    im = plt.imshow(quantity_to_plot[idx, 0, :, :], cmap=cmap, origin='lower')

                # Extend the imshow area with the buffer:
                f = lambda pixel: pixel - self.__flowfield.size_buffer
                im.set_extent([f(x) for x in im.get_extent()])

                # Visualize a rectangle that separates the proper PIV image area and the artificial buffer outline:
                rect = patches.Rectangle((-0.5, -0.5), self.__flowfield.size[1], self.__flowfield.size[0], linewidth=1, edgecolor='r', facecolor='none')
                ax = plt.gca()
                ax.add_patch(rect)

            else:

                if vmin_vmax is not None:
                    plt.imshow(quantity_to_plot[idx, 0, self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin='lower')
                else:
                    plt.imshow(quantity_to_plot[idx, 0, self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], cmap=cmap, origin='lower')

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

        if cbar:
            colorbar = plt.colorbar()
            for t in colorbar.ax.get_yticklabels():
                t.set_fontsize(cbar_fontsize)

        # Check if flowfield has been generated with a buffer:
        if self.__flowfield.size_buffer == 0:

            if add_quiver:
                X = np.arange(0,self.__flowfield.size[1],quiver_step)
                Y = np.arange(0,self.__flowfield.size[0],quiver_step)
                plt.quiver(X, Y, vector_to_plot[idx, 0, ::quiver_step, ::quiver_step], vector_to_plot[idx, 1, ::quiver_step, ::quiver_step], color=quiver_color)

            if add_streamplot:
                X = np.arange(0,self.__flowfield.size[1],1)
                Y = np.arange(0,self.__flowfield.size[0],1)
                plt.streamplot(X, Y, vector_to_plot[idx, 0, :, :], vector_to_plot[idx, 1, :, :], density=streamplot_density, color=streamplot_color)

        else:

            if with_buffer:

                if add_quiver:
                    X = np.arange(-self.__flowfield.size_buffer, self.__flowfield.size[1] + self.__flowfield.size_buffer, quiver_step)
                    Y = np.arange(-self.__flowfield.size_buffer, self.__flowfield.size[0] + self.__flowfield.size_buffer, quiver_step)

                    plt.quiver(X, Y, vector_to_plot[idx, 0, ::quiver_step, ::quiver_step], vector_to_plot[idx, 1, ::quiver_step, ::quiver_step], color=quiver_color)

                    plt.xlim([-self.__flowfield.size_buffer, self.__flowfield.size[1] + self.__flowfield.size_buffer])
                    plt.ylim([-self.__flowfield.size_buffer, self.__flowfield.size[0] + self.__flowfield.size_buffer])

                if add_streamplot:
                    X = np.arange(-self.__flowfield.size_buffer, self.__flowfield.size[1] + self.__flowfield.size_buffer, 1)
                    Y = np.arange(-self.__flowfield.size_buffer, self.__flowfield.size[0] + self.__flowfield.size_buffer, 1)
                    plt.streamplot(X, Y, vector_to_plot[idx, 0, :, :], vector_to_plot[idx, 1, :, :], density=streamplot_density, color=streamplot_color)

                    plt.xlim([-self.__flowfield.size_buffer, self.__flowfield.size[1] + self.__flowfield.size_buffer])
                    plt.ylim([-self.__flowfield.size_buffer, self.__flowfield.size[0] + self.__flowfield.size_buffer])

            else:

                if add_quiver:
                    X = np.arange(0, self.__flowfield.size[1], quiver_step)
                    Y = np.arange(0, self.__flowfield.size[0], quiver_step)

                    velocity_field_u_subset = vector_to_plot[idx, 0, self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer]
                    velocity_field_v_subset = vector_to_plot[idx, 1, self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer]

                    plt.quiver(X, Y, velocity_field_u_subset[::quiver_step, ::quiver_step], velocity_field_v_subset[::quiver_step, ::quiver_step], color=quiver_color)

                if add_streamplot:
                    X = np.arange(0, self.__flowfield.size[1], 1)
                    Y = np.arange(0, self.__flowfield.size[0], 1)

                    plt.streamplot(X, Y, vector_to_plot[idx, 0, self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer],
                                   vector_to_plot[idx, 1, self.__flowfield.size_buffer:-self.__flowfield.size_buffer, self.__flowfield.size_buffer:-self.__flowfield.size_buffer], density=streamplot_density, color=streamplot_color)

        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')

        return plt

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_image_histogram(self,
                             image,
                             logscale=False,
                             bins=None,
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
        :param bins:
            ``int`` specifying the number of bins on the histogram to generate.
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

        if (bins is not None) and (not isinstance(bins, int)):
            raise ValueError("Parameter `bins` has to be of type 'int'.")

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

        if bins is None:
            plt.hist(image.ravel(), color=color)
        else:
            plt.hist(image.ravel(), bins=bins, color=color)
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
