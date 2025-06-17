import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import chebyu, sph_harm_y
import random
import copy
import time
import scipy
import warnings
from pykitPIV.particle import Particle
from pykitPIV.checks import *

# Specify the available velocity fields in this class:
_available_velocity_fields = {'constant': 'generate_constant_velocity_field',
                              'random smooth': 'generate_random_velocity_field',
                              'sinusoidal': 'generate_sinusoidal_velocity_field',
                              'checkered': 'generate_checkered_velocity_field',
                              'Chebyshev polynomials': 'generate_chebyshev_velocity_field',
                              'spherical harmonics': 'generate_spherical_harmonics_velocity_field',
                              'radial': 'generate_radial_velocity_field',
                              'potential': 'generate_potential_velocity_field',
                              'Langevin': 'generate_langevin_velocity_field'}

########################################################################################################################
########################################################################################################################
####
####    Class: FlowFieldSpecs
####
########################################################################################################################
########################################################################################################################

class FlowFieldSpecs:
    """
    Configuration object for the ``FlowField`` class.

    **Example:**

    .. code:: python

        from pykitPIV import FlowFieldSpecs

        # Instantiate an object of FlowFieldSpecs class:
        flowfield_spec = FlowFieldSpecs()

        # Change one field of flowfield_spec:
        flowfield_spec.gaussian_filters = (4, 6)

        # You can print the current values of all attributes:
        print(flowfield_spec)
    """

    def __init__(self,
                 n_images=1,
                 size=(512, 512),
                 size_buffer=10,
                 time_separation=1,
                 random_seed=None,
                 flowfield_type='random smooth',
                 gaussian_filters=(10, 10),
                 n_gaussian_filter_iter=10,
                 displacement=(2, 2),
                 constant_u_magnitude=(1,4),
                 constant_v_magnitude=(1,4),
                 radial_source=True,
                 radial_imposed_source_location=None,
                 radial_sigma=20,
                 radial_epsilon=1e-6,
                 apply_SLM=False,
                 integral_time_scale=1,
                 sigma=1,
                 n_stochastic_particles=1000000,
                 n_iterations=100,
                 dtype=np.float64):

        self.n_images = n_images
        self.size = size
        self.size_buffer = size_buffer
        self.time_separation = time_separation
        self.random_seed = random_seed
        self.flowfield_type = flowfield_type
        self.gaussian_filters = gaussian_filters
        self.n_gaussian_filter_iter = n_gaussian_filter_iter
        self.displacement = displacement
        self.constant_u_magnitude = constant_u_magnitude
        self.constant_v_magnitude = constant_v_magnitude
        self.radial_source = radial_source
        self.radial_imposed_source_location = radial_imposed_source_location
        self.radial_sigma = radial_sigma
        self.radial_epsilon = radial_epsilon
        self.apply_SLM = apply_SLM
        self.integral_time_scale = integral_time_scale
        self.sigma = sigma
        self.n_stochastic_particles = n_stochastic_particles
        self.n_iterations = n_iterations
        self.dtype = dtype

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"(n_images={self.n_images},\n"
                f"size={self.size},\n"
                f"size_buffer={self.size_buffer},\n"
                f"time_separation={self.time_separation},\n"
                f"random_seed={self.random_seed},\n"
                f"flowfield_type={self.flowfield_type!r},\n"
                f"gaussian_filters={self.gaussian_filters},\n"
                f"n_gaussian_filter_iter={self.n_gaussian_filter_iter},\n"
                f"displacement={self.displacement},\n"
                f"constant_u_magnitude={self.constant_u_magnitude},\n"
                f"constant_v_magnitude={self.constant_v_magnitude},\n"
                f"radial_source={self.radial_source},\n"
                f"radial_imposed_source_location={self.radial_imposed_source_location},\n"
                f"radial_sigma={self.radial_sigma},\n"
                f"radial_epsilon={self.radial_epsilon},\n"
                f"apply_SLM={self.apply_SLM},\n"
                f"integral_time_scale={self.integral_time_scale},\n"
                f"sigma={self.sigma},\n"
                f"n_stochastic_particles={self.n_stochastic_particles},\n"
                f"n_iterations={self.n_iterations},\n"
                f"dtype={self.dtype})"
                )

################################################################################
################################################################################
####
####    Class: FlowField
####
################################################################################
################################################################################

class FlowField:
    """
    Generates or uploads velocity field(s) for a set of ``n_images`` number of PIV image pairs.
    The velocity field(s) can later be used to advect particles between two consecutive PIV images.

    The velocity vector, :math:`\\vec{V} = [u, v]`, is represented as a four-dimensional tensor array
    of size :math:`(N, 2, H, W)`, where the second index corresponds to :math:`u` and :math:`v` velocity component, respectively.

    The velocity magnitude is computed as :math:`|\\vec{V}| = \\sqrt{u^2 + v^2}`
    and is also represented as a four-dimensional tensor array of size :math:`(N, 1, H, W)`.

    The user also specifies the time separation, :math:`\Delta t`,
    between two PIV image pairs with the parameter ``time_separation``.
    With this, the user can populate the corresponding displacement field,
    :math:`d\\vec{\\mathbf{s}} = [dx, dy] = [u \\Delta t, v \\Delta t]`,
    and the displacement field magnitude, :math:`|d\\vec{\\mathbf{s}}|`.

    .. note::

        Only two-dimensional velocity fields are supported at the moment.

    **Example:**

    .. code:: python

        from pykitPIV import FlowField
        import numpy as np

        # We are going to generate 10 flow fields for 10 PIV image pairs:
        n_images = 10

        # Specify size in pixels for each image:
        image_size = (128, 512)

        # Initialize a flow field object:
        flowfield = FlowField(n_images=n_images,
                              size=image_size,
                              size_buffer=10,
                              time_separation=1,
                              dtype=np.float32,
                              random_seed=100)

    :param n_images:
        ``int`` specifying the number of PIV image pairs, :math:`N`, to create.
    :param size: (optional)
        ``tuple`` of two ``int`` elements specifying the size of each image in pixels. The first number is image height, :math:`h`, the second number is image width, :math:`w`.
    :param size_buffer: (optional)
        ``int`` specifying the buffer, :math:`b`, in pixels :math:`[\\text{px}]` to add to the image size in the width and height direction.
        This number should be approximately equal to the maximum displacement that particles are subject to in order to allow for new particles to arrive into the image area.
    :param time_separation: (optional)
        ``float`` or ``int`` specifying the time separation, :math:`\Delta t`, between two consecutive PIV images.
    :param dtype: (optional)
        ``numpy.dtype`` specifying the data type for produced vector fields. To reduce memory, you can switch from the
        default ``numpy.float64`` to ``numpy.float32``.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``.
        If specified, all operations are reproducible.

    **Attributes:**

    - **n_images** - (read-only) as per user input.
    - **size** - (read-only) as per user input.
    - **size_buffer** - (read-only) as per user input.
    - **time_separation** - (can be re-set) as per user input.
    - **dtype** - (read-only) as per user input.
    - **random_seed** - (read-only) as per user input.
    - **size_with_buffer** - (read-only) ``tuple`` specifying the size of each image in pixels with buffer added.
    - **displacement** - (read-only) ``tuple`` of two numerical elements specifying the minimum (first element)
      and maximum (second element) maximum displacement on each image.
      This attribute becomes available after calling one of the velocity field generators.
      The assumption is that this displacement value corresponds to time being unity.
    - **displacement_per_image** - (read-only) ``numpy.ndarray`` specifying the template for the maximum displacements
      per each image. Template displacements are random numbers between minimum and maximum displacement.
      This attribute becomes available after calling one of the velocity field generators.
      The assumption is that this displacement value corresponds to time being unity.
    - **velocity_field** - (read-only) ``numpy.ndarray`` specifying the velocity vector, :math:`\\vec{V} = [u, v]`,
      per each image. It has size :math:`(N, 2, H+2b, W+2b)`.
      The second index corresponds to :math:`u` and :math:`v` velocity component, respectively.
      This attribute becomes available after calling one of the velocity field generators.
    - **velocity_field_magnitude** - (read-only) ``numpy.ndarray`` specifying the velocity field magnitude,
      :math:`|\\vec{V}| = \\sqrt{u^2 + v^2}`, per each image. It has size :math:`(N, 1, H+2b, W+2b)`.
      This attribute becomes available after calling one of the velocity field generators.
    - **displacement_field** - (read-only) ``numpy.ndarray`` specifying the displacement field, :math:`d\\vec{\\mathbf{s}} = [dx, dy]`, in the :math:`x` and :math:`y` direction. It is computed as the velocity component multiplied by time separation and has a unit of :math:`\\text{px}`. It has size :math:`(N, 2, H+2b, W+2b)`. The second index corresponds to :math:`dx` and :math:`dy` displacement, respectively.
    - **displacement_field_magnitude** - (read-only) ``numpy.ndarray`` specifying the displacement field magnitude, :math:`|d\\vec{\\mathbf{s}}| = \\sqrt{dx^2 + dy^2}`. It has a unit of :math:`\\text{px}`. It has size :math:`(N, 1, H+2b, W+2b)`.
    """

    def __init__(self,
                 n_images,
                 size=(512, 512),
                 size_buffer=10,
                 time_separation=1,
                 dtype=np.float64,
                 random_seed=None):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if type(n_images) != int:
            raise ValueError("Parameter `n_images` has to be of type 'int'.")

        if n_images <= 0:
            raise ValueError("Parameter `n_images` has to be positive. At least one image has to be generated.")

        check_two_element_tuple(size, 'size')

        if not isinstance(size_buffer, int):
            raise ValueError("Parameter `size_buffer` has to be of type 'int'.")

        if size_buffer < 0:
            raise ValueError("Parameter `size_buffer` has to non-negative.")

        if (not isinstance(time_separation, float)) and (not isinstance(time_separation, int)):
            raise ValueError("Parameter `time_separation` has to be of type `float` or `int`.")

        if time_separation <= 0:
            raise ValueError("Parameter `time_separation` has to be a non-zero, positive number.")

        if not isinstance(dtype, type):
            raise ValueError("Parameter `dtype` has to be of type 'type'.")

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
        self.__time_separation = time_separation
        self.__dtype = dtype
        self.__random_seed = random_seed

        # Compute the image outline that serves as a buffer:
        self.__height_with_buffer = self.size[0] + 2 * self.size_buffer
        self.__width_with_buffer = self.size[1] + 2 * self.size_buffer
        self.__size_with_buffer = (self.__height_with_buffer, self.__width_with_buffer)

        # Initialize displacements per image:
        self.__displacement = None
        self.__displacement_per_image = None

        # Initialize the velocity field and its magnitude:
        self.__velocity_field = None
        self.__velocity_field_magnitude = None

        # Initialize the displacement field and its magnitude:
        self.__displacement_field = None
        self.__displacement_field_magnitude = None

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
    def time_separation(self):
        return self.__time_separation

    @property
    def dtype(self):
        return self.__dtype

    @property
    def random_seed(self):
        return self.__random_seed

    # Properties computed at class init:
    @property
    def size_with_buffer(self):
        return self.__size_with_buffer

    @property
    def displacement(self):
        return self.__displacement

    @property
    def displacement_per_image(self):
        return self.__displacement_per_image

    @property
    def velocity_field(self):
        return self.__velocity_field

    @property
    def velocity_field_magnitude(self):
        return self.__velocity_field_magnitude

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

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def print_available_fields(self):
        """
        Prints the available velocity fields and points to the function from ``FlowField`` that generates them.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # Initialize a flow field object:
            flowfield = FlowField(n_images=10)

            # Print the available velocity fields:
            flowfield.print_available_fields()

        The above will print:

        .. code-block:: text

            Velocity fields available in pykitPIV:

            - constant
                Use function: generate_constant_velocity_field

            - random smooth
                Use function: generate_random_velocity_field

            - sinusoidal
                Use function: generate_sinusoidal_velocity_field

            - checkered
                Use function: generate_checkered_velocity_field

            - Chebyshev
                Use function: generate_chebyshev_velocity_field

            - spherical harmonics
                Use function: generate_spherical_harmonics_velocity_field

            - radial
                Use function: generate_radial_velocity_field

            - potential
                Use function: generate_potential_velocity_field

            - Langevin
                Use function: generate_langevin_velocity_field
        """

        print('Velocity fields available in pykitPIV:\n')

        for key, value in _available_velocity_fields.items():

            print('- ' + key)
            print('\tUse function: ' + value)
            print()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_constant_velocity_field(self,
                                         u_magnitude=(1, 4),
                                         v_magnitude=(1, 4)):
        """
        Generates a constant velocity field.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128, 128)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Generate random velocity field:
            flowfield.generate_constant_velocity_field(u_magnitude=(1, 4),
                                                       v_magnitude=(1, 4))

            # Access the velocity components tensor:
            flowfield.velocity_field

            # Access the velocity field magnitude:
            flowfield.velocity_field_magnitude

        :param u_magnitude: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
            magnitude of the :math:`u` component of velocity.
        :param v_magnitude: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
            magnitude of the :math:`v` component of velocity.
       """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        check_two_element_tuple(u_magnitude, 'u_magnitude')
        check_min_max_tuple(u_magnitude, 'u_magnitude')
        check_two_element_tuple(v_magnitude, 'v_magnitude')
        check_min_max_tuple(v_magnitude, 'v_magnitude')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__u_magnitude = u_magnitude
        self.__v_magnitude = v_magnitude

        self.__velocity_field = np.ones((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)
        self.__velocity_field_magnitude = np.zeros((self.__n_images, 1, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)

        self.__u_magnitude_per_image = np.random.rand(self.__n_images) * (self.__u_magnitude[1] - self.__u_magnitude[0]) + self.__u_magnitude[0]
        self.__v_magnitude_per_image = np.random.rand(self.__n_images) * (self.__v_magnitude[1] - self.__v_magnitude[0]) + self.__v_magnitude[0]

        for i in range(0, self.n_images):

            self.__velocity_field[i, 0, :, :] = self.__velocity_field[i, 0, :, :] * self.__u_magnitude_per_image[i]
            self.__velocity_field[i, 1, :, :] = self.__velocity_field[i, 1, :, :] * self.__v_magnitude_per_image[i]

            self.__velocity_field_magnitude[i, 0, :, :] = np.sqrt(self.__velocity_field[i, 0, :, :] ** 2 + self.__velocity_field[i, 1, :, :] ** 2)

        # Compute the updated displacement per image:
        self.__displacement_per_image = np.max(np.abs(self.__velocity_field_magnitude), axis=(2,3)).ravel()

        # Compute the updated minimum and maximum displacement:
        self.__displacement = (np.min(self.__displacement_per_image), np.max(self.__displacement_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_random_velocity_field(self,
                                       displacement=(0, 10),
                                       gaussian_filters=(10, 30),
                                       n_gaussian_filter_iter=6):
        """
        Generates random velocity field by smoothing a random initialization of pixels with a series of Gaussian filters.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128, 512)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Generate random velocity field:
            flowfield.generate_random_velocity_field(displacement=(0, 10),
                                                     gaussian_filters=(10, 30),
                                                     n_gaussian_filter_iter=6)

            # Access the velocity components tensor:
            flowfield.velocity_field

            # Access the velocity field magnitude:
            flowfield.velocity_field_magnitude

        .. image:: ../images/FlowField-setting-spectrum.png
            :width: 800

        :param displacement: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
            displacement.
        :param gaussian_filters: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
            Gaussian filter size (bandwidth) for smoothing out the random velocity fields to randomly sample from.
        :param n_gaussian_filter_iter: (optional)
            ``int`` specifying the number of iterations applying a Gaussian filter to the random velocity field to
            eventually arrive at a smoothed velocity map. With no iterations,
            each pixel attains a random velocity component value.
       """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        check_two_element_tuple(displacement, 'displacement')
        check_min_max_tuple(displacement, 'displacement')
        check_two_element_tuple(gaussian_filters, 'gaussian_filters')
        check_min_max_tuple(gaussian_filters, 'gaussian_filters')

        if type(n_gaussian_filter_iter) != int:
            raise ValueError("Parameter `n_gaussian_filter_iter` has to be of type 'int'.")

        if n_gaussian_filter_iter < 0:
            raise ValueError("Parameter `n_gaussian_filter_iter` has to be positive.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__displacement = displacement
        self.__displacement_per_image = np.random.rand(self.__n_images) * (self.__displacement[1] - self.__displacement[0]) + self.__displacement[0]

        self.__gaussian_filters = gaussian_filters
        self.__gaussian_filter_per_image = np.random.rand(self.__n_images) * (self.__gaussian_filters[1] - self.__gaussian_filters[0]) + self.__gaussian_filters[0]

        self.__n_gaussian_filter_iter = n_gaussian_filter_iter

        self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)
        self.__velocity_field_magnitude = np.zeros((self.__n_images, 1, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)

        for i in range(0, self.n_images):

            velocity_field_u = np.random.rand(self.size_with_buffer[0], self.size_with_buffer[1])
            velocity_field_v = np.random.rand(self.size_with_buffer[0], self.size_with_buffer[1])

            # Smooth out the random velocity field `n_gaussian_filter_iter` times:
            for _ in range(0, n_gaussian_filter_iter):
                velocity_field_u = scipy.ndimage.gaussian_filter(velocity_field_u, self.__gaussian_filter_per_image[i])
                velocity_field_v = scipy.ndimage.gaussian_filter(velocity_field_v, self.__gaussian_filter_per_image[i])

            velocity_field_u = velocity_field_u - np.mean(velocity_field_u)
            velocity_field_v = velocity_field_v - np.mean(velocity_field_v)

            velocity_magnitude = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            velocity_magnitude_scale = self.__displacement_per_image[i] / np.max(velocity_magnitude)

            velocity_field_u = velocity_magnitude_scale * velocity_field_u
            velocity_field_v = velocity_magnitude_scale * velocity_field_v

            self.__velocity_field_magnitude[i, 0, :, :] = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            self.__velocity_field[i, 0, :, :] = velocity_field_u
            self.__velocity_field[i, 1, :, :] = velocity_field_v

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_sinusoidal_velocity_field(self,
                                           amplitudes=(2, 4),
                                           wavelengths=(8, 128),
                                           components='both'):
        """
        Generates a sinusoidal velocity field as per
        `Scarano and Riethmuller (2000) <https://link.springer.com/article/10.1007/s003480070007>`_.
        Each velocity component is computed as:

        .. math::

            u(w) = A \\sin(2 \pi w / \Lambda)

        .. math::

            v(h) = A \\sin(2 \pi h / \Lambda)

        where :math:`A` is the amplitude in pixels :math:`[\\text{px}]` and :math:`\Lambda` is the wavelength in pixels :math:`[\\text{px}]`.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128, 512)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Generate sinusoidal velocity field:
            flowfield.generate_sinusoidal_velocity_field(amplitudes=(2, 4),
                                                         wavelengths=(20, 40),
                                                         components='u')

            # Access the velocity components tensor:
            flowfield.velocity_field

            # Access the velocity field magnitude:
            flowfield.velocity_field_magnitude

        :param amplitudes: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
            amplitude, :math:`A`, in pixels :math:`[\\text{px}]`.
        :param wavelengths: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
            wavelength, :math:`\Lambda`, in pixels :math:`[\\text{px}]`.
        :param components: (optional)
            ``str`` specifying which velocity components should be generated. It can be one of the following:
            ``'both'``, ``'u'``, or ``'v'``. When only one component is generated, the other one is zero.
       """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        check_two_element_tuple(amplitudes, 'amplitudes')
        check_min_max_tuple(amplitudes, 'amplitudes')

        check_two_element_tuple(wavelengths, 'wavelengths')
        check_min_max_tuple(wavelengths, 'wavelengths')

        __components = ['both', 'u', 'v']

        if not isinstance(components, str):
            raise ValueError("Parameter `components` has to be of type 'str'.")

        if components not in __components:
            raise ValueError("Parameter `components` has to be 'both', or 'u', or 'v'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__amplitudes = amplitudes
        self.__amplitudes_per_image = np.random.rand(self.__n_images) * (self.__amplitudes[1] - self.__amplitudes[0]) + self.__amplitudes[0]

        self.__wavelengths = wavelengths
        self.__wavelengths_per_image = np.random.rand(self.__n_images) * (self.__wavelengths[1] - self.__wavelengths[0]) + self.__wavelengths[0]

        self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)
        self.__velocity_field_magnitude = np.zeros((self.__n_images, 1, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)

        h = np.linspace(0, self.size_with_buffer[0], self.size_with_buffer[0])
        w = np.linspace(0, self.size_with_buffer[1], self.size_with_buffer[1])

        (grid_w, grid_h) = np.meshgrid(w, h)

        for i in range(0, self.n_images):

            if components == 'both':
                velocity_field_u = self.__amplitudes_per_image[i] * np.sin(2 * np.pi * grid_w / self.__wavelengths_per_image[i])
                velocity_field_v = self.__amplitudes_per_image[i] * np.sin(2 * np.pi * grid_h / self.__wavelengths_per_image[i])

            elif components == 'u':
                velocity_field_u = self.__amplitudes_per_image[i] * np.sin(2 * np.pi * grid_w / self.__wavelengths_per_image[i])
                velocity_field_v = np.zeros((self.size_with_buffer[0], self.size_with_buffer[1]))

            elif components == 'v':
                velocity_field_u = np.zeros((self.size_with_buffer[0], self.size_with_buffer[1]))
                velocity_field_v = self.__amplitudes_per_image[i] * np.sin(2 * np.pi * grid_h / self.__wavelengths_per_image[i])

            self.__velocity_field_magnitude[i, 0, :, :] = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            self.__velocity_field[i, 0, :, :] = velocity_field_u
            self.__velocity_field[i, 1, :, :] = velocity_field_v

        # Compute the updated displacement per image:
        self.__displacement_per_image = np.max(np.abs(self.__velocity_field_magnitude), axis=(2,3)).ravel()

        # Compute the updated minimum and maximum displacement:
        self.__displacement = (np.min(self.__displacement_per_image), np.max(self.__displacement_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_checkered_velocity_field(self,
                                          displacement=(1, 10),
                                          m=10,
                                          n=10,
                                          rotation=None):
        """
        Generates a checkered velocity field. Each velocity component is computed as:

        .. math::

            u(h, w), v(h, w) = \\sin(m \cdot h) \cdot \\cos(n \cdot w)

        where :math:`m` and :math:`n` free parameters.

        Optionally, the checkered pattern can be rotated with the additional rotation term:

        .. math::

            u(h, w), v(h, w) = \\sin(m \cdot h) \cdot \\cos(n \cdot w) \cdot \\sin(r \cdot (h + w))

        were :math:`r` is a free parameter.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128,512)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Generate checkered velocity field:
            flowfield.generate_checkered_velocity_field(displacement=(1, 10),
                                                        m=10,
                                                        n=10,
                                                        rotation=10)

            # Access the velocity components tensor:
            flowfield.velocity_field

            # Access the velocity field magnitude:
            flowfield.velocity_field_magnitude

        :param displacement: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) displacement
        :param m: (optional)
            ``int`` or ``float`` specifying the scaling for the sine function, :math:`m`.
        :param n: (optional)
            ``int`` or ``float`` specifying the scaling for the cosine function, :math:`n`.
        :param rotation: (optional)
            ``int`` or ``float`` specifying the scaling for the rotation with the sine function, :math:`r`. If set to ``None``,
            the checkered pattern will not be rotated.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        check_two_element_tuple(displacement, 'displacement')
        check_min_max_tuple(displacement, 'displacement')

        if (not isinstance(m, int)) and (not isinstance(m, float)):
            raise ValueError("Parameter `m` has to be of type 'int' or 'float'.")

        if (not isinstance(n, int)) and (not isinstance(n, float)):
            raise ValueError("Parameter `n` has to be of type 'int' or 'float'.")

        if rotation is not None:
            if (not isinstance(rotation, int)) and (not isinstance(rotation, float)):
                raise ValueError("Parameter `rotation` has to be of type 'int' or 'float'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__displacement = displacement
        self.__displacement_per_image = np.random.rand(self.__n_images) * (self.__displacement[1] - self.__displacement[0]) + self.__displacement[0]

        self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)
        self.__velocity_field_magnitude = np.zeros((self.__n_images, 1, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)

        h = np.linspace(-1, 1, self.size_with_buffer[0])
        w = np.linspace(-1, 1, self.size_with_buffer[1])

        (grid_w, grid_h) = np.meshgrid(w, h)

        for i in range(0, self.n_images):

            if rotation is None:
                velocity_field_u = np.sin(m * grid_w) * np.cos(n * grid_h)
                velocity_field_v = np.sin(m * grid_w) * np.cos(n * grid_h)
            else:
                velocity_field_u = np.sin(m * grid_w) * np.cos(n * grid_h) * np.sin(rotation * (grid_w + grid_h))
                velocity_field_v = np.sin(m * grid_w) * np.cos(n * grid_h) * np.sin(rotation * (grid_w + grid_h))

            velocity_field_u = velocity_field_u - np.mean(velocity_field_u)
            velocity_field_v = velocity_field_v - np.mean(velocity_field_v)

            velocity_magnitude = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            velocity_magnitude_scale = self.__displacement_per_image[i] / np.max(velocity_magnitude)

            velocity_field_u = velocity_magnitude_scale * velocity_field_u
            velocity_field_v = velocity_magnitude_scale * velocity_field_v

            self.__velocity_field_magnitude[i, 0, :, :] = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            self.__velocity_field[i, 0, :, :] = velocity_field_u
            self.__velocity_field[i, 1, :, :] = velocity_field_v

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_chebyshev_velocity_field(self,
                                          displacement=(1, 10),
                                          start=0.3,
                                          stop=0.8,
                                          order=10):
        """
        Generates a velocity field using Chebyshev polynomials of the second kind. Each velocity component is computed as:

        .. math::

            u(h, w), v(h, w) = U_n(h) U_n(w) \\sin(n(h+w))

        where :math:`U_n` is the Chebyshev polynomial of the second kind and of order :math:`n`.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128, 512)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Generate Chebyshev velocity field:
            flowfield.generate_chebyshev_velocity_field(displacement=(0, 10),
                                                        start=0.3,
                                                        stop=0.8,
                                                        order=10)

            # Access the velocity components tensor:
            flowfield.velocity_field

            # Access the velocity field magnitude:
            flowfield.velocity_field_magnitude

        :param displacement: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) displacement.
        :param start: (optional)
            ``int`` or ``float`` specifying the start value for generating the coordinate grid.
        :param stop: (optional)
            ``int`` or ``float`` specifying the end value for generating the coordinate grid.
        :param order: (optional)
            ``int`` specifying the order of the Chebyshev polynomial.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        check_two_element_tuple(displacement, 'displacement')
        check_min_max_tuple(displacement, 'displacement')

        if (not isinstance(start, int)) and (not isinstance(start, float)):
            raise ValueError("Parameter `start` has to be of type 'int' or 'float'.")

        if (not isinstance(stop, int)) and (not isinstance(stop, float)):
            raise ValueError("Parameter `stop` has to be of type 'int' or 'float'.")

        if (not isinstance(order, int)):
            raise ValueError("Parameter `order` has to be of type 'int'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__displacement = displacement
        self.__displacement_per_image = np.random.rand(self.__n_images) * (self.__displacement[1] - self.__displacement[0]) + self.__displacement[0]

        self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)
        self.__velocity_field_magnitude = np.zeros((self.__n_images, 1, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)

        h = np.linspace(start, stop, self.size_with_buffer[0])
        w = np.linspace(start, stop, self.size_with_buffer[1])
        (grid_w, grid_h) = np.meshgrid(w, h)

        # Generate Chebyshev polynomials:
        un = chebyu(order)

        for i in range(0, self.n_images):

            # Generate a 2D field:
            velocity_field_u = un(grid_h) * un(grid_w) * np.sin((grid_h + grid_w) * order)
            velocity_field_v = un(grid_h) * un(grid_w) * np.sin((grid_h + grid_w) * order)

            velocity_magnitude = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            velocity_magnitude_scale = self.__displacement_per_image[i] / np.max(velocity_magnitude)

            velocity_field_u = velocity_magnitude_scale * velocity_field_u
            velocity_field_v = velocity_magnitude_scale * velocity_field_v

            self.__velocity_field_magnitude[i, 0, :, :] = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            self.__velocity_field[i, 0, :, :] = velocity_field_u
            self.__velocity_field[i, 1, :, :] = velocity_field_v

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_spherical_harmonics_velocity_field(self,
                                                    displacement=(1, 10),
                                                    start=0.3,
                                                    stop=0.8,
                                                    order=1,
                                                    degree=1):
        """
        Generates a velocity field using spherical harmonics. Each velocity component is computed as:

        .. math::

            u(h, w), v(h, w) = Y_n^k \\sin(n(h+w))

        where :math:`Y_n^k` is the spherical harmonics function of order :math:`n` and degree :math:`k`.

        .. note::

            This function uses
            `scipy.special.sph_harm_y <https://docs.scipy.org/doc//scipy/reference/generated/scipy.special.sph_harm_y.html>`_.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128, 512)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Generate spherical harmonics velocity field:
            flowfield.generate_spherical_harmonics_velocity_field(displacement=(0, 10),
                                                                  start=0.3,
                                                                  stop=0.8,
                                                                  order=1,
                                                                  degree=1)

            # Access the velocity components tensor:
            flowfield.velocity_field

            # Access the velocity field magnitude:
            flowfield.velocity_field_magnitude

        :param displacement: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) displacement.
        :param start: (optional)
            ``int`` or ``float`` specifying the start value for generating the azimuthal/polar coordinate grid.
        :param stop: (optional)
            ``int`` or ``float`` specifying the end value for generating the azimuthal/polar coordinate grid.
        :param order: (optional)
            ``int`` specifying the order, :math:`n`, of the spherical harmonics.
        :param degree: (optional)
            ``int`` specifying the degree, :math:`k`, of the spherical harmonics.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        check_two_element_tuple(displacement, 'displacement')
        check_min_max_tuple(displacement, 'displacement')

        if (not isinstance(start, int)) and (not isinstance(start, float)):
            raise ValueError("Parameter `start` has to be of type 'int' or 'float'.")

        if (not isinstance(stop, int)) and (not isinstance(stop, float)):
            raise ValueError("Parameter `stop` has to be of type 'int' or 'float'.")

        if (not isinstance(order, int)):
            raise ValueError("Parameter `order` has to be of type 'int'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__displacement = displacement
        self.__displacement_per_image = np.random.rand(self.__n_images) * (self.__displacement[1] - self.__displacement[0]) + self.__displacement[0]

        self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)
        self.__velocity_field_magnitude = np.zeros((self.__n_images, 1, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)

        h = np.linspace(start, stop, self.size_with_buffer[0])
        w = np.linspace(start, stop, self.size_with_buffer[1])

        # Rescale the azimuthal coordinate:
        h = (h - h.min()) / (h.max() - h.min()) * np.pi * 0.2 + np.pi * 0.4

        # Rescale the polar coordinate:
        w = (w - w.min()) / (w.max() - w.min()) * np.pi * 0.2 / 6 * 5 + (np.pi * 0.5 - np.pi * 0.2 / 6 * 5 / 2)

        (grid_w, grid_h) = np.meshgrid(w, h)

        for i in range(0, self.n_images):

            # Generate a 2D field:
            velocity_field_u = np.real(sph_harm_y(order, degree, grid_h, grid_w)) * np.sin((grid_h + grid_w) * degree)
            velocity_field_v = np.real(sph_harm_y(order, degree, grid_h, grid_w)) * np.sin((grid_h + grid_w) * degree)

            velocity_magnitude = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            velocity_magnitude_scale = self.__displacement_per_image[i] / np.max(velocity_magnitude)

            velocity_field_u = velocity_magnitude_scale * velocity_field_u
            velocity_field_v = velocity_magnitude_scale * velocity_field_v

            self.__velocity_field_magnitude[i, 0, :, :] = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            self.__velocity_field[i, 0, :, :] = velocity_field_u
            self.__velocity_field[i, 1, :, :] = velocity_field_v

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_radial_velocity_field(self,
                                       source=True,
                                       displacement=(1, 4),
                                       imposed_source_location=None,
                                       sigma=20.0,
                                       epsilon=1e-6):
        """
        Generates a radial velocity field that has a single source or sink.
        The velocity components are computed as:

        .. math::

            u(h, w) = \\Big( 1 - \\exp{ \\big( \\frac{-r^2}{2 \\sigma^2} \\big)} \\Big) \\frac{(w - w_s)}{r + \\varepsilon}

        .. math::

            v(h, w) = \\Big( 1 - \\exp{ \\big( \\frac{-r^2}{2 \\sigma^2} \\big)} \\Big) \\frac{(h - h_s)}{r + \\varepsilon}

        where :math:`\\sigma` is the size of source/sink, :math:`r` is the radial distance from the source/sink computed as:

        .. math::

            r = \\sqrt{((w - w_s)^2 + (h - h_s)^2)}

        and where :math:`w_s` and :math:`h_s` are the location of the source/sink.

        :math:`\\varepsilon` is a small value added to avoid dividing by zero.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (200, 200)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Generate radial velocity field:
            flowfield.generate_radial_velocity_field(source=True,
                                                     displacement=(1, 4),
                                                     imposed_source_location=(50, 50),
                                                     sigma=5.0,
                                                     epsilon=1e-6)

            # Access the velocity components tensor:
            flowfield.velocity_field

            # Access the velocity field magnitude:
            flowfield.velocity_field_magnitude

        :param source: (optional)
            ``bool`` specifying generation of the source (``True``) or sink (``False``).
        :param displacement: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
            displacement.
        :param imposed_source_location: (optional)
            ``tuple`` specifying the user-imposed source location, :math:`(h_s, w_s)`.
            Note that you need to account for the buffer size when specifying the values :math:`(h_s, w_s)`.
        :param sigma: (optional)
            ``int`` or ``float`` specifying the size of the source/sink, :math:`\\sigma`.
        :param epsilon: (optional)
            ``float`` specifying the :math:`\\varepsilon`.
       """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:
        check_two_element_tuple(displacement, 'displacement')
        check_min_max_tuple(displacement, 'displacement')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__displacement = displacement
        self.__displacement_per_image = np.random.rand(self.__n_images) * (self.__displacement[1] - self.__displacement[0]) + self.__displacement[0]

        self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)
        self.__velocity_field_magnitude = np.zeros((self.__n_images, 1, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)

        h = np.linspace(0, self.size_with_buffer[0], self.size_with_buffer[0])
        w = np.linspace(0, self.size_with_buffer[1], self.size_with_buffer[1])

        (grid_w, grid_h) = np.meshgrid(w, h)

        if imposed_source_location is not None:
            source_h, source_w = imposed_source_location

        for i in range(0, self.n_images):

            if imposed_source_location is None:
                source_h = np.random.rand(1) * (self.size_with_buffer[0] - 1)
                source_w = np.random.rand(1) * (self.size_with_buffer[1] - 1)

            # Vector from source to each point:
            dx = grid_w - source_w
            dy = grid_h - source_h

            # Radial distance:
            r = np.sqrt(dx ** 2 + dy ** 2)

            outflow_magnitude = 1.0 - np.exp(- (r ** 2) / (2.0 * sigma ** 2))

            velocity_field_u = outflow_magnitude * dx / (r + epsilon)
            velocity_field_v = outflow_magnitude * dy / (r + epsilon)

            velocity_magnitude = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            velocity_magnitude_scale = self.__displacement_per_image[i] / np.max(velocity_magnitude)

            velocity_field_u = velocity_magnitude_scale * velocity_field_u
            velocity_field_v = velocity_magnitude_scale * velocity_field_v

            self.__velocity_field_magnitude[i, 0, :, :] = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)

            if source:
                self.__velocity_field[i, 0, :, :] = velocity_field_u
                self.__velocity_field[i, 1, :, :] = velocity_field_v
            else:
                self.__velocity_field[i, 0, :, :] = -velocity_field_u
                self.__velocity_field[i, 1, :, :] = -velocity_field_v

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_potential_velocity_field(self,
                                          imposed_origin=(0, 0),
                                          displacement=(1, 4)):
        """
        Generates a potential velocity field of the form:

        .. math::

            u(h, w) = 2 (w - w_o)

        .. math::

            v(h, w) = - 2 (h - h_o)

        where :math:`(h_o, w_o)` is the origin location on the PIV image.

        This potential flow is irrotational and divergence-free and can therefore be used for generating BOS images.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128, 512)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Generate sinusoidal velocity field:
            flowfield.generate_potential_velocity_field(imposed_origin=None,
                                                        displacement=(2, 2))

            # Access the velocity components tensor:
            flowfield.velocity_field

            # Access the velocity field magnitude:
            flowfield.velocity_field_magnitude

        :param imposed_origin: (optional)
            ``tuple`` specifying the user-imposed origin location, :math:`(h_o, w_o)`.
            Note that you need to account for the buffer size when specifying the values :math:`(h_o, w_o)`.
            If set to ``None``, the origin location will be randomized in each image.
        :param displacement: (optional)
            ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
            displacement.
       """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if imposed_origin is not None:
            check_two_element_tuple(imposed_origin, 'imposed_origin')

        check_two_element_tuple(displacement, 'displacement')
        check_min_max_tuple(displacement, 'displacement')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__displacement = displacement
        self.__displacement_per_image = np.random.rand(self.__n_images) * (self.__displacement[1] - self.__displacement[0]) + self.__displacement[0]

        self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)
        self.__velocity_field_magnitude = np.zeros((self.__n_images, 1, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)

        h = np.linspace(0, self.size_with_buffer[0], self.size_with_buffer[0])
        w = np.linspace(0, self.size_with_buffer[1], self.size_with_buffer[1])

        (grid_w, grid_h) = np.meshgrid(w, h)

        if imposed_origin is not None:
            origin_h, origin_w = imposed_origin

        for i in range(0, self.n_images):

            if imposed_origin is None:
                origin_h = np.random.rand(1) * (self.size_with_buffer[0] - 1)
                origin_w = np.random.rand(1) * (self.size_with_buffer[1] - 1)

            # Vector from source to each point:
            dx = grid_w - origin_w
            dy = grid_h - origin_h

            # Potential flow:
            velocity_field_u = 2 * dx
            velocity_field_v = - 2 * dy

            velocity_magnitude = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            velocity_magnitude_scale = self.__displacement_per_image[i] / np.max(velocity_magnitude)

            velocity_field_u = velocity_magnitude_scale * velocity_field_u
            velocity_field_v = velocity_magnitude_scale * velocity_field_v

            self.__velocity_field_magnitude[i, 0, :, :] = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)

            self.__velocity_field[i, 0, :, :] = velocity_field_u
            self.__velocity_field[i, 1, :, :] = velocity_field_v

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def generate_langevin_velocity_field(self,
                                         mean_field,
                                         integral_time_scale=1,
                                         sigma=1,
                                         n_stochastic_particles=10000,
                                         n_iterations=100,
                                         verbose=False):
        """
        Generates a velocity field using the simplified Langevin model (SLM) by solving the following stochastic
        partial differential equation:

        .. math::

            dU^*(t) = - \\frac{1}{T_L} U^*(t) dt + \\sqrt{\\frac{1}{T_L} 2 \\sigma^2 }  d W(t)

        where :math:`U^*(t)` is one component of particle's velocity, :math:`T_L` is the Lagrangian integral time scale,
        :math:`\sigma` is the turbulent kinetic energy, and :math:`W(t)` is the stochastic variable.

        In essence, the SLM model simulates stationary isotropic turbulence in the Lagrangian sense, where stochastic
        particles are allowed to drift and carry fluid velocity to a different location.

        In practice, we solve the following finite-difference set of equations for both components of velocity:

        .. math::

            u^*(t + \\Delta t) = u^*(t) - (u^*(t) - u_m^*) \\frac{\\Delta t}{T_L} + \\sqrt{\\frac{1}{T_L} 2 \\sigma^2 \\Delta t}  \\xi(t)

            x(t + \\Delta t) = x(t) + u^*(t) \\Delta t

        .. math::

            v^*(t + \\Delta t) = v^*(t) - (v^*(t) - v_m^*) \\frac{\\Delta t}{T_L} + \\sqrt{\\frac{1}{T_L} 2 \\sigma^2 \\Delta t}  \\xi(t)

            y(t + \\Delta t) = y(t) + v^*(t) \\Delta t

        where :math:`u_m^*` and :math:`v_m^*` are the components of the mean velocity,
        :math:`\\xi(t)` is a random variable :math:`\\xi(t) \in \mathcal{N}(0,1)`,
        and :math:`x(t)` and :math:`y(t)` are coordinates of the stochastic particles.

        The above equations define the update to velocity components and particle positions over :math:`n` iterations
        (set by the parameter ``n_iterations``). :math:`\\Delta t` is then computed as:

        .. math::

            \\Delta t = \\frac{T_L}{n}

        Next, a per-pixel velocity is found as an average over the ensemble of stochastic particles present within that pixel.
        In the case where particle drift makes any pixel empty, velocity component is interpolated.

        For a thorough discussion of SLM see section 12.3 of
        `S.B. Pope - Turbulent Flows (1995) <https://www.cambridge.org/highereducation/books/turbulent-flows/C58EFF59AF9B81AE6CFAC9ED16486B3A#contents>`_.

        .. note::

            Note that the stochastic particles for solving the SLM are in practice generated using the ``Particle`` class
            but they are independent of the seeded particles used for PIV.

        .. note::

            If the Lagrangian integral time scale is *large*, you may want to create a large image buffer to account
            for the stochastic particles moving a considerable distance compared to image size.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128, 128)

            # Initialize a mean flow field object:
            mean_flowfield = FlowField(n_images=n_images,
                                       size=image_size,
                                       size_buffer=10,
                                       time_separation=1,
                                       random_seed=100)

            # Generate sinusoidal velocity field that will serve as the mean velocity field:
            mean_flowfield.generate_sinusoidal_velocity_field(amplitudes=(2, 4),
                                                              wavelengths=(20, 40),
                                                              components='both')

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Solve the SLM for the mean velocity fields:
            flowfield.generate_langevin_velocity_field(mean_field=mean_flowfield.velocity_field,
                                                       integral_time_scale=1,
                                                       sigma=1,
                                                       n_stochastic_particles=10000,
                                                       n_iterations=100,
                                                       verbose=True)

            # Access the velocity components tensor:
            flowfield.velocity_field

            # Access the velocity field magnitude:
            flowfield.velocity_field_magnitude

        :param mean_field:
            ``numpy.ndarray`` specifying the mean velocity field with components :math:`u_m^*` and :math:`v_m^*`, respectively.
            It should be of size :math:`(N, 2, H+2 \cdot b, W+2 \cdot b)`.
            The SLM model essentially provides a stochastic drift over that user-specified mean velocity, preserving
            the Lagrangian integral time scale and the turbulent kinetic energy. The mean velocity can be synthetically
            generated using one of the velocity field generators provided in this class.
        :param integral_time_scale: (optional)
            ``int`` or ``float`` specifying the Lagrangian integral time scale, :math:`T_L`.
        :param sigma: (optional)
            ``int`` or ``float`` specifying the turbulent kinetic energy.
        :param n_stochastic_particles: (optional)
            ``int`` specifying the number of stochastic particles for which the SLM will be solved.
        :param n_iterations: (optional)
            ``int`` specifying the number of iterations, :math:`n`, during which the finite-difference equations are updated.
        :param verbose: (optional)
            ``bool`` for printing verbose details.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if (not isinstance(mean_field, np.ndarray)):
            raise ValueError("Parameter `mean_field` has to be of type 'numpy.ndarray'.")

        if (not isinstance(integral_time_scale, int)) and (not isinstance(integral_time_scale, float)):
            raise ValueError("Parameter `integral_time_scale` has to be of type 'int' or 'float'.")

        if (not isinstance(sigma, int)) and (not isinstance(sigma, float)):
            raise ValueError("Parameter `sigma` has to be of type 'int' or 'float'.")

        if (not isinstance(n_stochastic_particles, int)):
            raise ValueError("Parameter `n_stochastic_particles` has to be of type 'int'.")

        if (not isinstance(n_iterations, int)):
            raise ValueError("Parameter `n_iterations` has to be of type 'int'.")

        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be of type 'bool'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)
        self.__velocity_field_magnitude = np.zeros((self.__n_images, 1, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)

        # Compute seeding density for the user-specified number of stochastic particles:
        __target_particle_density = n_stochastic_particles / (self.__height_with_buffer * self.__width_with_buffer)

        # Initialize the stochastic particles:
        particles = Particle(1,
                             size=(self.__height_with_buffer, self.__width_with_buffer),
                             size_buffer=0,
                             densities=(__target_particle_density, __target_particle_density),
                             random_seed=self.__random_seed)

        if (particles.n_of_particles[0] > n_stochastic_particles+10) or (particles.n_of_particles[0] < n_stochastic_particles-10):
            raise RuntimeError("The true number of stochastic particles generated (" + str(particles.n_of_particles[0]) + ") is different from the one requested by the user (" + str(n_stochastic_particles) + "). This should not happen and is the issue of pykitPIV.")

        __n_stochastic_particles = particles.n_of_particles[0]

        # Coordinates of the stochastic particles:
        Y, X = particles.particle_coordinates[0]

        tic = time.perf_counter()

        for im in range(0, self.n_images):

            if verbose: print('Generating velocity field for image ' + str(im) + '...')

            # Determine the initial velocity for the stochastic particles based on the pixel to which they initially belong:
            U_star_mean = np.zeros_like(X)
            V_star_mean = np.zeros_like(Y)

            for i in range(0, __n_stochastic_particles):
                U_star_mean[i] = mean_field[im, 0, int(np.floor(Y[i])), int(np.floor(X[i]))]
                V_star_mean[i] = mean_field[im, 1, int(np.floor(Y[i])), int(np.floor(X[i]))]

            # Define delta t:
            delta_t = integral_time_scale / n_iterations

            # Define the constant factor from the SLM equation:
            square_root_factor = np.sqrt((1 / integral_time_scale) * 2 * sigma ** 2 * delta_t)

            # Initialize the
            U_star = np.zeros((__n_stochastic_particles, ))
            V_star = np.zeros((__n_stochastic_particles, ))

            X_positions = np.zeros((__n_stochastic_particles, ))
            X_positions[:] = X
            Y_positions = np.zeros((__n_stochastic_particles, ))
            Y_positions[:] = Y

            # Update the velocity components and the positions of stochastic particles:
            for _ in range(0, n_iterations):

                # Update the x-coordinates of the stochastic particles:
                X_positions = X_positions + U_star * delta_t

                # Update the u-component of velocity:
                U_star = U_star - (U_star - U_star_mean) * delta_t / integral_time_scale + square_root_factor * np.random.randn(__n_stochastic_particles)

                # Update the y-coordinate of the stochastic particles:
                Y_positions = Y_positions + V_star * delta_t

                # Update the v-component of velocity:
                V_star = V_star - (V_star - V_star_mean) * delta_t / integral_time_scale + square_root_factor * np.random.randn(__n_stochastic_particles)

            # Average the velocity over the ensemble of stochastic particles in each pixel:
            velocity_field_u = np.zeros((self.__height_with_buffer, self.__width_with_buffer))
            velocity_field_v = np.zeros((self.__height_with_buffer, self.__width_with_buffer))

            # This part is the largest computational bottleneck -- need to figure out how we can vectorize this code:
            tic_averaging = time.perf_counter()

            for i in range(0, self.__height_with_buffer):
                for j in range(0, self.__width_with_buffer):

                    # Control volume that spans one pixel:
                    locations = np.where((X_positions >= j - 0.5) & (X_positions < j + 0.5) & (Y_positions >= i - 0.5) & (Y_positions < i + 0.5))

                    # Average particle velocities within the control volume defined by one pixel:
                    velocity_field_u[i, j] = np.mean(U_star[locations])
                    velocity_field_v[i, j] = np.mean(V_star[locations])

            toc_averaging = time.perf_counter()
            if verbose: print(f'\tAveraging time: {(toc_averaging - tic_averaging) / 60:0.1f} minutes.\n')

            self.__velocity_field_magnitude[im, 0, :, :] = np.sqrt(velocity_field_u ** 2 + velocity_field_v ** 2)
            self.__velocity_field[im, 0, :, :] = velocity_field_u
            self.__velocity_field[im, 1, :, :] = velocity_field_v

            # Define the time vector for later:
            # time_vector = np.linspace(0, integral_time_scale, n_iterations)

        toc = time.perf_counter()
        if verbose: print(f'Total time: {(toc - tic) / 60:0.1f} minutes.\n' + '- ' * 40)

        # Compute the updated displacement per image:
        self.__displacement_per_image = np.max(np.abs(self.__velocity_field_magnitude), axis=(2,3)).ravel()

        # Compute the updated minimum and maximum displacement:
        self.__displacement = (np.min(self.__displacement_per_image), np.max(self.__displacement_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def compute_displacement_field(self):
        """
        Computes the displacement field, :math:`d \\vec{\\mathbf{s}}`,
        and the displacement field magnitude, :math:`|d \\vec{\\mathbf{s}}|`,
        based on the current value of the time separation, :math:`\\Delta t`.
        Calling this function populates the class attributes ``displacement_field``
        and ``displacement_field_magnitude``.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to generate 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128, 512)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Generate random velocity field:
            flowfield.generate_random_velocity_field(displacement=(0, 10),
                                                     gaussian_filters=(10, 30),
                                                     n_gaussian_filter_iter=6)

            # Compute the displacement field:
            flowfield.compute_displacement_field()
        """

        if self.velocity_field is not None:

            # Compute the displacement field:
            self.__displacement_field = np.zeros((self.n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]), dtype=self.dtype)
            self.__displacement_field[:, 0, :, :] = self.velocity_field[:, 0, :, :] * self.time_separation
            self.__displacement_field[:, 1, :, :] = self.velocity_field[:, 1, :, :] * self.time_separation

            self.__displacement_field_magnitude = np.sqrt(self.displacement_field[:, 0:1, :, :] ** 2 + self.displacement_field[:, 1:2, :, :] ** 2)

        else:
            raise AttributeError("Velocity field has not been created yet! No possible displacement field to compute.")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def upload_velocity_field(self, velocity_field):
        """
        Uploads a custom velocity field, *e.g.*, generated from synthetic turbulence or from turbulence databases.

        An example synthetic turbulence generator implemented in Python can be found here:
        `Saad et al. - TurboGenPY <https://github.com/saadgroup/TurboGenPY>`_.
        You can also access its `web-based version <https://turbulence.utah.edu/>`_.

        Another popular source of turbulent velocity fields is the
        `Johns Hopkins Turbulence Database (JHTD) <https://turbulence.idies.jhu.edu/home>`_.

        **Example:**

        .. code:: python

            from pykitPIV import FlowField

            # We are going to use 10 flow fields for 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128, 512)

            # Initialize a flow field object:
            flowfield = FlowField(n_images=n_images,
                                  size=image_size,
                                  size_buffer=10,
                                  time_separation=1,
                                  random_seed=100)

            # Importing an external velocity field file:
            # ...
            # ...

            # Prepare the velocity_field variable that has shape (10, 2, 148, 532):
            # velocity_field = ...

            # Upload velocity field:
            flowfield.upload_velocity_field(velocity_field)

        :param velocity_field:
            ``numpy.ndarray`` specifying the velocity components. It should be of size :math:`(N, 2, H, W)`,
            where :math:`N` is the number of PIV image pairs, :math:`2` refers to each velocity component
            :math:`u` and :math:`v` respectively,
            :math:`H` is the height and :math:`W` is the width of each PIV image. It can also be of size :math:`(1, 2, H, W)`,
            in which case the same velocity field will be applied to all PIV image pairs.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(velocity_field, np.ndarray):
            raise ValueError("Parameter `velocity_field` has to be of type 'numpy.ndarray'.")

        (N, n_velocity_components, H, W) = np.shape(velocity_field)

        if N == 1 and self.n_images != 1:
            print('The same velocity field will be applied to all PIV image pairs.')
        else:
            if N != self.n_images:
                raise ValueError("The number of PIV image pairs does not match the value at class initialization.")

        if n_velocity_components != 2:
            raise ValueError("The number of velocity components has to be equal to 2.")

        if self.size_buffer > 0:
            if H != self.__height_with_buffer:
                raise ValueError("Improper height of the velocity field. It should be equal to " + str(self.__height_with_buffer) + '.')
            if W != self.__width_with_buffer:
                raise ValueError("Improper width of the velocity field. It should be equal to " + str(self.__width_with_buffer) + '.')
        else:
            if H != self.size[0]:
                raise ValueError("Improper height of the velocity field. It should be equal to " + str(self.size[0]) + '.')
            if W != self.size[1]:
                raise ValueError("Improper width of the velocity field. It should be equal to " + str(self.size[1]) + '.')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if N == 1:

            self.__velocity_field = np.zeros((self.__n_images, 2, self.size_with_buffer[0], self.size_with_buffer[1]))

            for i in range(0, self.__n_images):
                self.__velocity_field[i, 0, :, :] = velocity_field[0, 0, :, :]
                self.__velocity_field[i, 1, :, :] = velocity_field[0, 1, :, :]

            self.__velocity_field_magnitude = np.sqrt(self.__velocity_field[:, 0:1, :, :] ** 2 + self.__velocity_field[:, 1:2, :, :] ** 2)

        else:

            self.__velocity_field = velocity_field
            self.__velocity_field_magnitude = np.sqrt(velocity_field[:, 0:1, :, :] ** 2 + velocity_field[:, 1:2, :, :] ** 2)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_divergence(vector_field,
                       edge_order=1):
    """
    Computes the divergence of the specified vector field.

    If the vector field is the velocity field, :math:`\\vec{V} = [u, v]`:

    .. math::

        \\nabla \\cdot \\vec{V} = \\frac{\\partial u}{\\partial x} + \\frac{\\partial v}{\\partial y}

    .. note::

        Derivatives are computed using `numpy.gradient <https://numpy.org/doc/stable/reference/generated/numpy.gradient.html>`_.

    **Example:**

    .. code:: python

        from pykitPIV import FlowField, compute_divergence

        # Initialize a flow field object:
        flowfield = FlowField(10,
                              size=(200, 200),
                              size_buffer=0,
                              time_separation=1,
                              random_seed=100)

        # Generate random velocity field:
        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(1, 2))

        # Extract the velocity field components:
        velocity_field = flowfield.velocity_field

        # Compute the divergence of the velocity field:
        divergence = compute_divergence(vector_field=velocity_field,
                                        edge_order=1)

    :param vector_field:
        ``numpy.ndarray`` specifying the vector field components. It should be of size :math:`(N, 2, H, W)`,
        where :math:`N` is the number of PIV image pairs, :math:`2` refers to each vector field component and
        :math:`H` is the height and :math:`W` is the width of each PIV image.
        For example, it can be the velocity field with components :math:`u` and :math:`v`, or
        the displacement field with components :math:`dx` and :math:`dy`.
    :param edge_order: (optional)
        ``int`` specifying the order for the gradient computation at image boundaries
        as per `numpy.gradient <https://numpy.org/doc/stable/reference/generated/numpy.gradient.html>`_.

    :return:
        - **divergence** - divergence of the velocity field, :math:`\\nabla \\cdot \\vec{V}`.
          It has size :math:`(N, H, W)`.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Input parameter check:

    check_four_dimensional_2D_vector_field_tensor(vector_field, 'vector_field')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    (dudy, dudx) = np.gradient(vector_field[:,0,:,:], axis=(1,2), edge_order=edge_order)
    (dvdy, dvdx) = np.gradient(vector_field[:,1,:,:], axis=(1,2), edge_order=edge_order)

    divergence = dudx + dvdy

    return divergence

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_vorticity(vector_field,
                      edge_order=1):
    """
    Computes the vorticity of the specified vector field.

    If the vector field is the velocity field, :math:`\\vec{V} = [u, v]`:

    .. math::

        \\nabla \\times \\vec{V} = \\frac{\\partial v }{\\partial x} - \\frac{\\partial u}{\\partial y}

    .. note::

        Derivatives are computed using `numpy.gradient <https://numpy.org/doc/stable/reference/generated/numpy.gradient.html>`_.

    **Example:**

    .. code:: python

        from pykitPIV import FlowField, compute_vorticity

        # Initialize a flow field object:
        flowfield = FlowField(10,
                              size=(200, 200),
                              size_buffer=0,
                              time_separation=1,
                              random_seed=100)

        # Generate random velocity field:
        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(1, 2))

        # Extract the velocity field components:
        velocity_field = flowfield.velocity_field

        # Compute the vorticity of the velocity field:
        vorticity = compute_vorticity(vector_field=velocity_field,
                                      edge_order=1)

    :param vector_field:
        ``numpy.ndarray`` specifying the vector field components. It should be of size :math:`(N, 2, H, W)`,
        where :math:`N` is the number of PIV image pairs, :math:`2` refers to each vector field component and
        :math:`H` is the height and :math:`W` is the width of each PIV image.
        For example, it can be the velocity field with components :math:`u` and :math:`v`, or
        the displacement field with components :math:`dx` and :math:`dy`.
    :param edge_order: (optional)
        ``int`` specifying the order for the gradient computation at image boundaries
        as per `numpy.gradient <https://numpy.org/doc/stable/reference/generated/numpy.gradient.html>`_.

    :return:
        - **vorticity** - vorticity of the velocity field, :math:`\\nabla \\times \\vec{V}`.
          It has size :math:`(N, H, W)`.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Input parameter check:

    check_four_dimensional_2D_vector_field_tensor(vector_field, 'vector_field')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    (dudy, dudx) = np.gradient(vector_field[:,0,:,:], axis=(1,2), edge_order=edge_order)
    (dvdy, dvdx) = np.gradient(vector_field[:,1,:,:], axis=(1,2), edge_order=edge_order)

    vorticity = dvdx - dudy

    return vorticity

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_q_criterion(vector_field,
                        edge_order=1):
    """
    Computes the Q-criterion for the specified vector field, :math:`\\vec{V} = [u, v]`.

    The Q-criterion comes from decomposing the velocity gradient into the rate of strain tensor and the vorticity tensor:

    .. math::

        \\nabla \\vec{V} = \\mathbf{S} + \\pmb{\\Omega}

    where :math:`\\mathbf{S} = \\frac{1}{2} [ (\\nabla \\vec{V}) + (\\nabla \\vec{V})^{\\top}]` is the rate-of-strain tensor
    and :math:`\\pmb{\\Omega} = \\frac{1}{2} [ (\\nabla \\vec{V}) - (\\nabla \\vec{V})^{\\top}]` is the vorticity tensor.

    The Q-criterion is defined as regions where the vorticity tensor magnitude dominates over the strain rate tensor magnitude:

    .. math::

        Q = \\frac{1}{2} \\big( \\| \\pmb{\\Omega} \\|^2 - \\| \\mathbf{S} \\|^2 \\big) > 0

    where :math:`\\| \\bullet \\|` is the Euclidean (or Frobenius) norm.

    In 2D, the vorticity tensor is

    .. math::

        \\pmb{\\Omega} =
        \\begin{bmatrix}
        0 & \\frac{1}{2} \\left( \\frac{\\partial v}{\\partial x} - \\frac{\\partial u}{\\partial y} \\right) \\\\
        \\frac{1}{2} \\left( \\frac{\\partial u}{\\partial y} - \\frac{\\partial v}{\\partial x} \\right) & 0
        \\end{bmatrix}

    and the rate-of-strain tensor is

    .. math::

        \\mathbf{S} =
        \\begin{bmatrix}
        \\frac{\\partial u}{\\partial x} & \\frac{1}{2} \\left( \\frac{\\partial v}{\\partial x} + \\frac{\\partial u}{\\partial y} \\right) \\\\
        \\frac{1}{2} \\left( \\frac{\\partial u}{\\partial y} + \\frac{\\partial v}{\\partial x} \\right) & \\frac{\\partial v}{\\partial y}
        \\end{bmatrix}

    Hence, in 2D, we can write the Q-criterion as:

    .. math::

        Q = \\frac{1}{2} \\left( \\frac{1}{2} \\left( \\frac{\\partial v}{\\partial x} - \\frac{\\partial u}{\\partial y} \\right)^2 - \\left( \\frac{\\partial u}{\\partial x} \\right)^2 - \\left( \\frac{\\partial v}{\\partial y} \\right)^2 - \\frac{1}{2} \\left( \\frac{\\partial u}{\\partial y} + \\frac{\\partial v}{\\partial x} \\right)^2 \\right)

    See `Jeong & Hussain <https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/on-the-identification-of-a-vortex/D26006DDB95FB28DA80E28A581182DF1>`_
    and `Haller <https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/an-objective-definition-of-a-vortex/3CD781A3AEC4BA16571CBC2D9B4E973F>`_ for more info on the Q-criterion.

    .. note::

        Derivatives are computed using `numpy.gradient <https://numpy.org/doc/stable/reference/generated/numpy.gradient.html>`_.

    **Example:**

    .. code:: python

        from pykitPIV import FlowField, compute_q_criterion

        # Initialize a flow field object:
        flowfield = FlowField(10,
                              size=(200, 200),
                              size_buffer=0,
                              time_separation=1,
                              random_seed=100)

        # Generate random velocity field:
        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(1, 2))

        # Extract the velocity field components:
        velocity_field = flowfield.velocity_field

        # Compute the Q-criterion for the velocity field:
        q_criterion = compute_q_criterion(vector_field=velocity_field,
                                          edge_order=1)

    :param vector_field:
        ``numpy.ndarray`` specifying the vector field components. It should be of size :math:`(N, 2, H, W)`,
        where :math:`N` is the number of PIV image pairs, :math:`2` refers to each vector field component and
        :math:`H` is the height and :math:`W` is the width of each PIV image.
        For example, it can be the velocity field with components :math:`u` and :math:`v`, or
        the displacement field with components :math:`dx` and :math:`dy`.
    :param edge_order: (optional)
        ``int`` specifying the order for the gradient computation at image boundaries
        as per `numpy.gradient <https://numpy.org/doc/stable/reference/generated/numpy.gradient.html>`_.

    :return:
        - **q_criterion** - Q-criterion of the velocity field, :math:`Q`.
          It has size :math:`(N, H, W)`.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Input parameter check:

    check_four_dimensional_2D_vector_field_tensor(vector_field, 'vector_field')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Compute the vorticity:
    (dudy, dudx) = np.gradient(vector_field[:,0,:,:], axis=(1,2), edge_order=edge_order)
    (dvdy, dvdx) = np.gradient(vector_field[:,1,:,:], axis=(1,2), edge_order=edge_order)

    # Compute the components of the vorticity tensor:
    omega_12 = 0.5 * (dvdx - dudy)
    omega_21 = 0.5 * (dudy - dvdx)

    # Compute the components of the strain rate tensor:
    S_11 = dudx
    S_22 = dvdy
    S_12 = 0.5 * (dvdx + dudy)
    S_21 = 0.5 * (dudy + dvdx)

    # Compute the Q-criterion:
    q_criterion = 0.5 * ( (omega_12 ** 2 + omega_21 ** 2) - (S_11 ** 2 + S_22 ** 2 + S_12 ** 2 + S_21 ** 2) )

    return q_criterion

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
