import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import random
import copy
import scipy
from scipy.stats import qmc
from scipy.spatial import cKDTree
import warnings
from pykitPIV.checks import *

########################################################################################################################
########################################################################################################################
####
####    Class: ParticleSpecs
####
########################################################################################################################
########################################################################################################################

class ParticleSpecs:
    """
    Configuration object for the ``Particle`` class.

    **Example:**

    .. code:: python

        from pykitPIV import ParticleSpecs

        # Instantiate an object of ParticleSpecs class:
        particle_spec = ParticleSpecs()

        # Change one field of particle_spec:
        particle_spec.diameters = (2,2)

        # You can print the current values of all attributes:
        print(particle_spec)
    """

    def __init__(self,
                 n_images=1,
                 size=(512, 512),
                 size_buffer=10,
                 diameters=(3, 6),
                 distances=(0.5, 2),
                 densities=(0.05, 0.1),
                 diameter_std=0.1,
                 seeding_mode='random',
                 random_seed=None):

        self.n_images = n_images
        self.size = size
        self.size_buffer = size_buffer
        self.diameters = diameters
        self.distances = distances
        self.densities = densities
        self.diameter_std = diameter_std
        self.seeding_mode = seeding_mode
        self.random_seed = random_seed

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"(n_images={self.n_images},\n"
                f"size={self.size},\n"
                f"size_buffer={self.size_buffer},\n"
                f"diameters={self.diameters},\n"
                f"distances={self.distances},\n"
                f"densities={self.densities},\n"
                f"diameter_std={self.diameter_std},\n"
                f"seeding_mode={self.seeding_mode!r},\n"
                f"random_seed={self.random_seed})"
                )

########################################################################################################################
########################################################################################################################
####
####    Class: Particle
####
########################################################################################################################
########################################################################################################################

class Particle:
    """
    Generates tracer particles with specified properties for a set of :math:`N` PIV image pairs.
    This class generates the starting positions for tracer particles, *i.e.*, the ones used for :math:`I_1`.

    **Example:**

    .. code:: python

        from pykitPIV import Particle

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
                             diameter_std=1,
                             seeding_mode='random',
                             random_seed=100)

        # Access particle coordinates on the first image:
        (height_coordinates, width_coordinates) = particles.particle_coordinates[0]

    .. image:: ../images/Particle-setting-spectrum.png
        :width: 800

    :param n_images:
        ``int`` specifying the number of PIV image pairs, :math:`N`, to create.
    :param size: (optional)
        ``tuple`` of two ``int`` elements specifying the size of images in pixels :math:`[\\text{px}]`.
        The first number is the image height, :math:`H`, the second number is the image width, :math:`W`.
    :param size_buffer: (optional)
        ``int`` specifying the buffer, :math:`b`, in pixels :math:`[\\text{px}]` to add to the image size
        in the width and height direction.
        This number should be approximately equal to the maximum displacement that particles are subject to
        in order to allow new particles to arrive into the image area
        and prevent spurious disappearance of particles near image boundaries.
    :param diameters: (optional)
        ``tuple`` of two ``int`` elements specifying the minimum (first element) and maximum (second element)
        particle diameter in pixels :math:`[\\text{px}]` to randomly sample from across all generated PIV image pairs.
        Note, that one PIV pair will be associated with one (fixed) particle diameter, but the random sample
        between minimum and maximum diameter will generate variation in diameters across :math:`N` PIV image pairs.
        You can steer the deviation from that diameter within each single PIV image pair
        using the ``diameter_std`` parameter.
    :param distances: (optional)
        ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
        particle distances in pixels :math:`[\\text{px}]` to randomly sample from.
        Only used when ``seeding_mode`` is ``'poisson'``.
    :param densities: (optional)
        ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element)
        particle seeding density on an image in particle per pixel :math:`[\\text{ppp}]` to randomly sample from.
        Only used when ``seeding_mode`` is ``'random'``.
    :param diameter_std: (optional)
        ``float`` or ``int`` specifying the standard deviation in pixels :math:`[\\text{px}]` for the distribution
        of particle diameters within one PIV image pair. If set to zero, all particles in a PIV image pair will have
        diameters exactly equal.
    :param seeding_mode: (optional)
        ``str`` specifying the seeding mode for initializing particles in the image domain.
        It can be one of the following: ``'random'``, ``'poisson'``, or ``'user'``.

        - ``'random'`` seeding generates random locations of particles on the available image area.
        - ``'poisson'`` seeding is also random, but makes sure that particles are kept at minimum ``distances``
          from one another. This is particularly useful for generation BOS-like background image.
        - ``'user'`` seeding allows for particle coordinates to be provided by the user.
          This provides an interesting functionality where the user can chain movement of particles and create
          time-resolved PIV sequence of images.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``.
        If specified, all image generation is reproducible.

    **Attributes:**

    - **n_images** - (read-only) as per user input.
    - **size** - (read-only) as per user input.
    - **size_buffer** - (read-only) as per user input.
    - **diameters** - (read-only) as per user input.
    - **distances** - (read-only) as per user input.
    - **densities** - (read-only) as per user input.
    - **diameter_std** - (read-only) as per user input.
    - **seeding_mode** - (read-only) as per user input.
    - **random_seed** - (read-only) as per user input.
    - **size_with_buffer** - (read-only) ``tuple`` specifying the size of each image in pixels with buffer added.
    - **diameter_per_image** - (read-only) ``numpy.ndarray`` specifying the template for the particle diameters in pixels
      :math:`[\\text{px}]` for each image. Template diameters are random numbers between ``diameters[0]`` and ``diameters[1]``.
    - **distance_per_image** - (read-only) ``numpy.ndarray`` specifying the template for the particle distances in pixels
      :math:`[\\text{px}]` for each image. Template distances are random numbers between ``distances[0]`` and ``distances[1]``.
    - **density_per_image** - (read-only) ``numpy.ndarray`` specifying the template for the particle densities in particle
      per pixel :math:`[\\text{ppp}]` for each image. Template densities are random numbers between ``densities[0]`` and ``densities[1]``.
    - **n_of_particles** - (read-only) ``list`` specifying the number of particles created for each image, :math:`n_i`, based on each template density.
    - **particle_coordinates** - (read-only) ``list`` specifying the absolute coordinates of all particle centers for each image.
      The positions are computed based on the ``seeding_mode``. The first element in each tuple are the coordinates
      along the image height, and the second element are the coordinates along the image width.
    - **particle_positions** - (read-only) ``numpy.ndarray`` specifying the per-pixel starting positions of all particles' centers
      for each PIV image pair; these positions will later populate the first PIV image frame, :math:`I_1`.
      The positions are computed based on the ``seeding_mode``. If a particle's position falls into
      a specific pixel coordinate, this pixel's value is increased by one. Zero entry indicates that no particles are present
      inside that pixel.
      This array has size :math:`(N, C_{in}, H+2b, W+2b)`, where :math:`N` is the number PIV image pairs,
      :math:`C_{in}` is the number of channels (one channel, greyscale, is supported at the moment), :math:`H` is the height
      and :math:`W` the width of each PIV image, and :math:`b` is an optional image buffer.
    - **particle_diameters** - (read-only) ``list`` of ``numpy.ndarray``, each specifying the diameters of
      all seeded particles in pixels :math:`[\\text{px}]`
      for each image based on each template diameter. Each array in this list has length :math:`n_i`.
   """

    def __init__(self,
                 n_images,
                 size=(512,512),
                 size_buffer=10,
                 diameters=(3,6),
                 distances=(0.5,2),
                 densities=(0.05,0.1),
                 diameter_std=0.1,
                 seeding_mode='random',
                 random_seed=None):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if type(n_images) != int:
            raise ValueError("Parameter `n_images` has to of type 'int'.")

        if n_images <= 0:
            raise ValueError("Parameter `n_images` has to be positive. At least one image has to be generated.")

        check_two_element_tuple(size, 'size')

        if not isinstance(size_buffer, int):
            raise ValueError("Parameter `size_buffer` has to be of type 'int'.")

        if size_buffer < 0:
            raise ValueError("Parameter `size_buffer` has to non-negative.")

        check_two_element_tuple(diameters, 'diameters')
        check_min_max_tuple(diameters, 'diameters')
        check_two_element_tuple(distances, 'distances')
        check_min_max_tuple(distances, 'distances')
        check_two_element_tuple(densities, 'densities')
        check_min_max_tuple(densities, 'densities')

        if (not isinstance(diameter_std, float)) and (not isinstance(diameter_std, int)):
            raise ValueError("Parameter `diameter_std` has to be of type 'float' or 'int'.")

        __seeding_mode = ['random', 'user', 'poisson']
        if seeding_mode not in __seeding_mode:
            raise ValueError("Parameter `seeding_mode` has to be 'random', 'user', or 'poisson'.")

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
        self.__diameters = diameters
        self.__distances = distances
        self.__densities = densities
        self.__diameter_std = diameter_std
        self.__seeding_mode = seeding_mode
        self.__random_seed = random_seed

        # Compute the image outline that serves as a buffer:
        self.__height_with_buffer = self.size[0] + 2 * self.size_buffer
        self.__width_with_buffer = self.size[1] + 2 * self.size_buffer
        self.__size_with_buffer = (self.__height_with_buffer, self.__width_with_buffer)

        # Initialize parameters for particle generation:
        self.__particle_diameter_per_image = np.random.rand(self.__n_images) * (self.__diameters[1] - self.__diameters[0]) + self.__diameters[0]
        self.__particle_distance_per_image = np.random.rand(self.__n_images) * (self.__distances[1] - self.__distances[0]) + self.__distances[0]

        # Use one of the available modes for image generation: - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if seeding_mode == 'random':

            # Compute the seeding density for each image:
            self.__particle_density_per_image = np.random.rand(self.__n_images) * (self.__densities[1] - self.__densities[0]) + self.__densities[0]

            # Compute the total number of particles for a given particle density on each image:
            n_of_particles = self.__size[0] * self.__size[1] * self.__particle_density_per_image
            self.__n_of_particles = [int(i) for i in n_of_particles]

            # Initialize particle coordinates, positions, and diameters on each of the ``n_image`` images:
            particle_coordinates = []
            particle_positions = np.zeros((self.n_images, 1, self.__height_with_buffer, self.__width_with_buffer))
            particle_diameters = []

            # Populate particle data for each of the ``n_image`` images:
            for i in range(0,self.n_images):

                # Generate absolute coordinates for particles' centers within the total available image area (drawn from random uniform distribution):
                self.__y_coordinates = self.__height_with_buffer * np.random.rand(self.n_of_particles[i])
                self.__x_coordinates = self.__width_with_buffer * np.random.rand(self.n_of_particles[i])

                particle_coordinates.append((self.__y_coordinates, self.__x_coordinates))

                seeded_array = self.__populate_seeded_array()

                # Populate the 4D tensor of shape (N, C_in, H, W):
                particle_positions[i, 0, :, :] = seeded_array

                # Generate diameters for all particles in the current image:
                particle_diameters.append(np.random.normal(self.diameter_per_image[i], self.diameter_std, self.n_of_particles[i]))

            # Initialize particle coordinates:
            self.__particle_coordinates = particle_coordinates

            # Initialize particle positions:
            self.__particle_positions = particle_positions

            # Initialize particle diameters:
            self.__particle_diameters = particle_diameters

        elif seeding_mode == 'poisson':

            # Compute the seeding density for each image:
            self.__particle_density_per_image = np.random.rand(self.__n_images) * (self.__densities[1] - self.__densities[0]) + self.__densities[0]

            # Initialize particle coordinates, positions, and diameters on each of the ``n_image`` images:
            particle_coordinates = []
            particle_positions = np.zeros((self.n_images, 1, self.__height_with_buffer, self.__width_with_buffer))
            particle_diameters = []
            n_of_particles = []
            s=max(self.__size_with_buffer)
            for i in range(0,self.n_images):

                dia=self.__particle_diameter_per_image[i]
                den=self.__particle_density_per_image[i]
                dist=self.__particle_distance_per_image[i]
                radius=dia/den/s
                #radius = 0.2
                engine = qmc.PoissonDisk(d=2, radius=radius,ncandidates=30)
                sample = engine.fill_space()*s
                # check for  outlayers 
                ix=sample[:,1]<self.__size_with_buffer[1] 
                iy=sample[:,0]<self.__size_with_buffer[0]
                ind=ix * iy
                # delete outlayers 
                sample=sample[ind]
                x=sample[:,1]
                y=sample[:,0]
                n=len(x)
                # number of paricles
                n_of_particles.append(n)

                self.__x_coordinates=x
                self.__y_coordinates=y
                particle_coordinates.append((self.__y_coordinates, self.__x_coordinates))

                seeded_array = self.__populate_seeded_array()
                
                # Populate the 4D tensor of shape (N, C_in, H, W):
                particle_positions[i, 0, :, :] = seeded_array

                # Generate diameters for all particles in the current image:
                particle_diameters.append(np.random.normal(self.diameter_per_image[i], self.diameter_std, n))

            # Initialize the total number of particles:
            self.__n_of_particles = n_of_particles

            # Initialize particle coordinates:
            self.__particle_coordinates = particle_coordinates

            # Initialize particle positions:
            self.__particle_positions = particle_positions

            # Initialize particle diameters:
            self.__particle_diameters = particle_diameters

        elif seeding_mode == 'user':

            print('Use the function Particle.upload_particle_coordinates() to seed particles.')

            # Initialize the total number of particles:
            self.__n_of_particles = None

            # Initialize particle coordinates:
            self.__particle_coordinates = None

            # Initialize particle positions:
            self.__particle_positions = None

            # Initialize particle diameters:
            self.__particle_diameters = None

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
    def diameters(self):
        return self.__diameters

    @property
    def distances(self):
        return self.__distances

    @property
    def densities(self):
        return self.__densities

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
    def size_with_buffer(self):
        return self.__size_with_buffer

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
    def particle_coordinates(self):
        return self.__particle_coordinates

    @property
    def particle_positions(self):
        return self.__particle_positions

    @property
    def particle_diameters(self):
        return self.__particle_diameters

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __populate_seeded_array(self):
        """
        Populates a matrix that shows particle locations per pixel of the image area.
        """

        seeded_array = np.zeros((self.__height_with_buffer, self.__width_with_buffer))
        for x, y in zip(np.floor(self.__x_coordinates).astype(int), np.floor(self.__y_coordinates).astype(int)):
            seeded_array[y, x] += 1

        return seeded_array

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def upload_particle_coordinates(self,
                                    particle_coordinates):
        """
        Uploads user-specified starting particle coordinates.

        .. note::

            Note that only the :math:`x` and :math:`y` coordinates get uploaded to the current ``Particle`` class object
            and particle diameters (``Particle.particle_diameters``) get computed for the user-specified
            particles from the class attribute ``diameters`` that was given during ``Particle`` class object generation.

            The number of particles per image (``Particle.n_of_particles``) is computed directly from
            the user-specified ``particle_coordinates`` argument, so is ``Particle.particle_positions``.

        **Example:**

        .. code:: python

            from pykitPIV import Particle

            # We are going to generate 10 PIV image pairs:
            n_images = 10

            # Specify size in pixels for each image:
            image_size = (128,512)

            # Initialize the first particle object:
            particles_1 = Particle(n_images,
                                   size=image_size,
                                   diameters=(2,4),
                                   diameter_std=1,
                                   densities=(0.1, 0.1),
                                   seeding_mode='random')

            # Initialize the second particle object:
            particles_2 = Particle(n_images,
                                   size=image_size,
                                   densities=(0.1, 0.1),
                                   seeding_mode='random')

            # Overwrite the coordinates in the first particle object with the second one:
            particles_1.upload_particle_coordinates(particles_2.particle_coordinates)

        Alternatively, you can instantiate an object of the ``Particle`` class with the ``seeding_mode='user'``,
        which will initially make all the particle parameters empty. Those parameters get populated the moment you
        call the ``upload_particle_coordinates()`` function.

        **Example:**

        .. code:: python

            # Initialize the first particle object:
            particles_1 = Particle(n_images,
                                   size=image_size,
                                   diameters=(2,4),
                                   diameter_std=1,
                                   seeding_mode='user')

            # Initially, all of these attributes will be None:
            particles_1.particle_coordinates
            particles_1.particle_positions
            particles_1.particle_diameters
            particles_1.n_of_particles

            # Initialize the second particle object:
            particles_2 = Particle(n_images,
                                   size=image_size,
                                   densities=(0.1, 0.1),
                                   seeding_mode='random')

            # Overwrite the coordinates in the first particle object with the second one:
            particles_1.upload_particle_coordinates(particles_2.particle_coordinates)

        :param particle_coordinates:
            ``list`` of ``tuple`` specifying the coordinates of particles in image :math:`I_1`.
            The first element in each tuple are the coordinates along the image height,
            and the second element are the coordinates along the image width. It can be obtained directly from
            an object of the ``Motion`` class by accessing the attribute ``Motion.particle_coordinates_I1``. This allows
            to have starting particle coordinates with a more irregular pattern.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not isinstance(particle_coordinates, list):
            raise ValueError("Parameter `particle_coordinates` has to be of type `list`.")

        if len(particle_coordinates) != self.n_images:
            raise ValueError("Parameter `particle_coordinates` must have `n_images` number of elements.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Initialize particle positions and diameters on each of the ``n_image`` images:
        particle_positions = np.zeros((self.n_images, 1, self.__height_with_buffer, self.__width_with_buffer))
        particle_diameters = []
        n_of_particles = []

        # Populate particle data for each of the ``n_image`` images:
        for i in range(0,self.n_images):

            # Access absolute coordinates for particles' centers within the total available image area:
            self.__y_coordinates = particle_coordinates[i][0]
            self.__x_coordinates = particle_coordinates[i][1]

            n_particles = len(particle_coordinates[i][0])

            n_of_particles.append(n_particles)

            seeded_array = self.__populate_seeded_array()

            # Populate the 4D tensor of shape (N, C_in, H, W):
            particle_positions[i, 0, :, :] = seeded_array

            # Generate diameters for all particles in the current image:
            particle_diameters.append(np.random.normal(self.diameter_per_image[i], self.diameter_std, n_particles))

        # Initialize the total number of particles:
        self.__n_of_particles = n_of_particles

        # Initialize particle coordinates:
        self.__particle_coordinates = particle_coordinates

        # Initialize particle positions:
        self.__particle_positions = particle_positions

        # Initialize particle diameters:
        self.__particle_diameters = particle_diameters

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # ##################################################################################################################

    # Plotting functions

    # ##################################################################################################################

    def plot(self,
             idx,
             with_buffer=False,
             xlabel=None,
             ylabel=None,
             xticks=True,
             yticks=True,
             title=None,
             cmap='Greys',
             origin='lower',
             figsize=(5,5),
             dpi=300,
             filename=None):
        """
        Plots particles.

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

        if not isinstance(origin, str):
            raise ValueError("Parameter `origin` has to be of type 'str'.")

        check_two_element_tuple(figsize, 'figsize')

        if (filename is not None) and (not isinstance(filename, str)):
            raise ValueError("Parameter `filename` has to be of type 'str'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if self.__particle_coordinates is None:

            print('Note: Particles have not been generated yet!\n\n')

        else:
            y=self.__size_with_buffer[0]
            x=self.__size_with_buffer[1]
            sb=self.__size_buffer
 
            sample=self.__particle_coordinates[idx]   # y,x corrdiantes
            radius=self.__particle_diameters[idx]/2   
            pos=self.__particle_positions[idx][0]   # 0..n  pixel map
            pixelmap=pos
            pixelmap[pos>=1]=1

     
            fig,ax=plt.subplots(figsize=figsize)
            circles = [plt.Circle((xi, yi), radius=r, color='b',fill=False) for xi, yi ,r in zip(sample[1],sample[0],radius)]
            collection = PatchCollection(circles, match_original=True)
            rect=patches.Rectangle((sb-0.5,sb-0.5), self.size[1], self.size[0], linewidth=1, edgecolor='r', facecolor='none',fill=False)
            ax.add_collection(collection)           # circels in particle size
            _ = ax.scatter(sample[1], sample[0])    # centers of the particles 
            _ = ax.imshow(pixelmap,cmap=cmap)                    # pixel map of the centers
            _ = ax.add_patch(rect)                  # draw actual imageframe
            _ = ax.set(aspect='equal',  xlim=[0, x], ylim=[0, y])


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

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_properties_per_image(self):
        """
        Plots statistical properties of the generated particles on one selected image out of all ``n_images`` images.

        :return:
            - **plt** - ``matplotlib.pyplot`` image handle.
        """

        raise NotImplementedError('This function not implemented yet.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_properties_across_images(self):
        """
        Plots statistical properties of the generated particles across all ``n_images`` images.

        :return:
            - **plt** - ``matplotlib.pyplot`` image handle.
        """

        raise NotImplementedError('This function not implemented yet.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -