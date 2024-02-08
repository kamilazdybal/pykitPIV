import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy
import warnings
from pypiv.checks import *
from pypiv.particle import Particle
from pypiv.flowfield import FlowField
from pypiv.motion import Motion

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

    :param size: (optional)
        ``tuple`` of two ``int`` elements specifying the size of each image in pixels. The first number is image height, the second number is image width.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``.
    """

    def __init__(self,
                 size=(512,512),
                 random_seed=None,
                 ):

        # Input parameter check:
        check_two_element_tuple(size, 'size')

        if random_seed is not None:
            if type(random_seed) != int:
                raise ValueError("Parameter `random_seed` has to be of type 'int'.")
            else:
                np.random.seed(seed=random_seed)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Class init:
        self.__size = size
        self.__random_seed = random_seed

        # Create empty image at class init:
        self.__empty_image = np.zeros((size[0], size[1]))

        # Initialize images:
        self.__images = None

        # Initialize particles:
        self.__particles = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Properties coming from user inputs:
    @property
    def size(self):
        return self.__size

    @property
    def random_seed(self):
        return self.__random_seed

    # Properties computed at class init:
    @property
    def empty_image(self):
        return self.__empty_image

    @property
    def images(self):
        return self.__images

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_particles(self, particles):

        if not isinstance(particles, Particle):
            raise ValueError("Parameter `particles` has to be an instance of `Particle` class.")

        # Check that the size of images are consistent between the Image and Particle objects:
        if particles.size != self.size:
            raise ValueError("Inconsistent image sizes between the current `Image` object and the `Particle` object.")

        self.__particles = particles
        self.__images = self.__particles.particle_positions

        print('Particles added to the image.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def light_intensity_distribution(self,
                                     peak_intensity,
                                     particle_radius,
                                     coordinate_height,
                                     coordinate_width):

        bandwidth = particle_radius/3

        pixel_value = peak_intensity * np.exp(-0.5 * (coordinate_height**2 + coordinate_width**2) / bandwidth**2)

        return pixel_value

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_gaussian_light_distribution(self,
                                        exposures=(0.02,0.8),
                                        maximum_intensity=2**16-1,
                                        laser_beam_thickness=2,
                                        laser_over_exposure=1,
                                        laser_beam_shape=0.85):

        """
        Creates particle sizes and adds laser light reflected from particles. The reflected light follows a Gaussian distribution.

        :param exposures: (optional)
            ``tuple`` of two numerical elements specifying the light exposure.
        :param maximum_intensity: (optional)
            ``int`` specifying the maximum light intensity.
        :param laser_beam_thickness: (optional)
            ``int`` or ``float`` specifying the thickness of the laser beam. With a small thickness, particles that are even slightly off-plane will appear darker.
        :param laser_over_exposure: (optional)
            ``int`` or ``float`` specifying the overexposure of the laser beam.
        :param laser_beam_shape: (optional)
            ``int`` or ``float`` specifying the shape of the laser beam.
        """

        # Input parameter check:
        check_two_element_tuple(exposures, 'exposures')

        if self.__particles is None:
            raise NameError("Particles have not been added to the image yet! Use the `Image.add_particles()` method first.")

        # Randomize image exposure:
        image_exposure = np.random.rand(self.__particles.n_images) * (exposures[1] - exposures[0]) + exposures[0]

        images = []

        for i in range(0,self.__particles.n_images):

            # Initialize an empty image:
            particles_with_gaussian_light = np.zeros((self.__particles.size[0], self.__particles.size[1]))

            # Compute particle coordinates on the current image:
            (particle_height_coordinate, particle_width_coordinate) = np.where(self.__particles.particle_positions[i]==1)

            # Establish the peak intensity for each particle depending on its position with respect to the laser beam plane:
            particle_positions_off_laser_plane = laser_beam_thickness * np.random.rand(self.__particles.n_of_particles[i]) - laser_beam_thickness/2
            particle_position_relative_to_laser_centerline = np.abs(particle_positions_off_laser_plane) / (laser_beam_thickness/2)
            particle_peak_intensities = image_exposure[i] * maximum_intensity * np.exp(-0.5 * (particle_position_relative_to_laser_centerline**2 / laser_beam_shape**2))

            # Add Gaussian blur to each particle location that mimics the particle size:
            for p in range(0,self.__particles.n_of_particles[i]):

                px_c_height = int(np.floor(particle_height_coordinate[p]))
                px_c_width = int(np.floor(particle_width_coordinate[p]))
                ceil_of_particle_radius = int(np.ceil(self.__particles.particle_radii[i][p]))

                for h in range(px_c_height-ceil_of_particle_radius, px_c_height+ceil_of_particle_radius):
                    for w in range(px_c_width-ceil_of_particle_radius, px_c_width+ceil_of_particle_radius):
                        if (h >= 0 and h < self.__particles.size[0]) and (w >= 0 and w < self.__particles.size[1]):

                            coordinate_height = h + 0.5 - particle_height_coordinate[p]
                            coordinate_width = w + 0.5 - particle_width_coordinate[p]
                            particles_with_gaussian_light[h,w] = particles_with_gaussian_light[h,w] + self.light_intensity_distribution(particle_peak_intensities[p], self.__particles.particle_radii[i][p], coordinate_height, coordinate_width)

            images.append(particles_with_gaussian_light)

        self.__images = images

        print('Gaussian light added to the image.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot(self, idx, cmap='Greys_r', figsize=(5,5), filename=None):

        if self.__images is None:

            print('Note: Particles have not been added to the image yet!\n\n')

            fig = plt.figure(figsize=figsize)
            plt.imshow(self.empty_image, cmap=cmap)

        else:

            fig = plt.figure(figsize=figsize)
            plt.imshow(self.__images[idx], cmap=cmap)

        if filename is not None:

            plt.savefig(filename, dpi=300, bbox_inches='tight')

        return plt

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
