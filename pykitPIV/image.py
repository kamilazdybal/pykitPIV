import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy
import warnings
import matplotlib.patches as patches
from pykitPIV.checks import *
from pykitPIV.particle import Particle
from pykitPIV.flowfield import FlowField
from pykitPIV.motion import Motion

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
        ``int`` specifying the random seed for random number generation in ``numpy``. If specified, all image generation is reproducible.

    **Attributes:**

    - **size** - (read-only) as per user input.
    - **random_seed** - (read-only) as per user input.
    - **empty_image** - (read-only) ``numpy.ndarray`` with an empty image of a given size.
    - **images** - (read-only) ``list`` of ``numpy.ndarray``, where each element is the current version of a PIV image of a given size.
    - **particles** - (read-only) object of ``pypiv.Particle`` class.
    - **exposures_per_image** - (read-only) ``numpy.ndarray`` specifying the template for the light exposure for each image.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __init__(self,
                 size=(512,512),
                 random_seed=None,
                 ):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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

        # Initialize flow field:
        self.__flowfield = None

        # Initialize motion:
        self.__motion = None

        # Initialize exposures per image:
        self.__exposures_per_image = None

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

    @property
    def exposures_per_image(self):
        return self.__exposures_per_image

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

        # Check that the size of images are consistent between the Image and Particle objects:
        if particles.size != self.size:
            raise ValueError("Inconsistent image sizes between the current `Image` object and the `Particle` object.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.__particles = particles
        self.__images = self.__particles.particle_positions

        print('Particles added to the image.')

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
                            exposures=(0.02,0.8),
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
            ``int`` or ``float`` specifying the thickness of the laser beam. With a small thickness, particles that are even slightly off-plane will appear darker.
        :param laser_over_exposure: (optional)
            ``int`` or ``float`` specifying the overexposure of the laser beam.
        :param laser_beam_shape: (optional)
            ``int`` or ``float`` specifying the spread of the Gaussian shape of the laser beam. The larger this number is, the wider the Gaussian light distribution from the laser and more particles will be illuminated.
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

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Randomize image exposure:
        self.__exposures_per_image = np.random.rand(self.__particles.n_images) * (exposures[1] - exposures[0]) + exposures[0]

        images = []

        for i in range(0,self.__particles.n_images):

            # Initialize an empty image:
            particles_with_gaussian_light = np.zeros((self.__particles.size_with_buffer[0], self.__particles.size_with_buffer[1]))

            # Compute particle coordinates on the current image:
            (particle_height_coordinate, particle_width_coordinate) = np.where(self.__particles.particle_positions[i]==1)

            # Establish the peak intensity for each particle depending on its position with respect to the laser beam plane:
            particle_positions_off_laser_plane = laser_beam_thickness * np.random.rand(self.__particles.n_of_particles[i]) - laser_beam_thickness/2
            particle_position_relative_to_laser_centerline = np.abs(particle_positions_off_laser_plane) / (laser_beam_thickness/2)
            particle_peak_intensities = self.exposures_per_image[i] * maximum_intensity * np.exp(-0.5 * (particle_position_relative_to_laser_centerline**2 / laser_beam_shape**2))

            # Add Gaussian blur to each particle location that mimics the particle size:
            for p in range(0,self.__particles.n_of_particles[i]):

                px_c_height = int(np.floor(particle_height_coordinate[p]))
                px_c_width = int(np.floor(particle_width_coordinate[p]))
                ceil_of_particle_radius = int(np.ceil(self.__particles.particle_diameters[i][p]/2))

                for h in range(px_c_height-ceil_of_particle_radius, px_c_height+ceil_of_particle_radius):
                    for w in range(px_c_width-ceil_of_particle_radius, px_c_width+ceil_of_particle_radius):
                        if (h >= 0 and h < self.__particles.size_with_buffer[0]) and (w >= 0 and w < self.__particles.size_with_buffer[1]):

                            coordinate_height = h + 0.5 - particle_height_coordinate[p]
                            coordinate_width = w + 0.5 - particle_width_coordinate[p]
                            particles_with_gaussian_light[h,w] = particles_with_gaussian_light[h,w] + self.compute_light_intensity_at_pixel(particle_peak_intensities[p],
                                                                                                                                            self.__particles.particle_diameters[i][p],
                                                                                                                                            coordinate_height,
                                                                                                                                            coordinate_width,
                                                                                                                                            alpha=alpha)

            images.append(particles_with_gaussian_light)

        self.__images = images

        print('Reflected light added to the image.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_velocity_field(self,
                           flowfield):
        """
        Adds velocity field to the image. The velocity field should be defined using the ``FlowField`` class.

        :param flowfield:
            ``FlowField`` class instance specifying the flow field.
        """

        self.__flowfield = flowfield

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_motion(self,
                   motion):
        """
        Adds particle movement to the image. The movement should be defined using the ``Motion`` class.

        :param motion:
            ``Motion`` class instance specifying the movement of particles from one instance in time to the next.
            In general, the movement is defined by the ``FlowField`` class applied to particles defined by the ``Particle`` class.
        """

        self.__motion = motion

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot(self,
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
        Plots a single, static PIV image, :math:`I_1`, at time :math:`t`.

        :param idx:
            ``int`` specifying the index of the image to plot out of ``n_images`` number of images.
        :param with_buffer:
            ``bool`` specifying whether the buffer for the image size should be plotted. If set to ``False``, the true image size is visualized. If set to ``True``, the image size with an outline that represents the buffer is visualized.
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

        if self.__images is None:

            print('Note: Particles have not been added to the image yet!\n\n')

            fig = plt.figure(figsize=figsize)
            plt.imshow(self.empty_image, cmap=cmap, origin='lower')

        else:

            fig = plt.figure(figsize=figsize)

            # Check if particles were generated with a buffer:

            if self.__particles.size_buffer == 0:

                plt.imshow(self.__images[idx], cmap=cmap, origin='lower')

            else:

                if with_buffer:

                    im = plt.imshow(self.__images[idx], cmap=cmap, origin='lower')

                    # Extend the imshow area with the buffer:
                    f = lambda pixel: pixel - self.__particles.size_buffer
                    im.set_extent([f(x) for x in im.get_extent()])

                    # Visualize a rectangle that separates the proper PIV image area and the artificial buffer outline:
                    rect = patches.Rectangle((0, 0), self.__size[1], self.__size[0], linewidth=1, edgecolor='r', facecolor='none')
                    ax = plt.gca()
                    ax.add_patch(rect)

                else:

                    plt.imshow(self.__images[idx][self.__particles.size_buffer:-self.__particles.size_buffer, self.__particles.size_buffer:-self.__particles.size_buffer], cmap=cmap, origin='lower')

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
                        xlabel=None,
                        ylabel=None,
                        title=None,
                        cmap='Greys_r',
                        figsize=(5,5),
                        dpi=300,
                        filename=None):
        """
        Plots a PIV image pair, :math:`\mathbf{I} = (I_1, I_2)^{\\top}`, at time :math:`t` and :math:`t + \\Delta t` respectively.

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

        if self.__motion is None:

            print('Note: Movement of particles has not been added to the image yet!\n\n')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def plot_velocity_field(self,
                            idx,
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
            if vmin_vmax is not None:
                plt.imshow(self.__flowfield.velocity_field[idx][0], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin='lower')
            else:
                plt.imshow(self.__flowfield.velocity_field[idx][0], cmap=cmap, origin='lower')

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
            if vmin_vmax is not None:
                plt.imshow(self.__flowfield.velocity_field[idx][1], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin='lower')
            else:
                plt.imshow(self.__flowfield.velocity_field[idx][1], cmap=cmap, origin='lower')

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
            if vmin_vmax is not None:
                plt.imshow(self.__flowfield.velocity_field_magnitude[idx], cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], origin='lower')
            else:
                plt.imshow(self.__flowfield.velocity_field_magnitude[idx], cmap=cmap, origin='lower')

            if xlabel is not None:
                plt.xlabel(xlabel)

            if ylabel is not None:
                plt.ylabel(ylabel)

            if title is not None:
                plt.title(title)

            plt.colorbar()

            if add_quiver:
                X = np.arange(0,self.size[1],quiver_step)
                Y = np.arange(0,self.size[0],quiver_step)
                plt.quiver(X, Y, self.__flowfield.velocity_field[idx][0][::quiver_step,::quiver_step], self.__flowfield.velocity_field[idx][1][::quiver_step,::quiver_step], color=quiver_color)

            if add_streamplot:
                X = np.arange(0,self.size[1],1)
                Y = np.arange(0,self.size[0],1)
                plt.streamplot(X, Y, self.__flowfield.velocity_field[idx][0], self.__flowfield.velocity_field[idx][1], density=streamplot_density, color=streamplot_color)

            if filename is not None:

                plt.savefig(filename, dpi=dpi, bbox_inches='tight')

        return plt

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
