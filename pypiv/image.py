import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy
import warnings

################################################################################
################################################################################
####
####    Class: Image
####
################################################################################
################################################################################

class Image:
    """
    Stores and plots synthetic PIV images and the associated flow-fields at any stage of particle generation and movement.
    """

    def __init__(self,
                 particles
                 ):

        self.__particles = particles

        # Create empty image at class init:
        self.__empty_image = np.zeros((particles.size[0], particles.size[1]))

        self.__images = None

    @property
    def empty_image(self):
        return self.__empty_image

    @property
    def images(self):
        return self.__images

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_particles(self):

        self.__images = self.__particles.particle_positions

        print('Particles added to the image.')

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
