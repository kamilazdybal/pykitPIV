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

    @property
    def empty_image(self):
        return self.__empty_image

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_particles(self):

        self.__image = self.__particles.seed_particles()




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def plot_image(image, cmap='Greys', figsize=(8,8)):

    fig = plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)

    return plt