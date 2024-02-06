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
                 size=(512,512),
                 ):

        if type(size) != tuple:
            raise ValueError("Parameter `size` has to of type 'tuple'.")

        if len(size) != 2:
            raise ValueError("Parameter `size` has to have two elements.")

        # Class init:
        self.__size = size

        # Create empty image at class init:
        self.image = np.zeros((size[0], size[1]))

    @property
    def size(self):
        return self.__size

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_particles(self, particles):

        self.image = particles




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def plot_image(image, cmap='Greys', figsize=(8,8)):

    fig = plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)

    return plt