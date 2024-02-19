import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import scipy
import warnings
from pykitPIV.checks import *

################################################################################
################################################################################
####
####    Class: Motion
####
################################################################################
################################################################################

class Motion:
    """
    Applies movement defined by the ``FlowField`` class instance to particles defined by the ``Particle`` class instance.

    :param particles:
        ``Particle`` class instance specifying the properties and positions of particles.
    :param flowfield:
        ``FlowField`` class instance specifying the flow field.
    :param particle_loss: (optional)
        ``tuple`` of two numerical elements specifying the minimum (first element) and maximum (second element) percentage of lost particles.
        Between two consecutive image pairs, this percentage of particles will be randomly removed and replaced due to movement of particles off the laser plane.
    :param warp_images:
        ``str`` specifying the method to warp PIV images.
    """

    def __init__(self,
                 particles,
                 flowfield,
                 particle_loss=(0, 2),
                 warp_images='forward',
                 n_steps=5,
                 method='SINC-8'):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if particles.size_with_buffer != flowfield.size_with_buffer:
            raise ValueError(f"Inconsistent PIV image sizes between `Particle` class instance {particles.size_with_buffer} and `FlowField` class instance {flowfield.size_with_buffer}.")

        check_two_element_tuple(particle_loss, 'particle_loss')
        check_min_max_tuple(particle_loss, 'particle_loss')

        __warp_images = ['forward', 'symmetric', 'two-step-forward']
        __method = ['cubic-interpolation', 'SINC-8']

        if warp_images not in __warp_images:
            raise ValueError("Parameter `warp_images` has to be 'forward', 'random-symmetric', or 'two-step-forward'.")

        if method not in __method:
            raise ValueError("Parameter `method` has to be 'cubic-interpolation', or 'SINC-8''.")

        # Class init:
        self.__particle_loss = particle_loss
        self.__warp_images = warp_images
        self.__n_steps = n_steps
        self.__method = method

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        @property
        def particle_loss(self):
            return self.__particle_loss

        @property
        def warp_images(self):
            return self.__warp_images

        @property
        def n_steps(self):
            return self.__n_steps

        @property
        def method(self):
            return self.__method
