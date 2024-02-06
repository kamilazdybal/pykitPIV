import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import scipy
import warnings

################################################################################
################################################################################
####
####    Class: Motion
####
################################################################################
################################################################################

class Motion:

    def __init__(self,
                 lost_particles=(0, 20),
                 warp_images='forward',
                 n_steps=5,
                 method='SINC-8',
                 ):

        __warp_images = ['forward', 'symmetric', 'two-step-forward']
        __method = ['cubic-interpolation', 'SINC-8']

        if warp_images not in __warp_images:
            raise ValueError("Parameter `warp_images` has to be 'forward', 'random-symmetric', or 'two-step-forward'.")

        if method not in __method:
            raise ValueError("Parameter `method` has to be 'cubic-interpolation', or 'SINC-8''.")

        # Class init:
        self.__warp_images = warp_images
        self.__n_steps = n_steps
        self.__method = method

        @property
        def warp_images(self):
            return self.__warp_images

        @property
        def n_steps(self):
            return self.__n_steps

        @property
        def method(self):
            return self.__method
