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
####    Class: Camera
####
################################################################################
################################################################################

class Camera:

    def __init__(self,
                 laser_beam_thickness=2,
                 laser_beam_shape=0.85,
                 over_exposure=1,
                 light_enhancement_factor=(0.02,0.8),
                 maximum_intensity=2**16-1,
                 ):

        # Class init:
        self.__laser_beam_thickness = laser_beam_thickness
        self.__laser_beam_shape = laser_beam_shape
        self.__over_exposure = over_exposure
        self.__light_enhancement_factor = light_enhancement_factor
        self.__maximum_intensity = maximum_intensity

        self.__LEF = self.__light_enhancement_factor[0] + np.random.rand(self.__n_images) * (self.__light_enhancement_factor[1] - self.__light_enhancement_factor[0])
        self.__noise_s = self.__LEF / self.__SNR # Not sure yet what that is...

    @property
    def laser_beam_thickness(self):
        return self.__laser_beam_thickness

    @property
    def laser_beam_shape(self):
        return self.__laser_beam_shape

    @property
    def over_exposure(self):
        return self.__over_exposure

    @property
    def light_enhancement_factor(self):
        return self.__light_enhancement_factor

    @property
    def maximum_intensity(self):
        return self.__maximum_intensity