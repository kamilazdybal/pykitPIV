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
####    Class: FlowField
####
################################################################################
################################################################################

class FlowField:

    def __init__(self, 
                 flow_mode='random',
                 filter_size=(10,300),
                 sin_period=(30,300),
                 displacement=(0,10),

                 n_gaussian_filter_iter=6):

        __flow_mode = ['random', 'random-sinusoidal', 'quadrant', 'checkerboard']

        if flow_mode not in __flow_mode:
            raise ValueError("Parameter `flow_mode` has to be 'random', 'random-sinusoidal', 'quadrant', or 'checkerboard'.")
        
        self.__flow_mode = flow_mode
        self.__filter_size = filter_size, 
        self.__sin_period = sin_period, 
        self.__displacement = displacement, 
        self.__lost_particles = lost_particles, 
        self.__n_gaussian_filter_iter = n_gaussian_filter_iter

    @property
    def flow_mode(self):
        return self.__flow_mode
        
    @property
    def filter_size(self):
        return self.__filter_size

    @property
    def sin_period(self):
        return self.__sin_period

    @property
    def displacement(self):
        return self.__displacement
        
    @property
    def lost_particles(self):
        return self.__lost_particles

    @property
    def n_gaussian_filter_iter(self):
        return self.__n_gaussian_filter_iter


    def generate_field(self, ):


        pass
