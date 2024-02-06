import numpy as np
import pandas as pd
import random
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

    def __init__(self,
                 n_images,
                 buffer=10,
                 size=(512,512),
                 signal_to_noise=(5,20),
                 particle_diameter=(3,6),
                 particle_distance=(0.5,2),
                 particle_std=0.1,
                 particle_density=(0.05,0.1),
                 laser_beam_thickness=2,
                 laser_beam_shape=0.85,
                 over_exposure=1,
                 light_enhancement_factor=(0.02,0.8),
                 maximum_intensity=2**16-1,
                 warp_images='forward',
                 n_steps=5,
                 method='SINC-8'):

        __warp_images = ['forward', 'symmetric', 'two-step-forward']
        __method = ['cubic-interpolation', 'SINC-8']

        if warp_images not in __warp_images:
            raise ValueError("Parameter `warp_images` has to be 'forward', 'random-symmetric', or 'two-step-forward'.")
        
        if method not in __method:
            raise ValueError("Parameter `method` has to be 'cubic-interpolation', or 'SINC-8''.")

        # Class init:
        self.__n_images = n_images
        self.__buffer = buffer
        self.__size = size
        self.__signal_to_noise = signal_to_noise
        self.__particle_diameter = particle_diameter
        self.__particle_distance = particle_distance
        self.__particle_std = particle_std
        self.__particle_density = particle_density
        self.__laser_beam_thickness = laser_beam_thickness
        self.__laser_beam_shape = laser_beam_shape
        self.__over_exposure = over_exposure
        self.__light_enhancement_factor = light_enhancement_factor
        self.__maximum_intensity = maximum_intensity
        self.__warp_images = warp_images
        self.__n_steps = n_steps
        self.__method = method

        # Initialize parameters for image generation:
        self.__D = np.random.rand(self.__n_images) * (self.__particle_diameter[0] - self.__particle_diameter[1]) + self.__particle_diameter[0]
        self.__L = np.random.rand(self.__n_images) * (self.__particle_distance[0] - self.__particle_distance[1]) + self.__particle_distance[0]
        self.__SNR = self.__signal_to_noise[0] + np.random.rand(self.__n_images) * (self.__signal_to_noise[1] - self.__signal_to_noise[0])
        self.__LEF = self.__light_enhancement_factor[0] + np.random.rand(self.__n_images) * (self.__light_enhancement_factor[1] - self.__light_enhancement_factor[0])
        self.__noise_s = self.__LEF / self.__SNR # Not sure yet what that is...

        # Quantities computed later:
        self.__seeded_particle_density = None

    @property
    def D(self):
        return self.__D

    @property
    def L(self):
        return self.__L
        
    @property
    def SNR(self):
        return self.__SNR
   
    @property
    def LEF(self):
        return self.__LEF

    @property
    def noise_s(self):
        return self.__noise_s
    
    @property
    def n_images(self):
        return self.__n_images
    
    @property
    def buffer(self):
        return self.__buffer
    
    @property
    def size(self):
        return self.__size
            
    @property
    def signal_to_noise(self):
        return self.__signal_to_noise
        
    @property
    def particle_diameter(self):
        return self.__particle_diameter    
        
    @property
    def particle_distance(self):
        return self.__particle_distance
        
    @property
    def particle_std(self):
        return self.__particle_std
        
    @property
    def particle_density(self):
        return self.__particle_density
        
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
        
    @property
    def warp_images(self):
        return self.__warp_images
        
    @property
    def n_steps(self):
        return self.__n_steps
        
    @property
    def method(self):
        return self.__method
        
    @property
    def seeded_particle_density(self):
        return self.__seeded_particle_density

    def seed_particles(self, particle_seeding_mode):

        __particle_seeding_mode = ['poisson', 'random']
        
        if particle_seeding_mode not in __particle_seeding_mode:
            raise ValueError("Parameter `__particle_seeding_mode` has to be 'poisson', or 'random'.")
        
        if particle_seeding_mode == 'poisson':
            
            seeded_particle_density = np.zeros((self.n_images,))
            
            for i in range(0, self.n_images):
                
                sx = np.arange(((self.D[i] + self.L[i])/2), (self.size[1] - (self.D[i] + self.L[i])/2), (self.D[i] + self.L[i]))
                sy = np.arange(((self.D[i] + self.L[i])/2), (self.size[0] - (self.D[i] + self.L[i])/2), (self.D[i] + self.L[i]))
                seeded_particle_density[i] = len(sx) * len(sy) / self.size[1] / self.size[0]

        elif particle_seeding_mode == 'random':
            
            seeded_particle_density = np.random.rand(self.__n_images) * (self.particle_density[1] - self.particle_density[0]) + self.particle_density[0]
            n_par = self.size[1] * self.size[0] * seeded_particle_density # Don't know what that's for yet...

        self.__seeded_particle_density = seeded_particle_density
        
        return seeded_particle_density