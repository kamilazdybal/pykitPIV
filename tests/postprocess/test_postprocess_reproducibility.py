import unittest
import numpy as np
from pykitPIV import Particle, Image, Postprocess

particles = Particle(1,
                     size=(50, 50),
                     size_buffer=0,
                     diameters=4,
                     densities=0.05,
                     diameter_std=0,
                     seeding_mode='random',
                     random_seed=100)

image = Image(random_seed=100)

image.add_particles(particles)

image.add_reflected_light(exposures=(0.99, 0.99),
                          maximum_intensity=2 ** 16 - 1,
                          laser_beam_thickness=1,
                          laser_over_exposure=1,
                          laser_beam_shape=0.95,
                          alpha=1 / 8)

images_tensor = image.images_I1[:, 0, :, :]
images_tensor_paired = image.concatenate_tensors((image.images_I1,
                                                  image.images_I2))

class TestPostprocessClassReproducibility(unittest.TestCase):

    pass

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
