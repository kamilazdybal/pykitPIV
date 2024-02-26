import unittest
import numpy as np
from pykitPIV import Particle

loop_for = 2

class TestParticleClass(unittest.TestCase):

    def test_particle__Particle__no_random_seed(self):

        particles_1 = Particle(1, random_seed=None)
        particles_2 = Particle(1, random_seed=None)

        self.assertTrue(particles_1.particle_coordinates[0][0].shape[0] != particles_2.particle_coordinates[0][0].shape[0])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__with_random_seed(self):

        particles_1 = Particle(1, random_seed=10)
        particles_2 = Particle(1, random_seed=10)

        self.assertTrue(
            particles_1.particle_coordinates[0][0].shape[0] == particles_2.particle_coordinates[0][0].shape[0])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
