import unittest
import numpy as np
from pykitPIV import Particle

loop_for = 2

n_images = 1
size = (20, 20)
size_buffer = 1

class TestParticleClassReproducibility(unittest.TestCase):

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Random seed set to None
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__no_random_seed_diameter_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        self.assertTrue(not np.array_equal(particles_1.diameter_per_image, particles_2.diameter_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__no_random_seed_distance_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        self.assertTrue(not np.array_equal(particles_1.distance_per_image, particles_2.distance_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__no_random_seed_n_particles_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.01,0.6),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.01,0.6),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        self.assertTrue(not np.array_equal(np.array(particles_1.n_of_particles), np.array(particles_2.n_of_particles)))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__no_random_seed_density_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        self.assertTrue(not np.array_equal(particles_1.density_per_image, particles_2.density_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__no_random_seed_diameter_std_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        self.assertTrue(not np.array_equal(particles_1.diameter_std_per_image, particles_2.diameter_std_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__no_random_seed_particle_coordinates(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(2, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(2, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        self.assertTrue(not np.array_equal(particles_1.particle_coordinates[0][0], particles_2.particle_coordinates[0][0]))
        self.assertTrue(not np.array_equal(particles_1.particle_coordinates[0][1], particles_2.particle_coordinates[0][1]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__no_random_seed_particle_positions(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        self.assertTrue(not np.array_equal(particles_1.particle_positions[0,0,:,:], particles_2.particle_positions[0,0,:,:]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__no_random_seed_particle_diameters(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=None)

        self.assertTrue(not np.array_equal(particles_1.particle_diameters[0][0], particles_2.particle_diameters[0][0]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Random seed set to two different values
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__different_random_seed_diameter_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=200)

        self.assertTrue(not np.array_equal(particles_1.diameter_per_image, particles_2.diameter_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__different_random_seed_distance_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=200)

        self.assertTrue(not np.array_equal(particles_1.distance_per_image, particles_2.distance_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__different_random_seed_n_particles_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.01,0.6),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.01,0.6),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=200)

        self.assertTrue(not np.array_equal(np.array(particles_1.n_of_particles), np.array(particles_2.n_of_particles)))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__different_random_seed_density_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=200)

        self.assertTrue(not np.array_equal(particles_1.density_per_image, particles_2.density_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__different_random_seed_diameter_std_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=200)

        self.assertTrue(not np.array_equal(particles_1.diameter_std_per_image, particles_2.diameter_std_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__different_random_seed_particle_coordinates(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(2, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(2, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=200)

        self.assertTrue(not np.array_equal(particles_1.particle_coordinates[0][0], particles_2.particle_coordinates[0][0]))
        self.assertTrue(not np.array_equal(particles_1.particle_coordinates[0][1], particles_2.particle_coordinates[0][1]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__different_random_seed_particle_positions(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=200)

        self.assertTrue(not np.array_equal(particles_1.particle_positions[0,0,:,:], particles_2.particle_positions[0,0,:,:]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__different_random_seed_particle_diameters(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=200)

        self.assertTrue(not np.array_equal(particles_1.particle_diameters[0][0], particles_2.particle_diameters[0][0]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Random seed set to the same value
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_diameter_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(np.array_equal(particles_1.diameter_per_image, particles_2.diameter_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_distance_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(np.array_equal(particles_1.distance_per_image, particles_2.distance_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_n_particles_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.01,0.6),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.01,0.6),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(np.array_equal(np.array(particles_1.n_of_particles), np.array(particles_2.n_of_particles)))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_density_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(np.array_equal(particles_1.density_per_image, particles_2.density_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_diameter_std_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(np.array_equal(particles_1.diameter_std_per_image, particles_2.diameter_std_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_particle_coordinates(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(2, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(2, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(np.array_equal(particles_1.particle_coordinates[0][0], particles_2.particle_coordinates[0][0]))
        self.assertTrue(np.array_equal(particles_1.particle_coordinates[0][1], particles_2.particle_coordinates[0][1]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_particle_positions(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(np.array_equal(particles_1.particle_positions[0,0,:,:], particles_2.particle_positions[0,0,:,:]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_particle_diameters(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(np.array_equal(particles_1.particle_diameters[0][0], particles_2.particle_diameters[0][0]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Random seed set to the same value but different spread
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_different_spread_diameter_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1,2),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,4),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(not np.array_equal(particles_1.diameter_per_image, particles_2.diameter_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_different_spread_distance_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,1),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(1.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(not np.array_equal(particles_1.distance_per_image, particles_2.distance_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_different_spread_n_particles_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.01,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.2,0.6),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(not np.array_equal(np.array(particles_1.n_of_particles), np.array(particles_2.n_of_particles)))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_different_spread_density_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.01,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.2,0.6),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(not np.array_equal(particles_1.density_per_image, particles_2.density_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_different_spread_diameter_std_per_image(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0,0.1),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3,6),
                               distances=(0.5,2),
                               densities=(0.05,0.1),
                               diameter_std=(0.2,0.5),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(not np.array_equal(particles_1.diameter_std_per_image, particles_2.diameter_std_per_image))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_different_spread_particle_coordinates(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(2, 2),
                               distances=(2, 2),
                               densities=(0.01, 0.1),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(2, 2),
                               distances=(2, 2),
                               densities=(0.2, 0.3),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(not np.array_equal(particles_1.particle_coordinates[0][0], particles_2.particle_coordinates[0][0]))
        self.assertTrue(not np.array_equal(particles_1.particle_coordinates[0][1], particles_2.particle_coordinates[0][1]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_different_spread_particle_positions(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.1),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.2, 0.3),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(not np.array_equal(particles_1.particle_positions[0,0,:,:], particles_2.particle_positions[0,0,:,:]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__same_random_seed_different_spread_particle_diameters(self):

        particles_1 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(1, 2),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        particles_2 = Particle(n_images,
                               size=size,
                               size_buffer=size_buffer,
                               diameters=(3, 4),
                               distances=(2, 2),
                               densities=(0.05, 0.05),
                               diameter_std=(0, 0),
                               min_diameter=1e-2,
                               seeding_mode='random',
                               random_seed=100)

        self.assertTrue(not np.array_equal(particles_1.particle_diameters[0][0], particles_2.particle_diameters[0][0]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
