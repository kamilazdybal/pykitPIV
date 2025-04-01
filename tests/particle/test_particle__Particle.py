import unittest
import numpy as np
from pykitPIV import Particle, Image

loop_for = 2

class TestParticleClass(unittest.TestCase):

    def test_particle__Particle__allowed_calls(self):

        try:
            particles = Particle(1)
        except Exception:
            self.assertTrue(False)

        try:
            particles = Particle(10)
        except Exception:
            self.assertTrue(False)

        try:
            particles = Particle(1,
                                 size=(20, 20),
                                 size_buffer=10,
                                 densities=(0.1, 0.11),
                                 diameters=(6, 10),
                                 distances=(1, 1),
                                 diameter_std=1,
                                 seeding_mode='random',
                                 random_seed=None)
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__attributes_available_after_class_init(self):

        particles = Particle(1)

        # Attributes coming from user input:
        try:
            particles.n_images
            particles.size
            particles.size_buffer
            particles.diameters
            particles.distances
            particles.densities
            particles.diameter_std
            particles.seeding_mode
            particles.random_seed
        except Exception:
            self.assertTrue(False)

        # Attributes computed at class init:
        try:
            particles.size_with_buffer
            particles.diameter_per_image
            particles.distance_per_image
            particles.density_per_image
            particles.n_of_particles
            particles.particle_positions
            particles.particle_diameters
        except Exception:
            self.assertTrue(False)

        self.assertTrue(isinstance(particles.size_with_buffer, tuple))
        self.assertTrue(isinstance(particles.diameter_per_image, np.ndarray))
        self.assertTrue(isinstance(particles.distance_per_image, np.ndarray))
        self.assertTrue(isinstance(particles.density_per_image, np.ndarray))

        self.assertTrue(isinstance(particles.n_of_particles, list))
        self.assertTrue(isinstance(particles.particle_positions, np.ndarray))
        self.assertTrue(isinstance(particles.particle_diameters, list))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__not_allowed_calls(self):

        # Wrong type:
        with self.assertRaises(ValueError):
            particles = Particle([])

        with self.assertRaises(ValueError):
            particles = Particle(1, size=[])

        with self.assertRaises(ValueError):
            particles = Particle(1, size_buffer=[])

        with self.assertRaises(ValueError):
            particles = Particle(1, diameters=[])

        with self.assertRaises(ValueError):
            particles = Particle(1, distances=[])

        with self.assertRaises(ValueError):
            particles = Particle(1, densities=[])

        with self.assertRaises(ValueError):
            particles = Particle(1, diameter_std=[])

        with self.assertRaises(ValueError):
            particles = Particle(1, seeding_mode=[])

        with self.assertRaises(ValueError):
            particles = Particle(1, random_seed=[])

        # Not a two-element tuple:
        with self.assertRaises(ValueError):
            particles = Particle(1, size=(10,))

        with self.assertRaises(ValueError):
            particles = Particle(1, diameters=(10,))

        with self.assertRaises(ValueError):
            particles = Particle(1, distances=(10,))

        with self.assertRaises(ValueError):
            particles = Particle(1, densities=(10,))

        # Not a min-max tuple:
        with self.assertRaises(ValueError):
            particles = Particle(1, diameters=(10,1))

        with self.assertRaises(ValueError):
            particles = Particle(1, distances=(10,1))

        with self.assertRaises(ValueError):
            particles = Particle(1, densities=(10, 1))

        # Not allowed string:
        with self.assertRaises(ValueError):
            particles = Particle(1, seeding_mode='Test')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__not_allowed_attribute_set(self):

        particles = Particle(1)

        with self.assertRaises(AttributeError):
            particles.n_images = 10

        with self.assertRaises(AttributeError):
            particles.size = (512, 512)

        with self.assertRaises(AttributeError):
            particles.size_buffer = 20

        with self.assertRaises(AttributeError):
            particles.diameters = (3, 6)

        with self.assertRaises(AttributeError):
            particles.distances = (0.5, 2)

        with self.assertRaises(AttributeError):
            particles.densities = (0.05, 0.1)

        with self.assertRaises(AttributeError):
            particles.diameter_std = 0.1

        with self.assertRaises(AttributeError):
            particles.seeding_mode = 'random'

        with self.assertRaises(AttributeError):
            particles.random_seed = 100

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__diameter_per_image(self):

        for i in range(0,loop_for):

            particles = Particle(100,
                                 size=(200, 200),
                                 diameters=(6.5, 10))

            self.assertTrue(np.min(particles.diameter_per_image) >= 6.5)
            self.assertTrue(np.max(particles.diameter_per_image) <= 10)

        for i in range(0,loop_for):

            particles = Particle(100,
                                 size=(200, 200),
                                 diameters=(1, 8.5))

            self.assertTrue(np.min(particles.diameter_per_image) >= 1)
            self.assertTrue(np.max(particles.diameter_per_image) <= 8.5)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__distance_per_image(self):

        for i in range(0,loop_for):

            particles = Particle(100,
                                 size=(200, 200),
                                 distances=(6, 10.5))

            self.assertTrue(np.min(particles.distance_per_image) >= 6)
            self.assertTrue(np.max(particles.distance_per_image) <= 10.5)

        for i in range(0,loop_for):

            particles = Particle(100,
                                 size=(200, 200),
                                 distances=(1.5, 8))

            self.assertTrue(np.min(particles.distance_per_image) >= 1.5)
            self.assertTrue(np.max(particles.distance_per_image) <= 8)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__density_per_image(self):

        for i in range(0,loop_for):

            particles = Particle(100,
                                 size=(200, 200),
                                 densities=(0.01, 0.05))

            self.assertTrue(np.min(particles.density_per_image) >= 0.01)
            self.assertTrue(np.max(particles.density_per_image) <= 0.05)

        for i in range(0,loop_for):

            particles = Particle(100,
                                 size=(200, 200),
                                 densities=(0.1, 0.5))

            self.assertTrue(np.min(particles.density_per_image) >= 0.1)
            self.assertTrue(np.max(particles.density_per_image) <= 0.5)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__particle_diameters_std(self):

        tolerance = 0.05

        # Check that the standard deviation for particle diameters is within the specified tolerance:

        diameter_std = 1

        for i in range(0,loop_for):

            particles = Particle(1,
                                 size=(200, 200),
                                 densities=(0.1, 0.11),
                                 diameters=(6, 10),
                                 diameter_std=diameter_std,
                                 seeding_mode='random')

            actual_std = np.std(particles.particle_diameters[0])

            self.assertTrue(np.abs(actual_std - diameter_std) <= tolerance)

        diameter_std = 0.1

        for i in range(0,loop_for):

            particles = Particle(1,
                                 size=(200, 200),
                                 densities=(0.1, 0.11),
                                 diameters=(6, 10),
                                 diameter_std=diameter_std,
                                 seeding_mode='random')

            actual_std = np.std(particles.particle_diameters[0])

            self.assertTrue(np.abs(actual_std - diameter_std) <= tolerance)

        diameter_std = 0.01

        for i in range(0,loop_for):

            particles = Particle(1,
                                 size=(200, 200),
                                 densities=(0.1, 0.11),
                                 diameters=(6, 10),
                                 diameter_std=diameter_std,
                                 seeding_mode='random')

            actual_std = np.std(particles.particle_diameters[0])

            self.assertTrue(np.abs(actual_std - diameter_std) <= tolerance)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__seeding_density(self):

        particles_1 = Particle(1,
                             size=(200, 200),
                             densities=(0.1, 0.11),
                             seeding_mode='random')

        particles_2 = Particle(1,
                             size=(200, 200),
                             densities=(0.2, 0.21),
                             seeding_mode='random')

        self.assertTrue(particles_1.n_of_particles < particles_2.n_of_particles)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__seeding_density_quantitatively(self):

        sizes = [(10,10), (50,50), (100,100), (200,200)]

        seeding_densities = [0.01 * i for i in range(0,20)]

        for seeding_density in seeding_densities:

            for size in sizes:

                particles_1 = Particle(1,
                                       size=size,
                                       densities=(seeding_density, seeding_density),
                                       seeding_mode='random')

                particle_positions = particles_1.particle_positions

                n_particles = np.sum(particle_positions[0,0,:,:])

                image_area = size[0] * size[1]

                self.assertTrue(n_particles==int(image_area * seeding_density))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__adding_image_buffers(self):

        particles = Particle(1,
                             size=(100, 200),
                             size_buffer=10,
                             densities=(0.1, 0.11),
                             seeding_mode='random')

        self.assertTrue(particles.size_with_buffer[0] == 120)
        self.assertTrue(particles.size_with_buffer[1] == 220)
        self.assertTrue(particles.particle_positions.shape[2] == 120)
        self.assertTrue(particles.particle_positions.shape[3] == 220)

        particles = Particle(1,
                             size=(100, 200),
                             size_buffer=0,
                             densities=(0.1, 0.11),
                             seeding_mode='random')

        self.assertTrue(particles.size_with_buffer[0] == 100)
        self.assertTrue(particles.size_with_buffer[1] == 200)
        self.assertTrue(particles.particle_positions.shape[2] == 100)
        self.assertTrue(particles.particle_positions.shape[3] == 200)

        particles = Particle(1,
                             size=(100, 200),
                             size_buffer=2,
                             densities=(0.1, 0.11),
                             seeding_mode='random')

        self.assertTrue(particles.size_with_buffer[0] == 104)
        self.assertTrue(particles.size_with_buffer[1] == 204)
        self.assertTrue(particles.particle_positions.shape[2] == 104)
        self.assertTrue(particles.particle_positions.shape[3] == 204)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__particle_diameters(self):

        particles = Particle(1,
                             size=(200,200),
                             size_buffer=10,
                             diameters=(1, 1),
                             distances=(1, 2),
                             densities=(0.1, 0.1),
                             diameter_std=1,
                             min_diameter=0,
                             seeding_mode='random',
                             random_seed=None)

        self.assertTrue(np.all(particles.particle_diameters[0] >= 0))

        particles = Particle(1,
                             size=(200,200),
                             size_buffer=10,
                             diameters=(1, 1),
                             distances=(1, 2),
                             densities=(0.1, 0.1),
                             diameter_std=1,
                             min_diameter=0.4,
                             seeding_mode='random',
                             random_seed=None)

        self.assertTrue(np.all(particles.particle_diameters[0] >= 0.4))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__upload_particle_coordinates(self):

        particles_1 = Particle(1,
                             size=(200, 200),
                             densities=(0.1, 0.1),
                             seeding_mode='random')

        particles_2 = Particle(1,
                             size=(200, 200),
                             densities=(0.1, 0.1),
                             seeding_mode='random')

        # Check that x-coordinates are not the same:
        self.assertTrue(np.all(particles_1.particle_coordinates[0][0] != particles_2.particle_coordinates[0][0]))

        # Check that y-coordinates are not the same:
        self.assertTrue(np.all(particles_1.particle_coordinates[0][1] != particles_2.particle_coordinates[0][1]))

        # Upload particle coordinates:
        particles_1.upload_particle_coordinates(particles_2.particle_coordinates)

        # Check that x-coordinates are the same:
        self.assertTrue(np.all(particles_1.particle_coordinates[0][0] == particles_2.particle_coordinates[0][0]))

        # Check that y-coordinates are the same:
        self.assertTrue(np.all(particles_1.particle_coordinates[0][1] == particles_2.particle_coordinates[0][1]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_particle__Particle__upload_particle_coordinates_to_empty_object(self):

        particles_1 = Particle(1,
                               size=(200, 200),
                               diameters=(2, 4),
                               diameter_std=1,
                               seeding_mode='user')

        self.assertTrue(particles_1.particle_coordinates is None)
        self.assertTrue(particles_1.particle_positions is None)
        self.assertTrue(particles_1.particle_diameters is None)
        self.assertTrue(particles_1.n_of_particles is None)

        particles_2 = Particle(1,
                               size=(200, 200),
                               densities=(0.1, 0.1),
                               seeding_mode='random')

        # Upload particle coordinates:
        particles_1.upload_particle_coordinates(particles_2.particle_coordinates)

        self.assertTrue(particles_1.particle_coordinates is not None)
        self.assertTrue(particles_1.particle_positions is not None)
        self.assertTrue(particles_1.particle_diameters is not None)
        self.assertTrue(particles_1.n_of_particles is not None)

        # Check that x-coordinates are the same:
        self.assertTrue(np.all(particles_1.particle_coordinates[0][0] == particles_2.particle_coordinates[0][0]))

        # Check that y-coordinates are the same:
        self.assertTrue(np.all(particles_1.particle_coordinates[0][1] == particles_2.particle_coordinates[0][1]))

        # Check that the number of particles are the same:
        self.assertTrue(particles_1.n_of_particles == particles_2.n_of_particles)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
