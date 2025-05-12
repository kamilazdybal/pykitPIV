import unittest
import numpy as np
from pykitPIV import Particle, FlowField, Motion

class TestMotionClass(unittest.TestCase):

    def test_motion__Motion__allowed_calls(self):

        particles = Particle(1)
        flowfield = FlowField(1)

        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(0, 10))

        try:
            motion = Motion(particles, flowfield)
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_motion__Motion__not_allowed_calls(self):

        particles = Particle(1)
        flowfield = FlowField(1)

        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(0, 10))

        with self.assertRaises(ValueError):
            motion = Motion(particles, flowfield, particle_loss=[])

        with self.assertRaises(ValueError):
            motion = Motion(particles, flowfield, particle_loss=(10,2))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_motion__Motion__no_velocity_field_generated(self):

        particles = Particle(1)
        flowfield = FlowField(1)

        with self.assertRaises(AttributeError):
            motion = Motion(particles, flowfield)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_motion__Motion__attributes_available_after_class_init(self):

        particles = Particle(1)
        flowfield = FlowField(1)

        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(0, 10))

        motion = Motion(particles,
                        flowfield,
                        particle_loss=(1, 2),
                        particle_gain=(1, 2),
                        verbose=False,
                        random_seed=None
                        )

        # Attributes coming from user input:
        try:
            motion.particle_loss
            motion.particle_gain
            motion.random_seed
        except Exception:
            self.assertTrue(False)

        # Attributes computed at class init:
        try:
            motion.loss_percentage_per_image
            motion.gain_percentage_per_image
            motion.particle_coordinates_I1
            motion.particle_coordinates_I2
            motion.updated_particle_diameters
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_motion__Motion__allowed_attribute_set(self):

        particles = Particle(1)
        flowfield = FlowField(1)

        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(0, 10))

        motion = Motion(particles,
                        flowfield,
                        particle_loss=2,
                        particle_gain=3)

        self.assertTrue(motion.particle_loss[0] == 2)
        self.assertTrue(motion.particle_loss[1] == 2)
        self.assertTrue(motion.particle_gain[0] == 3)
        self.assertTrue(motion.particle_gain[1] == 3)

        # Re-set:
        motion.particle_loss = 4
        motion.particle_gain = 1

        self.assertTrue(motion.particle_loss[0] == 4)
        self.assertTrue(motion.particle_loss[1] == 4)
        self.assertTrue(motion.particle_gain[0] == 1)
        self.assertTrue(motion.particle_gain[1] == 1)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_motion__Motion__not_allowed_attribute_set(self):

        particles = Particle(1)
        flowfield = FlowField(1)

        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(0, 10))

        motion = Motion(particles, flowfield)

        with self.assertRaises(AttributeError):
            motion.random_seed = 1

        with self.assertRaises(AttributeError):
            motion.particle_coordinates_I1 = 1

        with self.assertRaises(AttributeError):
            motion.particle_coordinates_I2 = 2

        with self.assertRaises(AttributeError):
            motion.loss_percentage_per_image = 2

        with self.assertRaises(AttributeError):
            motion.gain_percentage_per_image = 2

        with self.assertRaises(AttributeError):
            motion.updated_particle_diameters = 2

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_motion__Motion__removal_of_particles(self):

        # Scenario 1
        particles = Particle(1, size_buffer=20)
        flowfield = FlowField(1, size_buffer=20)

        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(0, 10))

        motion = Motion(particles,
                        flowfield,
                        particle_loss=(20,20),
                        particle_gain=(0,0),
                        random_seed=100)

        motion.forward_euler(n_steps=10)

        size_of_diameters = motion.updated_particle_diameters[0].shape
        size_of_coordinates_y = motion.particle_coordinates_I2[0][0].shape
        size_of_coordinates_x = motion.particle_coordinates_I2[0][1].shape

        self.assertTrue(size_of_diameters[0] == size_of_coordinates_y[0])
        self.assertTrue(size_of_diameters[0] == size_of_coordinates_x[0])

        motion_no_loss = Motion(particles,
                                flowfield,
                                particle_loss=(0,0),
                                particle_gain=(0,0),
                                random_seed=100)

        motion_no_loss.forward_euler(n_steps=10)

        # Check that particles have been removed:

        self.assertTrue(motion.updated_particle_diameters[0].shape != motion_no_loss.updated_particle_diameters[0].shape)

        motion_with_loss_again = Motion(particles,
                                        flowfield,
                                        particle_loss=(20,20),
                                        particle_gain=(0,0),
                                        random_seed=100)

        motion_with_loss_again.forward_euler(n_steps=10)

        self.assertTrue(motion.updated_particle_diameters[0].shape == motion_with_loss_again.updated_particle_diameters[0].shape)

        # Scenario 2
        particles = Particle(1, size_buffer=2)
        flowfield = FlowField(1, size_buffer=2)

        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(0, 10))

        motion = Motion(particles, flowfield)
        motion.forward_euler(n_steps=10)

        size_of_diameters = motion.updated_particle_diameters[0].shape
        size_of_coordinates_y = motion.particle_coordinates_I2[0][0].shape
        size_of_coordinates_x = motion.particle_coordinates_I2[0][1].shape

        self.assertTrue(size_of_diameters[0] == size_of_coordinates_y[0])
        self.assertTrue(size_of_diameters[0] == size_of_coordinates_x[0])

        # Scenario 3
        particles = Particle(1, size_buffer=10)
        flowfield = FlowField(1, size_buffer=10)

        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(0, 10))

        motion = Motion(particles, flowfield)
        motion.forward_euler(n_steps=10)

        size_of_diameters = motion.updated_particle_diameters[0].shape
        size_of_coordinates_y = motion.particle_coordinates_I2[0][0].shape
        size_of_coordinates_x = motion.particle_coordinates_I2[0][1].shape

        self.assertTrue(size_of_diameters[0] == size_of_coordinates_y[0])
        self.assertTrue(size_of_diameters[0] == size_of_coordinates_x[0])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_motion__Motion__addition_of_particles(self):

        # Scenario 1
        particles = Particle(1, size_buffer=20)
        flowfield = FlowField(1, size_buffer=20)

        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(0, 10))

        motion = Motion(particles,
                        flowfield,
                        particle_loss=(0,0),
                        particle_gain=(0,0),
                        random_seed=100)

        motion.forward_euler(n_steps=10)

        motion_with_gain = Motion(particles,
                                flowfield,
                                particle_loss=(20,20),
                                particle_gain='matching',
                                random_seed=100)

        motion_with_gain.forward_euler(n_steps=10)

        self.assertTrue(motion.updated_particle_diameters[0].shape == motion_with_gain.updated_particle_diameters[0].shape)

        motion_with_non_matching_gain = Motion(particles,
                                flowfield,
                                particle_loss=(20,20),
                                particle_gain=(10,10),
                                random_seed=100)

        motion_with_non_matching_gain.forward_euler(n_steps=10)

        self.assertTrue(motion.updated_particle_diameters[0].shape > motion_with_non_matching_gain.updated_particle_diameters[0].shape)

        motion_with_non_matching_loss = Motion(particles,
                                                flowfield,
                                                particle_loss=(0,0),
                                                particle_gain=(20,20),
                                                random_seed=100)

        motion_with_non_matching_loss.forward_euler(n_steps=10)

        self.assertTrue(motion.updated_particle_diameters[0].shape < motion_with_non_matching_loss.updated_particle_diameters[0].shape)
        self.assertTrue(motion_with_non_matching_gain.updated_particle_diameters[0].shape < motion_with_non_matching_loss.updated_particle_diameters[0].shape)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
