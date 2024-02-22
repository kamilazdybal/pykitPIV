import unittest
import numpy as np
from pykitPIV import Particle, FlowField, Motion

class TestMotionClass(unittest.TestCase):

    def test_motion__Motion__allowed_calls(self):

        particles = Particle(1)
        flowfield = FlowField(1)

        try:
            motion = Motion(particles, flowfield)
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_motion__Motion__not_allowed_calls(self):

        particles = Particle(1)
        flowfield = FlowField(1)

        with self.assertRaises(ValueError):
            motion = Motion(particles, flowfield, time_separation=0)

        with self.assertRaises(ValueError):
            motion = Motion(particles, flowfield, time_separation=[])

        with self.assertRaises(ValueError):
            motion = Motion(particles, flowfield, time_separation=-1)

        with self.assertRaises(ValueError):
            motion = Motion(particles, flowfield, particle_loss=[])

        with self.assertRaises(ValueError):
            motion = Motion(particles, flowfield, particle_loss=(10,2))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_motion__Motion__attributes_available_after_class_init(self):

        particles = Particle(1)
        flowfield = FlowField(1)
        motion = Motion(particles, flowfield)

        # Attributes coming from user input:
        try:
            motion.time_separation
            motion.particle_loss
        except Exception:
            self.assertTrue(False)

        # Attributes computed at class init:
        try:
            motion.particle_coordinates_I1
            motion.particle_coordinates_I2
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_motion__Motion__allowed_attribute_set(self):

        particles = Particle(1)
        flowfield = FlowField(1)

        # Re-setting time separation is allowed:
        motion = Motion(particles, flowfield, time_separation = 1)
        self.assertTrue(motion.time_separation == 1)

        try:
            motion.time_separation = 0.5
        except Exception:
            self.assertTrue(False)

        self.assertTrue(motion.time_separation == 0.5)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_motion__Motion__not_allowed_attribute_set(self):

        particles = Particle(1)
        flowfield = FlowField(1)
        motion = Motion(particles, flowfield)

        with self.assertRaises(AttributeError):
            motion.particle_loss = (0, 2)

        with self.assertRaises(AttributeError):
            motion.particle_coordinates_I1 = 1

        with self.assertRaises(AttributeError):
            motion.particle_coordinates_I2 = 2

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -