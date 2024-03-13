import unittest
import numpy as np
import pandas as pd
import random
import copy
from pykitPIV import Particle, FlowField, Motion, Image

class TestImageClass(unittest.TestCase):

    def test_image__Image__allowed_calls(self):

        try:
            image = Image()
        except Exception:
            self.assertTrue(False)

        try:
            image = Image(random_seed=100)
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__not_allowed_calls(self):

        with self.assertRaises(ValueError):
            image = Image(random_seed=[])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__attributes_available_after_class_init(self):

        image = Image(random_seed=100)

        # Attributes coming from user input:
        try:
            image.random_seed

        except Exception:
            self.assertTrue(False)

        # Attributes computed at class init:
        try:
            image.images_I1
            image.images_I2
            image.exposures_per_image
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__not_allowed_attribute_set(self):

        image = Image(random_seed=100)

        with self.assertRaises(AttributeError):
            image.random_seed = 2

        with self.assertRaises(AttributeError):
            image.images_I1 = 2

        with self.assertRaises(AttributeError):
            image.images_I2 = 2

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__adding_particles(self):

        image = Image(random_seed=100)
        particles = Particle(1)

        try:
            image.add_particles(particles)
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__adding_velocity_field(self):

        image = Image(random_seed=100)
        flowfield = FlowField(1)

        try:
            image.add_flowfield(flowfield)
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__adding_motion(self):

        image = Image(random_seed=100)
        particles = Particle(1)
        flowfield = FlowField(1)
        motion = Motion(particles, flowfield)

        try:
            image.add_motion(motion)
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__plotting(self):

        pass

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__creating_particle_blur(self):

        image = Image(random_seed=100)









    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
