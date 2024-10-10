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

    def test_image__Image__adding_flowfield(self):

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

        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(0, 10))

        motion = Motion(particles, flowfield)

        try:
            image.add_motion(motion)
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__removing_buffers_no_light_no_motion(self):

        # Remove buffers from the I1 frames before reflected light is added to the image and before motion is applied:

        size = (100,80)
        size_buffer = 5

        image = Image(random_seed=100)
        particles = Particle(1,
                             size=size,
                             size_buffer=size_buffer)

        image.add_particles(particles)

        self.assertTrue(image.images_I1.shape[2] == 110)
        self.assertTrue(image.images_I1.shape[3] == 90)

        self.assertTrue(image.images_I2 is None)

        self.assertTrue(image.images_I1_no_buffer is None)
        self.assertTrue(image.images_I2_no_buffer is None)

        image.remove_buffers()

        self.assertTrue(image.images_I1_no_buffer is not None)
        self.assertTrue(image.images_I2_no_buffer is None)

        self.assertTrue(image.images_I1_no_buffer.shape[2] == 100)
        self.assertTrue(image.images_I1_no_buffer.shape[3] == 80)

        self.assertTrue(image.images_I1.shape[2] == 110)
        self.assertTrue(image.images_I1.shape[3] == 90)

        self.assertTrue(image.images_I2 is None)

        # Check that the image without buffer is equal to the interior part of the image with buffer:

        self.assertTrue(np.array_equal(image.images_I1[0,0,size_buffer:size[0]+size_buffer,size_buffer:size[1]+size_buffer], image.images_I1_no_buffer[0,0,:,:]))

    def test_image__Image__removing_buffers_with_light_no_motion(self):

        # Remove buffers from the I1 frames after reflected light is added to the image and before motion is applied:

        size = (100, 80)
        size_buffer = 5

        image = Image(random_seed=100)
        particles = Particle(1,
                             size=size,
                             size_buffer=size_buffer)

        image.add_particles(particles)
        image.add_reflected_light(exposures=(0.6,0.65),
                                  maximum_intensity=2**16-1,
                                  laser_beam_thickness=1,
                                  laser_over_exposure=1,
                                  laser_beam_shape=0.95,
                                  alpha=1/20)

        self.assertTrue(image.images_I1.shape[2] == 110)
        self.assertTrue(image.images_I1.shape[3] == 90)

        self.assertTrue(image.images_I2 is None)

        self.assertTrue(image.images_I1_no_buffer is None)
        self.assertTrue(image.images_I2_no_buffer is None)

        image.remove_buffers()

        self.assertTrue(image.images_I1_no_buffer is not None)
        self.assertTrue(image.images_I2_no_buffer is None)

        self.assertTrue(image.images_I1_no_buffer.shape[2] == 100)
        self.assertTrue(image.images_I1_no_buffer.shape[3] == 80)

        self.assertTrue(image.images_I1.shape[2] == 110)
        self.assertTrue(image.images_I1.shape[3] == 90)

        self.assertTrue(image.images_I2 is None)

        # Check that the image without buffer is equal to the interior part of the image with buffer:

        self.assertTrue(np.array_equal(image.images_I1[0,0,size_buffer:size[0]+size_buffer,size_buffer:size[1]+size_buffer], image.images_I1_no_buffer[0,0,:,:]))

    def test_image__Image__removing_buffers_with_light_with_motion(self):

        # Remove buffers from the I1 frames after reflected light is added to the image and after motion is applied:

        size = (100, 80)
        size_buffer = 5

        image = Image(random_seed=100)

        particles = Particle(1,
                             size=size,
                             size_buffer=size_buffer)

        image.add_particles(particles)

        image.add_reflected_light(exposures=(0.6,0.65),
                                  maximum_intensity=2**16-1,
                                  laser_beam_thickness=1,
                                  laser_over_exposure=1,
                                  laser_beam_shape=0.95,
                                  alpha=1/20)

        flowfield = FlowField(1,
                              size=size,
                              size_buffer=size_buffer,
                              random_seed=100)

        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(0, 10))

        image.add_flowfield(flowfield)

        motion = Motion(particles,
                        flowfield,
                        time_separation=1)

        motion.forward_euler(n_steps=10)

        image.add_motion(motion)

        image.add_reflected_light(exposures=(0.6,0.65),
                                  maximum_intensity=2**16-1,
                                  laser_beam_thickness=1,
                                  laser_over_exposure=1,
                                  laser_beam_shape=0.95,
                                  alpha=1/20)

        self.assertTrue(image.images_I1.shape[2] == 110)
        self.assertTrue(image.images_I1.shape[3] == 90)
        self.assertTrue(image.images_I2.shape[2] == 110)
        self.assertTrue(image.images_I2.shape[3] == 90)

        self.assertTrue(image.images_I1_no_buffer is None)
        self.assertTrue(image.images_I2_no_buffer is None)

        image.remove_buffers()

        self.assertTrue(image.images_I1_no_buffer is not None)
        self.assertTrue(image.images_I2_no_buffer is not None)

        self.assertTrue(image.images_I1_no_buffer.shape[2] == 100)
        self.assertTrue(image.images_I1_no_buffer.shape[3] == 80)
        self.assertTrue(image.images_I2_no_buffer.shape[2] == 100)
        self.assertTrue(image.images_I2_no_buffer.shape[3] == 80)

        self.assertTrue(image.images_I1.shape[2] == 110)
        self.assertTrue(image.images_I1.shape[3] == 90)
        self.assertTrue(image.images_I2.shape[2] == 110)
        self.assertTrue(image.images_I2.shape[3] == 90)

        # Check that the image without buffer is equal to the interior part of the image with buffer:

        self.assertTrue(np.array_equal(image.images_I1[0,0,size_buffer:size[0]+size_buffer,size_buffer:size[1]+size_buffer], image.images_I1_no_buffer[0,0,:,:]))
        self.assertTrue(np.array_equal(image.images_I2[0,0,size_buffer:size[0]+size_buffer,size_buffer:size[1]+size_buffer], image.images_I2_no_buffer[0,0,:,:]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__plotting(self):

        pass

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__creating_particle_blur(self):

        image = Image(random_seed=100)









    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
