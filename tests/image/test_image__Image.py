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

        flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(0, 10))

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

        images_I1_no_buffer = image.remove_buffers(image.images_I1)

        self.assertTrue(images_I1_no_buffer.shape[2] == 100)
        self.assertTrue(images_I1_no_buffer.shape[3] == 80)

        self.assertTrue(image.images_I1.shape[2] == 110)
        self.assertTrue(image.images_I1.shape[3] == 90)

        # Check that the image without buffer is equal to the interior part of the image with buffer:

        self.assertTrue(np.array_equal(image.images_I1[0,0,size_buffer:size[0]+size_buffer,size_buffer:size[1]+size_buffer], images_I1_no_buffer[0,0,:,:]))

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

        images_I1_no_buffer = image.remove_buffers(image.images_I1)

        self.assertTrue(images_I1_no_buffer.shape[2] == 100)
        self.assertTrue(images_I1_no_buffer.shape[3] == 80)

        self.assertTrue(image.images_I1.shape[2] == 110)
        self.assertTrue(image.images_I1.shape[3] == 90)

        # Check that the image without buffer is equal to the interior part of the image with buffer:

        self.assertTrue(np.array_equal(image.images_I1[0,0,size_buffer:size[0]+size_buffer,size_buffer:size[1]+size_buffer], images_I1_no_buffer[0,0,:,:]))

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
                        flowfield)

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

        images_I1_no_buffer = image.remove_buffers(image.images_I1)
        images_I2_no_buffer = image.remove_buffers(image.images_I2)

        self.assertTrue(images_I1_no_buffer.shape[2] == 100)
        self.assertTrue(images_I1_no_buffer.shape[3] == 80)
        self.assertTrue(images_I2_no_buffer.shape[2] == 100)
        self.assertTrue(images_I2_no_buffer.shape[3] == 80)

        self.assertTrue(image.images_I1.shape[2] == 110)
        self.assertTrue(image.images_I1.shape[3] == 90)
        self.assertTrue(image.images_I2.shape[2] == 110)
        self.assertTrue(image.images_I2.shape[3] == 90)

        # Check that the image without buffer is equal to the interior part of the image with buffer:

        self.assertTrue(np.array_equal(image.images_I1[0,0,size_buffer:size[0]+size_buffer,size_buffer:size[1]+size_buffer], images_I1_no_buffer[0,0,:,:]))
        self.assertTrue(np.array_equal(image.images_I2[0,0,size_buffer:size[0]+size_buffer,size_buffer:size[1]+size_buffer], images_I2_no_buffer[0,0,:,:]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__get_velocity_field(self):

        n_images = 1
        size_buffer = 10
        image_size = (20, 40)
        time_separation = 1
        random_seed = 100

        particles = Particle(n_images,
                             size=image_size,
                             size_buffer=size_buffer,
                             diameters=(2, 4),
                             distances=(1, 2),
                             densities=(0.05, 0.06),
                             diameter_std=(0.5,0.5),
                             seeding_mode='random',
                             random_seed=random_seed)

        flowfield = FlowField(n_images,
                              size=image_size,
                              size_buffer=size_buffer,
                              time_separation=time_separation,
                              random_seed=random_seed)

        flowfield.generate_random_velocity_field(gaussian_filters=(2, 10),
                                                 n_gaussian_filter_iter=10,
                                                 displacement=(2, 10))

        motion = Motion(particles,
                        flowfield)

        motion.runge_kutta_4th(n_steps=10)

        image = Image(random_seed=random_seed)

        image.add_particles(particles)
        image.add_flowfield(flowfield)

        try:
            V = image.get_velocity_field()
        except Exception:
            self.assertTrue(False)

        (N, C, H, W) = V.shape
        self.assertTrue(N == 1)
        self.assertTrue(C == 2)
        self.assertTrue(H == 40)
        self.assertTrue(W == 60)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__get_velocity_field_magnitude(self):

        n_images = 1
        size_buffer = 10
        image_size = (20, 40)
        time_separation = 1
        random_seed = 100

        particles = Particle(n_images,
                             size=image_size,
                             size_buffer=size_buffer,
                             diameters=(2, 4),
                             distances=(1, 2),
                             densities=(0.05, 0.06),
                             diameter_std=(0.5,0.5),
                             seeding_mode='random',
                             random_seed=random_seed)

        flowfield = FlowField(n_images,
                              size=image_size,
                              size_buffer=size_buffer,
                              time_separation=time_separation,
                              random_seed=random_seed)

        flowfield.generate_random_velocity_field(gaussian_filters=(2, 10),
                                                 n_gaussian_filter_iter=10,
                                                 displacement=(2, 10))

        motion = Motion(particles,
                        flowfield)

        motion.runge_kutta_4th(n_steps=10)

        image = Image(random_seed=random_seed)

        image.add_particles(particles)
        image.add_flowfield(flowfield)

        try:
            V = image.get_velocity_field_magnitude()
        except Exception:
            self.assertTrue(False)

        (N, C, H, W) = V.shape
        self.assertTrue(N == 1)
        self.assertTrue(C == 1)
        self.assertTrue(H == 40)
        self.assertTrue(W == 60)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__get_displacement_field(self):

        n_images = 1
        size_buffer = 10
        image_size = (20, 40)
        time_separation = 1
        random_seed = 100

        particles = Particle(n_images,
                             size=image_size,
                             size_buffer=size_buffer,
                             diameters=(2, 4),
                             distances=(1, 2),
                             densities=(0.05, 0.06),
                             diameter_std=(0.5,0.5),
                             seeding_mode='random',
                             random_seed=random_seed)

        flowfield = FlowField(n_images,
                              size=image_size,
                              size_buffer=size_buffer,
                              time_separation=time_separation,
                              random_seed=random_seed)

        flowfield.generate_random_velocity_field(gaussian_filters=(2, 10),
                                                 n_gaussian_filter_iter=10,
                                                 displacement=(2, 10))

        motion = Motion(particles,
                        flowfield)

        motion.runge_kutta_4th(n_steps=10)

        image = Image(random_seed=random_seed)

        image.add_particles(particles)

        with self.assertRaises(NameError):
            ds = image.get_displacement_field()

        image.add_flowfield(flowfield)
        image.add_motion(motion)

        try:
            ds = image.get_displacement_field()
        except Exception:
            self.assertTrue(False)

        (N, C, H, W) = ds.shape
        self.assertTrue(N == 1)
        self.assertTrue(C == 2)
        self.assertTrue(H == 40)
        self.assertTrue(W == 60)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__get_displacement_field_magnitude(self):

        n_images = 1
        size_buffer = 10
        image_size = (20, 40)
        time_separation = 1
        random_seed = 100

        particles = Particle(n_images,
                             size=image_size,
                             size_buffer=size_buffer,
                             diameters=(2, 4),
                             distances=(1, 2),
                             densities=(0.05, 0.06),
                             diameter_std=(0.5,0.5),
                             seeding_mode='random',
                             random_seed=random_seed)

        flowfield = FlowField(n_images,
                              size=image_size,
                              size_buffer=size_buffer,
                              random_seed=random_seed)

        flowfield.generate_random_velocity_field(gaussian_filters=(2, 10),
                                                 n_gaussian_filter_iter=10,
                                                 displacement=(2, 10))

        motion = Motion(particles,
                        flowfield)

        motion.runge_kutta_4th(n_steps=10)

        image = Image(random_seed=random_seed)

        image.add_particles(particles)

        with self.assertRaises(NameError):
            ds = image.get_displacement_field_magnitude()

        image.add_flowfield(flowfield)
        image.add_motion(motion)

        try:
            ds = image.get_displacement_field_magnitude()
        except Exception:
            self.assertTrue(False)

        (N, C, H, W) = ds.shape
        self.assertTrue(N == 1)
        self.assertTrue(C == 1)
        self.assertTrue(H == 40)
        self.assertTrue(W == 60)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__clip_or_normalize_maximum_intensity_I1(self):

        maximum_intensity = 2 ** 16 - 1

        particles = Particle(1,
                             size=(100, 100),
                             size_buffer=5,
                             densities=(0.1,0.1))

        image = Image(random_seed=100)

        image.add_particles(particles)

        image.add_reflected_light(exposures=(0.6,0.65),
                                  maximum_intensity=maximum_intensity,
                                  laser_beam_thickness=1,
                                  laser_over_exposure=1,
                                  laser_beam_shape=0.95,
                                  alpha=1/8,
                                  clip_intensities=False,
                                  normalize_intensities=False)

        self.assertTrue(np.max(image.images_I1) > maximum_intensity)

        image.add_reflected_light(exposures=(0.6,0.65),
                                  maximum_intensity=maximum_intensity,
                                  laser_beam_thickness=1,
                                  laser_over_exposure=1,
                                  laser_beam_shape=0.95,
                                  alpha=1/8,
                                  clip_intensities=True,
                                  normalize_intensities=False)

        self.assertTrue(np.max(image.images_I1) <= maximum_intensity)

        image.add_reflected_light(exposures=(0.6,0.65),
                                  maximum_intensity=maximum_intensity,
                                  laser_beam_thickness=1,
                                  laser_over_exposure=1,
                                  laser_beam_shape=0.95,
                                  alpha=1/8,
                                  clip_intensities=False,
                                  normalize_intensities=True)

        self.assertTrue(np.max(image.images_I1) <= maximum_intensity)

        # Only one of clip_intensities or normalize_intensities can be True at a time:
        with self.assertRaises(ValueError):
            image.add_reflected_light(exposures=(0.6,0.65),
                                      maximum_intensity=maximum_intensity,
                                      laser_beam_thickness=1,
                                      laser_over_exposure=1,
                                      laser_beam_shape=0.95,
                                      alpha=1/8,
                                      clip_intensities=True,
                                      normalize_intensities=True)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__clip_or_normalize_maximum_intensity_I2(self):

        maximum_intensity = 2 ** 16 - 1

        particles = Particle(1,
                             size=(100, 100),
                             size_buffer=5,
                             densities=(0.1,0.1))

        flowfield = FlowField(1,
                              size=(100, 100),
                              size_buffer=5,
                              random_seed=100)

        flowfield.generate_random_velocity_field(gaussian_filters=(2, 10),
                                                 n_gaussian_filter_iter=10,
                                                 displacement=(2, 10))

        motion = Motion(particles,
                        flowfield)

        motion.runge_kutta_4th(n_steps=10)

        image = Image(random_seed=100)

        image.add_particles(particles)
        image.add_flowfield(flowfield)
        image.add_motion(motion)

        image.add_reflected_light(exposures=(0.6,0.65),
                                  maximum_intensity=maximum_intensity,
                                  laser_beam_thickness=1,
                                  laser_over_exposure=1,
                                  laser_beam_shape=0.95,
                                  alpha=1/8,
                                  clip_intensities=False,
                                  normalize_intensities=False)

        self.assertTrue(np.max(image.images_I2) > maximum_intensity)

        image.add_reflected_light(exposures=(0.6,0.65),
                                  maximum_intensity=maximum_intensity,
                                  laser_beam_thickness=1,
                                  laser_over_exposure=1,
                                  laser_beam_shape=0.95,
                                  alpha=1/8,
                                  clip_intensities=True,
                                  normalize_intensities=False)

        self.assertTrue(np.max(image.images_I2) <= maximum_intensity)

        image.add_reflected_light(exposures=(0.6,0.65),
                                  maximum_intensity=maximum_intensity,
                                  laser_beam_thickness=1,
                                  laser_over_exposure=1,
                                  laser_beam_shape=0.95,
                                  alpha=1/8,
                                  clip_intensities=False,
                                  normalize_intensities=True)

        self.assertTrue(np.max(image.images_I2) <= maximum_intensity)

        # Only one of clip_intensities or normalize_intensities can be True at a time:
        with self.assertRaises(ValueError):
            image.add_reflected_light(exposures=(0.6,0.65),
                                      maximum_intensity=maximum_intensity,
                                      laser_beam_thickness=1,
                                      laser_over_exposure=1,
                                      laser_beam_shape=0.95,
                                      alpha=1/8,
                                      clip_intensities=True,
                                      normalize_intensities=True)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__concatenate_tensors(self):

        pass




    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__plotting(self):

        pass

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_image__Image__creating_particle_blur(self):

        image = Image(random_seed=100)









    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
