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

class TestPostprocessClass(unittest.TestCase):

    def test_postprocess__Postprocess__allowed_calls(self):

        try:
            pp = Postprocess(image_tensor=images_tensor,
                             random_seed=100)
        except Exception:
            self.assertTrue(False)

        try:
            pp_paired = Postprocess(image_tensor=images_tensor_paired,
                                    random_seed=100)
        except Exception:
            self.assertTrue(False)

        self.assertTrue(not pp.image_pair)
        self.assertTrue(pp_paired.image_pair)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_postprocess__Postprocess__not_allowed_calls(self):

        # Wrong type:
        with self.assertRaises(ValueError):
            pp = Postprocess(image_tensor=[],
                             random_seed=100)

        with self.assertRaises(ValueError):
            pp = Postprocess(image_tensor=images_tensor,
                             random_seed=[])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_postprocess__Postprocess__attributes_available_after_class_init(self):

        pp = Postprocess(image_tensor=images_tensor,
                         random_seed=100)

        # Attributes coming from user input:
        try:
            pp.image_tensor
            pp.random_seed
        except Exception:
            self.assertTrue(False)

        # Attributes initialized at class init:
        try:
            pp.image_pair
            pp.processed_image_tensor
            pp.scale_per_image
        except Exception:
            self.assertTrue(False)

        self.assertTrue(np.array_equal(pp.processed_image_tensor, pp.image_tensor))
        self.assertEqual(pp.scale_per_image, None)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_postprocess__Postprocess__not_allowed_attribute_set(self):

        pp = Postprocess(image_tensor=images_tensor,
                         random_seed=100)

        with self.assertRaises(AttributeError):
            pp.image_tensor = images_tensor

        with self.assertRaises(AttributeError):
            pp.random_seed = 100

        with self.assertRaises(AttributeError):
            pp.image_pair = False

        with self.assertRaises(AttributeError):
            pp.processed_image_tensor = images_tensor

        with self.assertRaises(AttributeError):
            pp.scale_per_image = 1

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_postprocess__Postprocess__add_gaussian_noise(self):

        pp = Postprocess(image_tensor=images_tensor,
                         random_seed=100)

        self.assertTrue(np.array_equal(pp.processed_image_tensor, pp.image_tensor))

        try:
            pp.add_gaussian_noise(loc=0.0,
                                  scale=(500,1000),
                                  clip=2**16-1)
        except Exception:
            self.assertTrue(False)

        self.assertTrue(not np.array_equal(pp.processed_image_tensor, pp.image_tensor))

        pp = Postprocess(image_tensor=images_tensor,
                         random_seed=100)

        try:
            pp.add_gaussian_noise(loc=0.0,
                                  scale=1000,
                                  clip=2**16-1)
        except Exception:
            self.assertTrue(False)

        self.assertTrue(not np.array_equal(pp.processed_image_tensor, pp.image_tensor))

        pp = Postprocess(image_tensor=images_tensor,
                         random_seed=100)

        try:
            pp.add_gaussian_noise(loc=0.0,
                                  scale=1000,
                                  clip=None)
        except Exception:
            self.assertTrue(False)

        self.assertTrue(not np.array_equal(pp.processed_image_tensor, pp.image_tensor))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_postprocess__Postprocess__add_shot_noise(self):

        pp = Postprocess(image_tensor=images_tensor,
                         random_seed=100)

        self.assertTrue(np.array_equal(pp.processed_image_tensor, pp.image_tensor))

        try:
            pp.add_shot_noise(strength=1,
                              clip=2**16-1)
        except Exception:
            self.assertTrue(False)

        self.assertTrue(not np.array_equal(pp.processed_image_tensor, pp.image_tensor))

        pp = Postprocess(image_tensor=images_tensor,
                         random_seed=100)

        try:
            pp.add_shot_noise(strength=1,
                              clip=None)
        except Exception:
            self.assertTrue(False)

        self.assertTrue(not np.array_equal(pp.processed_image_tensor, pp.image_tensor))

        pp = Postprocess(image_tensor=images_tensor,
                         random_seed=100)

        try:
            pp.add_shot_noise(strength=0.1,
                              clip=None)
        except Exception:
            self.assertTrue(False)

        self.assertTrue(not np.array_equal(pp.processed_image_tensor, pp.image_tensor))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_postprocess__Postprocess__log_transform_images(self):

        pp = Postprocess(image_tensor=images_tensor,
                         random_seed=100)

        self.assertTrue(np.array_equal(pp.processed_image_tensor, pp.image_tensor))

        try:
            pp.log_transform_images(addition=10000)
        except Exception:
            self.assertTrue(False)

        self.assertTrue(not np.array_equal(pp.processed_image_tensor, pp.image_tensor))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_postprocess__Postprocess__chain_transformations(self):

        pp = Postprocess(image_tensor=images_tensor,
                         random_seed=100)

        self.assertTrue(np.array_equal(pp.processed_image_tensor, pp.image_tensor))

        try:
            pp.add_shot_noise(strength=1,
                              clip=2**16-1)

            first_transform = pp.processed_image_tensor

            self.assertTrue(not np.array_equal(pp.processed_image_tensor, pp.image_tensor))

            pp.add_gaussian_noise(loc=0.0,
                                  scale=1000,
                                  clip=2**16-1)

            second_transform = pp.processed_image_tensor

            self.assertTrue(not np.array_equal(pp.processed_image_tensor, pp.image_tensor))
            self.assertTrue(not np.array_equal(pp.processed_image_tensor, first_transform))

            pp.log_transform_images(addition=10000)

            self.assertTrue(not np.array_equal(pp.processed_image_tensor, pp.image_tensor))
            self.assertTrue(not np.array_equal(pp.processed_image_tensor, first_transform))
            self.assertTrue(not np.array_equal(pp.processed_image_tensor, second_transform))

        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
