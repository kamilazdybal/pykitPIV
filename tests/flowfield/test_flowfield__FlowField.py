import unittest
import numpy as np
from pykitPIV import FlowField

class TestFlowFieldClass(unittest.TestCase):

    def test_flowfield__FlowField__allowed_calls(self):

        try:
            flowfield = FlowField(1)
        except Exception:
            self.assertTrue(False)

        try:
            flowfield = FlowField(10)
        except Exception:
            self.assertTrue(False)

        try:
            flowfield = FlowField(1,
                                  size=(128, 512),
                                  random_seed=100)
        except Exception:
            self.assertTrue(False)

        try:
            flowfield = FlowField(1,
                                  size=(128, 512),
                                  size_buffer=10,
                                  random_seed=100)
        except Exception:
            self.assertTrue(False)

        try:
            flowfield = FlowField(1,
                                  size=(128, 512),
                                  size_buffer=0,
                                  random_seed=100)
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_flowfield__FlowField__not_allowed_calls(self):

        # Wrong type:
        with self.assertRaises(ValueError):
            flowfield = FlowField(n_images=[])

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, size=[])

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, size_buffer=[])

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, random_seed=[])

        # Not a two-element tuple:
        with self.assertRaises(ValueError):
            flowfield = FlowField(1, size=(10,))

        # Wrong numerical value:
        with self.assertRaises(ValueError):
            flowfield = FlowField(n_images=0)

        with self.assertRaises(ValueError):
            flowfield = FlowField(n_images=-1)

        with self.assertRaises(ValueError):
            flowfield = FlowField(n_images=1.5)

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, size_buffer=-1)

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, size_buffer=1.5)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_flowfield__FlowField__attributes_available_after_class_init(self):

        flowfield = FlowField(1)

        # Attributes coming from user input:
        try:
            flowfield.n_images
            flowfield.size
            flowfield.size_buffer
            flowfield.random_seed
        except Exception:
            self.assertTrue(False)

        # Attributes initialized at class init:
        try:
            flowfield.displacement
            flowfield.gaussian_filters
            flowfield.n_gaussian_filter_iter
            flowfield.gaussian_filter_per_image
            flowfield.displacement_per_image
            flowfield.velocity_field
            flowfield.velocity_field_magnitude
        except Exception:
            self.assertTrue(False)

        self.assertEqual(flowfield.displacement, None)
        self.assertEqual(flowfield.gaussian_filters, None)
        self.assertEqual(flowfield.n_gaussian_filter_iter, None)
        self.assertEqual(flowfield.gaussian_filter_per_image, None)
        self.assertEqual(flowfield.displacement_per_image, None)
        self.assertEqual(flowfield.velocity_field, None)
        self.assertEqual(flowfield.velocity_field_magnitude, None)

        # Attributes computed at class init:
        try:
            flowfield.size_with_buffer
        except Exception:
            self.assertTrue(False)

        flowfield = FlowField(1, size=(100,200), size_buffer=10)

        self.assertEqual(flowfield.size_with_buffer, (120,220))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_flowfield__FlowField__not_allowed_attribute_set(self):

        flowfield = FlowField(1)

        with self.assertRaises(AttributeError):
            flowfield.n_images = 1

        with self.assertRaises(AttributeError):
            flowfield.size = (512, 512)

        with self.assertRaises(AttributeError):
            flowfield.size_buffer = 10

        with self.assertRaises(AttributeError):
            flowfield.displacement = (0, 10)

        with self.assertRaises(AttributeError):
            flowfield.gaussian_filters = (10, 30)

        with self.assertRaises(AttributeError):
            flowfield.n_gaussian_filter_iter = 6

        with self.assertRaises(AttributeError):
            flowfield.random_seed = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_flowfield__Flowfield__upload_velocity_field(self):

        n_images = 10
        image_size = (128, 512)

        flowfield = FlowField(n_images=n_images,
                              size=image_size,
                              size_buffer=10,
                              random_seed=100)

        try:
            velocity_field = np.ones((10, 2, 148, 532))
            flowfield.upload_velocity_field(velocity_field)

            velocity_field = np.ones((1, 2, 148, 532))
            flowfield.upload_velocity_field(velocity_field)
        except Exception:
            self.assertTrue(False)

        with self.assertRaises(ValueError):
            velocity_field = []
            flowfield.upload_velocity_field(velocity_field)

        with self.assertRaises(ValueError):
            velocity_field = np.ones((2, 148, 532))
            flowfield.upload_velocity_field(velocity_field)

        with self.assertRaises(ValueError):
            velocity_field = np.ones((11, 2, 148, 532))
            flowfield.upload_velocity_field(velocity_field)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_flowfield__Flowfield__generate_random_velocity_field(self):

        flowfield = FlowField(n_images=10,
                              size=(128, 512),
                              size_buffer=10,
                              random_seed=100)

        self.assertTrue(flowfield.velocity_field is None)
        self.assertTrue(flowfield.velocity_field_magnitude is None)

        try:

            flowfield.generate_random_velocity_field(displacement=(0, 10),
                                                     gaussian_filters=(10, 30),
                                                     n_gaussian_filter_iter=6)

            flowfield.velocity_field
            flowfield.velocity_field_magnitude

        except Exception:

            self.assertTrue(False)

        self.assertTrue(flowfield.velocity_field is not None)
        self.assertTrue(flowfield.velocity_field_magnitude is not None)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_flowfield__Flowfield__generate_sinusoidal_velocity_field(self):

        flowfield = FlowField(n_images=10,
                              size=(128, 512),
                              size_buffer=10,
                              random_seed=100)

        self.assertTrue(flowfield.velocity_field is None)
        self.assertTrue(flowfield.velocity_field_magnitude is None)

        try:

            flowfield.generate_sinusoidal_velocity_field(amplitudes=(2, 4),
                                                         wavelengths=(20,40),
                                                         components='u')

            flowfield.velocity_field
            flowfield.velocity_field_magnitude

        except Exception:

            self.assertTrue(False)

        self.assertTrue(flowfield.velocity_field is not None)
        self.assertTrue(flowfield.velocity_field_magnitude is not None)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_flowfield__Flowfield__generate_checkered_velocity_field(self):

        flowfield = FlowField(n_images=10,
                              size=(128, 512),
                              size_buffer=10,
                              random_seed=100)

        self.assertTrue(flowfield.velocity_field is None)
        self.assertTrue(flowfield.velocity_field_magnitude is None)

        try:

            flowfield.generate_checkered_velocity_field(displacement=(0, 10),
                                                        m=10,
                                                        n=10,
                                                        rotation=10)

            flowfield.velocity_field
            flowfield.velocity_field_magnitude

        except Exception:

            self.assertTrue(False)

        self.assertTrue(flowfield.velocity_field is not None)
        self.assertTrue(flowfield.velocity_field_magnitude is not None)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_flowfield__Flowfield__generate_chebyshev_velocity_field(self):

        flowfield = FlowField(n_images=10,
                              size=(128, 512),
                              size_buffer=10,
                              random_seed=100)

        self.assertTrue(flowfield.velocity_field is None)
        self.assertTrue(flowfield.velocity_field_magnitude is None)

        try:

            flowfield.generate_chebyshev_velocity_field(displacement=(0, 10),
                                                        start=0.3,
                                                        stop=0.8,
                                                        order=10)

            flowfield.velocity_field
            flowfield.velocity_field_magnitude

        except Exception:

            self.assertTrue(False)

        self.assertTrue(flowfield.velocity_field is not None)
        self.assertTrue(flowfield.velocity_field_magnitude is not None)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_flowfield__Flowfield__generate_spherical_harmonics_velocity_field(self):

        flowfield = FlowField(n_images=10,
                              size=(128, 512),
                              size_buffer=10,
                              random_seed=100)

        self.assertTrue(flowfield.velocity_field is None)
        self.assertTrue(flowfield.velocity_field_magnitude is None)

        try:

            flowfield.generate_spherical_harmonics_velocity_field(displacement=(0, 10),
                                                                  start=0.3,
                                                                  stop=0.8,
                                                                  order=1,
                                                                  degree=1)

            flowfield.velocity_field
            flowfield.velocity_field_magnitude

        except Exception:

            self.assertTrue(False)

        self.assertTrue(flowfield.velocity_field is not None)
        self.assertTrue(flowfield.velocity_field_magnitude is not None)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -