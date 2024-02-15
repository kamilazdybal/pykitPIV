import unittest
import numpy as np
from pypiv import FlowField

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
                                  flow_mode='random',
                                  gaussian_filters=(8, 10),
                                  n_gaussian_filter_iter=10,
                                  sin_period=(30, 300),
                                  displacement=(0, 10),
                                  lost_particles_percentage=10,
                                  random_seed=100)
        except Exception:
            self.assertTrue(False)

    def test_flowfield__FlowField__not_allowed_calls(self):

        # Wrong type:
        with self.assertRaises(ValueError):
            flowfield = FlowField(1, size=[])

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, flow_mode=[])

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, gaussian_filters=[])

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, n_gaussian_filter_iter=[])

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, sin_period=[])

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, displacement=[])

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, lost_particles_percentage=[])

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, random_seed=[])

        # Not a two-element tuple:
        with self.assertRaises(ValueError):
            flowfield = FlowField(1, size=(10,))

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, displacement=(10,))

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, gaussian_filters=(10,))

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, sin_period=(10,))

        # Not a min-max tuple:
        with self.assertRaises(ValueError):
            flowfield = FlowField(1, displacement=(10,1))

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, gaussian_filters=(10,1))

        with self.assertRaises(ValueError):
            flowfield = FlowField(1, sin_period=(10,1))

        # Not allowed string:
        with self.assertRaises(ValueError):
            flowfield = FlowField(1, flow_mode='Test')

    def test_flowfield__FlowField__attributes_available_after_class_init(self):

        flowfield = FlowField(1)

        # Attributes coming from user input:
        try:
            flowfield.n_images
            flowfield.size
            flowfield.flow_mode
            flowfield.displacement
            flowfield.gaussian_filters
            flowfield.n_gaussian_filter_iter
            flowfield.lost_particles_percentage
            flowfield.sin_period
            flowfield.random_seed
        except Exception:
            self.assertTrue(False)

        # Attributes computed at class init:
        try:
            flowfield.gaussian_filter_per_image
            flowfield.displacement_per_image
            flowfield.velocity_field
            flowfield.velocity_field_magnitude
        except Exception:
            self.assertTrue(False)