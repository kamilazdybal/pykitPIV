import unittest
import numpy as np
from pykitPIV import Particle, FlowField, Motion, Image
from pykitPIV import ParticleSpecs, FlowFieldSpecs, MotionSpecs, ImageSpecs
from pykitPIV.ml import PIVEnv, Rewards, Cues

class TestRewardsClass(unittest.TestCase):

    def test_ml__Rewards__allowed_calls(self):

        try:
            rewards = Rewards()
        except Exception:
            self.assertTrue(False)

        try:
            rewards = Rewards(verbose=True)
        except Exception:
            self.assertTrue(False)

        try:
            rewards = Rewards(random_seed=100)
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_ml__Rewards__not_allowed_calls(self):

        # Wrong type:
        with self.assertRaises(ValueError):
            rewards = Rewards(verbose=[])

        with self.assertRaises(ValueError):
            rewards = Rewards(random_seed=[])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_ml__Rewards__attributes_available_after_class_init(self):

        rewards = Rewards()

        # Attributes coming from user input:
        try:
            rewards.verbose
            rewards.random_seed
        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_ml__Rewards__not_allowed_attribute_set(self):

        rewards = Rewards()

        with self.assertRaises(AttributeError):
            rewards.verbose = True

        with self.assertRaises(AttributeError):
            rewards.random_seed = 100

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_ml__Rewards__transformation_not_callable(self):

        rewards = Rewards()

        flowfield = FlowField(n_images=10,
                              size=(50,100),
                              size_buffer=2,
                              time_separation=1,
                              random_seed=100)

        # Generate random velocity field:
        flowfield.generate_constant_velocity_field(u_magnitude=(1, 4),
                                                   v_magnitude=(1, 4))

        # Access the velocity components tensor:
        vector_field = flowfield.velocity_field

        transformation = []

        with self.assertRaises(ValueError):
            reward = rewards.divergence(vector_field=vector_field,
                                        transformation=transformation)

        with self.assertRaises(ValueError):
            reward = rewards.vorticity(vector_field=vector_field,
                                       transformation=transformation)

        with self.assertRaises(ValueError):
            reward = rewards.q_criterion(vector_field=vector_field,
                                         transformation=transformation)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_ml__Rewards__divergence(self):

        rewards = Rewards(verbose=False,
                          random_seed=100)

        flowfield = FlowField(n_images=1,
                              size=(50,100),
                              size_buffer=2,
                              time_separation=1,
                              random_seed=100)

        # Generate random velocity field:
        flowfield.generate_constant_velocity_field(u_magnitude=(1, 4),
                                                   v_magnitude=(1, 4))

        # Access the velocity components tensor:
        vector_field = flowfield.velocity_field

        def transformation(div):
            return np.max(np.abs(div))

        try:
            reward = rewards.divergence(vector_field=vector_field,
                                        transformation=transformation)
        except Exception:
            self.assertTrue(False)

        # Check that this reward is zero since the flow field is constant:
        self.assertTrue(reward == 0.0)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_ml__Rewards__vorticity(self):

        rewards = Rewards(verbose=False,
                          random_seed=100)

        flowfield = FlowField(n_images=1,
                              size=(50,100),
                              size_buffer=2,
                              time_separation=1,
                              random_seed=100)

        # Generate random velocity field:
        flowfield.generate_constant_velocity_field(u_magnitude=(1, 4),
                                                   v_magnitude=(1, 4))

        # Access the velocity components tensor:
        vector_field = flowfield.velocity_field

        def transformation(div):
            return np.max(np.abs(div))

        try:
            reward = rewards.vorticity(vector_field=vector_field,
                                       transformation=transformation)
        except Exception:
            self.assertTrue(False)

        # Check that this reward is zero since the flow field is constant:
        self.assertTrue(reward == 0.0)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_ml__Rewards__q_criterion(self):

        rewards = Rewards(verbose=False,
                          random_seed=100)

        flowfield = FlowField(n_images=1,
                              size=(50,100),
                              size_buffer=2,
                              time_separation=1,
                              random_seed=100)

        # Generate random velocity field:
        flowfield.generate_constant_velocity_field(u_magnitude=(1, 4),
                                                   v_magnitude=(1, 4))

        # Access the velocity components tensor:
        vector_field = flowfield.velocity_field

        def transformation(div):
            return np.max(np.abs(div))

        try:
            reward = rewards.q_criterion(vector_field=vector_field,
                                         transformation=transformation)
        except Exception:
            self.assertTrue(False)

        # Check that this reward is zero since the flow field is constant:
        self.assertTrue(reward == 0.0)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
