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
