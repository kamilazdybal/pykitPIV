import unittest
import numpy as np
from pykitPIV import Particle, FlowField, Motion, Image
from pykitPIV import ParticleSpecs, FlowFieldSpecs, MotionSpecs, ImageSpecs
from pykitPIV.ml import PIVEnv

class TestPIVEnvClass(unittest.TestCase):

    def test_ml__PIVEnv__initialize_from_spec_dictionaries(self):

        try:

            # Prepare specs for pykitPIV parameters:
            particle_spec = {'diameters': (1, 1),
                             'distances': (2, 2),
                             'densities': (0.2, 0.2),
                             'diameter_std': 1,
                             'seeding_mode': 'random'}

            flowfield_spec = {'size': (200, 300),
                              'time_separation': 2,
                              'flowfield_type': 'random smooth',
                              'gaussian_filters': (30, 30),
                              'n_gaussian_filter_iter': 5,
                              'displacement': (2, 2)}

            motion_spec = {'n_steps': 10,
                           'particle_loss': (0, 2),
                           'particle_gain': (0, 2)}

            image_spec = {'exposures': (0.5, 0.9),
                          'maximum_intensity': 2 ** 16 - 1,
                          'laser_beam_thickness': 1,
                          'laser_over_exposure': 1,
                          'laser_beam_shape': 0.95,
                          'alpha': 1 / 8,
                          'clip_intensities': True,
                          'normalize_intensities': False}

            # Define a custom cues function that returns an array of three cues:
            def cues_function(displacement_field_tensor):

                mean_displacement = np.mean(displacement_field_tensor)
                max_displacement = np.max(displacement_field_tensor)
                min_displacement = np.min(displacement_field_tensor)

                cues = np.array([[mean_displacement, max_displacement, min_displacement]])

                return cues

            # Initialize the Gymnasium environment:
            env = PIVEnv(interrogation_window_size=(100, 200),
                         interrogation_window_size_buffer=10,
                         cues_function=cues_function,
                         particle_spec=particle_spec,
                         motion_spec=motion_spec,
                         image_spec=image_spec,
                         flowfield_spec=flowfield_spec,
                         user_flowfield=None,
                         inference_model=None,
                         random_seed=100)

        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_ml__PIVEnv__initialize_from_spec_constructors(self):

        try:

            # Prepare specs for pykitPIV parameters:
            particle_spec = ParticleSpecs()

            flowfield_spec = FlowFieldSpecs()

            motion_spec = MotionSpecs()

            image_spec = ImageSpecs()

            # Define a custom cues function that returns an array of three cues:
            def cues_function(displacement_field_tensor):

                mean_displacement = np.mean(displacement_field_tensor)
                max_displacement = np.max(displacement_field_tensor)
                min_displacement = np.min(displacement_field_tensor)

                cues = np.array([[mean_displacement, max_displacement, min_displacement]])

                return cues

            # Initialize the Gymnasium environment:
            env = PIVEnv(interrogation_window_size=(100, 200),
                         interrogation_window_size_buffer=10,
                         cues_function=cues_function,
                         particle_spec=particle_spec,
                         motion_spec=motion_spec,
                         image_spec=image_spec,
                         flowfield_spec=flowfield_spec,
                         user_flowfield=None,
                         inference_model=None,
                         random_seed=100)

        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_ml__PIVEnv__custom_user_flowfield(self):

        try:

            # Prepare specs for pykitPIV parameters:
            particle_spec = ParticleSpecs()

            motion_spec = MotionSpecs()

            image_spec = ImageSpecs()

            # Define a custom cues function that returns an array of three cues:
            def cues_function(displacement_field_tensor):

                mean_displacement = np.mean(displacement_field_tensor)
                max_displacement = np.max(displacement_field_tensor)
                min_displacement = np.min(displacement_field_tensor)

                cues = np.array([[mean_displacement, max_displacement, min_displacement]])

                return cues

            # Initialize an object of the FlowField class that will store the user-specified flow field:
            user_flowfield = FlowField(1,
                                       size=(500, 1000),
                                       size_buffer=0,
                                       time_separation=1)

            # Create a dummy velocity field:
            user_flowfield.generate_random_velocity_field(displacement=(10, 10),
                                                          gaussian_filters=(30, 30),
                                                          n_gaussian_filter_iter=6)

            velocity_field = user_flowfield.velocity_field

            # Upload a user-specified velocity field tensor:
            user_flowfield.upload_velocity_field(velocity_field)

            # Initialize the Gymnasium environment:
            env = PIVEnv(interrogation_window_size=(100, 200),
                         interrogation_window_size_buffer=10,
                         cues_function=cues_function,
                         particle_spec=particle_spec,
                         motion_spec=motion_spec,
                         image_spec=image_spec,
                         user_flowfield=user_flowfield,
                         inference_model=None,
                         random_seed=100)

        except Exception:
            self.assertTrue(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_ml__PIVEnv__user_flowfield_not_FlowField_object(self):

        # Prepare specs for pykitPIV parameters:
        particle_spec = ParticleSpecs()

        motion_spec = MotionSpecs()

        image_spec = ImageSpecs()

        # Define a custom cues function that returns an array of three cues:
        def cues_function(displacement_field_tensor):
            mean_displacement = np.mean(displacement_field_tensor)
            max_displacement = np.max(displacement_field_tensor)
            min_displacement = np.min(displacement_field_tensor)

            cues = np.array([[mean_displacement, max_displacement, min_displacement]])

            return cues

        # Initialize an object of the FlowField class that will store the user-specified flow field:
        flowfield = FlowField(1,
                                   size=(200, 300),
                                   size_buffer=0,
                                   time_separation=1)

        # Create a dummy velocity field:
        flowfield.generate_random_velocity_field(displacement=(10, 10),
                                                 gaussian_filters=(30, 30),
                                                 n_gaussian_filter_iter=6)

        with self.assertRaises(ValueError):

            # Initialize the Gymnasium environment:
            env = PIVEnv(interrogation_window_size=(60, 60),
                         interrogation_window_size_buffer=2,
                         cues_function=cues_function,
                         particle_spec=particle_spec,
                         motion_spec=motion_spec,
                         image_spec=image_spec,
                         user_flowfield=flowfield.velocity_field,
                         inference_model=None,
                         random_seed=100)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
