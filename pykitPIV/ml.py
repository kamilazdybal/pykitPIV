import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import gymnasium as gym
import pygame
import random
import tensorflow as tf
from pykitPIV.checks import *
from pykitPIV.flowfield import FlowField
from pykitPIV.motion import Motion
from pykitPIV.particle import Particle
from pykitPIV.image import Image
from pykitPIV.postprocess import Postprocess
from pykitPIV.flowfield import __available_velocity_fields

########################################################################################################################
########################################################################################################################
####
####    Class: PIVDataset
####
########################################################################################################################
########################################################################################################################

class PIVDataset(Dataset):
    """
    Loads and stores the **pykitPIV**-generated dataset for **PyTorch**.

    This is a subclass of ``torch.utils.data.Dataset``.

    **Example:**

    .. code:: python

        from pykitPIV import PIVDataset

        # Specify the path to the saved dataset:
        path = 'docs/data/pykitPIV-dataset-10-PIV-pairs-256-by-256.h5'

        # Load and store the dataset:
        PIV_data = PIVDataset(dataset=path)

    :param dataset:
        ``str`` specifying the path to the saved dataset.
        It can also be directly passed as a ``dict`` defining the **pykitPIV** dataset.
    :param transform: (optional)
        ``torchvision.transform`` specifying vision transformations to augment the training dataset.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __init__(self, dataset, transform=None):

        if isinstance(dataset, str):

            # Upload the dataset:
            f = h5py.File(dataset, "r")

            # Access image intensities:
            self.data = np.array(f["I"]).astype("float32")

            # Access flow targets:
            self.target = np.array(f["targets"]).astype("float32")

        elif isinstance(dataset, dict):

            # Access image intensities:
            self.data = np.array(dataset["I"]).astype("float32")

            # Access flow targets:
            self.target = np.array(dataset["targets"]).astype("float32")

        # Multiply the v-component of velocity by -1:
        self.target[:,1,:,:] = -self.target[:,1,:,:]

        if isinstance(dataset, str): f.close()

        # Allow for any custom data transforms to be used later:
        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        # Get the sample:
        sample = self.data[idx], self.target[idx]

        # Apply any custom data transforms on this sample:
        if self.transform:
            sample = self.transform(sample)

        return sample

########################################################################################################################
########################################################################################################################
####
####    Class: PIVEnv
####
########################################################################################################################
########################################################################################################################

class PIVEnv(gym.Env):
    """
    Provides a `Gymnasium <https://gymnasium.farama.org/>`_-based virtual PIV environment for a reinforcement learning (RL) agent.

    The environment simulates a 2D section in a wind tunnel of a user-specified size, with synthetic or user-specified
    static velocity field and provides synthetic PIV recordings under a (usually smaller) interrogation window.
    The overall mechanics of this class is visualized below:

    .. image:: ../images/PIVEnv.png
        :width: 700
        :align: center

    The RL agent interacting with this environment is free to locate an interrogation window within
    the larger flow field satisfying certain condition
    modeled by the reward function (see the parameter ``reward_function`` in the ``PIVEnv.step()`` method).
    The agent moves smoothly, i.e., pixel-by-pixel from one interrogation window to the next,
    always performing one of the five actions:

    - `0`: Move up by one pixel
    - `1`: Move right by one pixel
    - `2`: Move down by one pixel
    - `3`: Move left by one pixel
    - `4`: Stay

    **Future functionality will include temporally-evolving flow fields.**

    This is a subclass of `gymnasium.Env <https://gymnasium.farama.org/api/env/#gymnasium.Env>`_.

    **Example:**

    .. code:: python

        from pykitPIV.ml import PIVEnv

        # Prepare specs for pykitPIV parameters:
        particle_spec = {'diameters': (1, 1),
                         'distances': (2, 2),
                         'densities': (0.2, 0.2),
                         'diameter_std': 1,
                         'seeding_mode': 'random'}

        flowfield_spec = {'flowfield_size': (500, 1000),
                          'flowfield_type': 'random smooth',
                          'gaussian_filters': (30, 30),
                          'n_gaussian_filter_iter': 5,
                          'displacement': (2, 2)}

        motion_spec = {'n_steps': 10,
                       'time_separation': 5,
                       'particle_loss': (0, 2),
                       'particle_gain': (0, 2)}

        image_spec = {'exposures': (0.5, 0.9),
                      'maximum_intensity': 2**16-1,
                      'laser_beam_thickness': 1,
                      'laser_over_exposure': 1,
                      'laser_beam_shape': 0.95,
                      'alpha': 1/8,
                      'clip_intensities': True,
                      'normalize_intensities': False}

        # Initialize the Gymnasium environment:
        env = PIVEnv(interrogation_window_size=(100,200),
                     interrogation_window_size_buffer=10,
                     particle_spec=particle_spec,
                     motion_spec=motion_spec,
                     image_spec=image_spec,
                     flowfield_spec=flowfield_spec,
                     user_flowfield=None,
                     random_seed=100)

    :param interrogation_window_size:
        ``tuple`` of two ``int`` elements specifying the size of the interrogation window in pixels :math:`[\\text{px}]`.
        The first number is the window height, :math:`H`, the second number is the window width, :math:`W`.
    :param interrogation_window_size_buffer:
        ``int`` specifying the buffer, :math:`b`, in pixels :math:`[\\text{px}]` to add to the interrogation window size
        in the width and height direction.
        This number should be approximately equal to the maximum displacement that particles are subject to
        in order to allow new particles to arrive into the image area
        and prevent spurious disappearance of particles near image boundaries.
    :param particle_spec:
        ``dict`` or ``particle.ParticleSpec`` object containing specifications for constructing
        an instance of ``Particle`` class.
    :param motion_spec:
        ``dict`` or ``motion.MotionSpec`` object containing specifications for constructing
        an instance of ``Motion`` class.
    :param image_spec:
        ``dict`` or ``image.ImageSpec`` object containing specifications for constructing
        an instance of ``Image`` class.
    :param flowfield_spec: (optional)
        ``dict`` or ``flowfield.FlowFieldSpec`` object containing specifications for constructing
        an instance of ``FlowField`` class. If not specified, the user has to provide their own flow field
        through the ``user_flowfield`` parameter.
    :param user_flowfield: (optional)
        ``numpy.ndarray`` specifying the velocity components for the entire virtual wind tunnel.
        It should be of size :math:`(1, 2, H_{\\text{wt}}, W_{\\text{wt}})`,
        where :math:`1` is just one, fixed flow field, :math:`2` refers to each velocity component
        :math:`u` and :math:`v` respectively,
        :math:`H_{\\text{wt}}` is the height and :math:`W_{\\text{wt}}` is the width of the virtual wind tunnel.
        If not specified, the user has to provide a flow field specification
        through the ``flowfield_spec`` parameter to create a synthetic **pykitPIV**-generated flow field.
        **Future functionality will include temporally-evolving flow fields.**
    :param inference_model: (optional)
        ``class`` specifying the inference model for predicting flow targets from PIV image intensities.
        It can be a CNN-based or a WIDIM-based model.
        If set to ``None``, inference is not done, instead the true flow target within the interrogation window is
        returned.
        The inference model has to have a method ``inference_model.inference()``
        that returns the predicted flow targets tensor of size :math:`(1, 2, H+2b, W+2b)`.
        The ``inference_model.inference()`` method must be able to take as an input the raw PIV images and
        pre-process them as needed.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``.
        If specified, all image generation is reproducible.
    """

    def __init__(self,
                 interrogation_window_size,
                 interrogation_window_size_buffer,
                 particle_spec,
                 motion_spec,
                 image_spec,
                 flowfield_spec=None,
                 user_flowfield=None,
                 inference_model=None,
                 random_seed=None):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:

        # Size of the interrogation window:
        self.__interrogation_window_size = interrogation_window_size

        # Size of the buffer for the interrogation window:
        self.__interrogation_window_size_buffer = interrogation_window_size_buffer

        # Specs for all pykitPIV classes:
        self.__particle_spec = particle_spec
        self.__flowfield_spec = flowfield_spec
        self.__motion_spec = motion_spec
        self.__image_spec = image_spec

        self.__inference_model = inference_model
        self.__random_seed = random_seed

        # Compute the total size of the interrogation window:
        self.__interrogation_window_size_with_buffer = (self.__interrogation_window_size[0] + 2 * self.__interrogation_window_size_buffer,
                                                        self.__interrogation_window_size[1] + 2 * self.__interrogation_window_size_buffer)

        # If the user did not supply their own flow field, a pykitPIV-generated flow field is used:
        if user_flowfield is None:

            # Size of the entire visible flow field:
            self.__flowfield_size = self.__flowfield_spec['flowfield_size']

            self.__flowfield_type = self.__flowfield_spec['flowfield_type']

            # Generate the flow field that is fixed throughout training:
            flowfield = FlowField(n_images=1,
                                  size=self.__flowfield_size,
                                  size_buffer=0,
                                  random_seed=self.__random_seed)

            if self.__flowfield_type == 'random smooth':

                flowfield.generate_random_velocity_field(gaussian_filters=self.__flowfield_spec['gaussian_filters'],
                                                         n_gaussian_filter_iter=self.__flowfield_spec['n_gaussian_filter_iter'],
                                                         displacement=self.__flowfield_spec['displacement'])

            if 'apply_SLM' in self.__flowfield_spec.keys():
                if self.__flowfield_spec['apply_SLM']:

                    # Solve the simplified Langevin model (SLM) for the mean velocity fields:
                    flowfield.generate_langevin_velocity_field(mean_field=flowfield.velocity_field,
                                                               integral_time_scale=self.__flowfield_spec['integral_time_scale'],
                                                               sigma=self.__flowfield_spec['sigma'],
                                                               n_stochastic_particles=self.__flowfield_spec['n_stochastic_particles'],
                                                               n_iterations=self.__flowfield_spec['n_iterations'],
                                                               verbose=False)

            self.flowfield = flowfield

        # Otherwise, use the flow field provided by the user:
        else:

            self.flowfield = user_flowfield

            self.__flowfield_type = 'user'

            self.__flowfield_size = user.flowfield.shape[2::]

        # Observation space for the RL agent: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Calculate the admissible observation space from which the agent can sample the interrogation windows
        self.__admissible_observation_space = (self.__flowfield_size[0] - self.__interrogation_window_size_with_buffer[0],
                                               self.__flowfield_size[1] - self.__interrogation_window_size_with_buffer[1])

        # The observation space is the camera's location in the virtual environement. In practice, this is
        # the position of the camera looking at a specific interrogation window.
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]),
                                                high=np.array([self.__admissible_observation_space[0], self.__admissible_observation_space[1]]),
                                                dtype=int)

        # Action space for the RL agent: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Actions are the camera's movement on the pixel grid:
        self.action_space = gym.spaces.Discrete(5)

        # Dictionary that maps the abstract actions to the directions on the pixel grid:
        self._action_to_direction = {
            0: np.array([1, 0]),  # up
            1: np.array([0, 1]),  # right
            2: np.array([-1, 0]),  # down
            3: np.array([0, -1]),  # left
            4: np.array([0, 0]),  # stay
        }

        self._action_to_verbose_direction = {
            0: 'Up',
            1: 'Right',
            2: 'Down',
            3: 'Left',
            4: 'Stay',
        }

        self.__n_actions = 5

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def n_actions(self):
        return self.__n_actions

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def record_particles(self, camera_position):
        """
        Creates virtual PIV recordings based on the current interrogation window.

        **Example:**

        .. code:: python

            # Once the environment has been initialized:
            env = PIVEnv(...)

            # We can sample an observation which is the camera position:
            camera_position = env.observation_space.sample()

            # We can now create a virtual PIV recording at that camera position:
            image_obj = env.record_particles(camera_position)

        :param camera_position:
            ``numpy.ndarray`` specifying the camera position in pixels :math:`[\\text{px}]`.
            This defines the bottom-left corner of the interrogation window.
            Example can be ``numpy.array([10,50])`` which positions camera at the location :math:`10 \\text{px}`
            along the height dimension and :math:`50 \\text{px}` along the width dimension.

        :return:
            - **image** - ``pykitPIV.Image`` object specifying the generated PIV image pairs.
        """

        # Extract the velocity field under the current interrogation window:
        h_start = camera_position[0]
        h_stop = h_start + self.__interrogation_window_size_with_buffer[0]

        w_start = camera_position[1]
        w_stop = w_start + self.__interrogation_window_size_with_buffer[1]

        velocity_field_magnitude_at_interrogation_window = self.flowfield.velocity_field_magnitude[:, :, h_start:h_stop, w_start:w_stop]
        velocity_field_at_interrogation_window = self.flowfield.velocity_field[:, :, h_start:h_stop, w_start:w_stop]

        # Construct virtual PIV measurements: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Initialize a particle object:
        particles = Particle(n_images=1,
                             size=self.__interrogation_window_size,
                             size_buffer=self.__interrogation_window_size_buffer,
                             diameters=self.__particle_spec['diameters'],
                             distances=self.__particle_spec['distances'],
                             densities=self.__particle_spec['densities'],
                             diameter_std=self.__particle_spec['diameter_std'],
                             seeding_mode=self.__particle_spec['seeding_mode'],
                             random_seed=self.__random_seed)

        # Initialize a flow field object:
        flowfield = FlowField(n_images=1,
                              size=self.__interrogation_window_size,
                              size_buffer=self.__interrogation_window_size_buffer,
                              random_seed=self.__random_seed)

        flowfield.upload_velocity_field(velocity_field_at_interrogation_window)

        # Initialize a motion object:
        motion = Motion(particles,
                        flowfield,
                        time_separation=self.__motion_spec['time_separation'],
                        particle_loss=self.__motion_spec['particle_loss'],
                        particle_gain=self.__motion_spec['particle_gain'],
                        random_seed=self.__random_seed)
        motion.runge_kutta_4th(n_steps=self.__motion_spec['n_steps'])

        # Initialize an image object:
        image = Image(random_seed=self.__random_seed)
        image.add_particles(particles)
        image.add_flowfield(flowfield)
        image.add_motion(motion)

        image.add_reflected_light(exposures=self.__image_spec['exposures'],
                                  maximum_intensity=self.__image_spec['maximum_intensity'],
                                  laser_beam_thickness=self.__image_spec['laser_beam_thickness'],
                                  laser_over_exposure=self.__image_spec['laser_over_exposure'],
                                  laser_beam_shape=self.__image_spec['laser_beam_shape'],
                                  alpha=self.__image_spec['alpha'],
                                  clip_intensities=self.__image_spec['clip_intensities'],
                                  normalize_intensities=self.__image_spec['normalize_intensities'])

        return image

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def make_inference(self, image_obj):
        """
        Makes inference of the displacement field based on the recorded PIV images.

        :param image_obj:
            ``pykitPIV.Image`` object specifying the generated PIV image pairs.

        :return:
            - **prediction_tensor** - ``numpy.ndarray`` specifying the tensor of predicted flow targets.
            - **prediction_tensor** - ``numpy.ndarray`` specifying the tensor of true flow targets.
        """

        images_I1 = image_obj.remove_buffers(image_obj.images_I1)
        images_I2 = image_obj.remove_buffers(image_obj.images_I2)

        images_tensor = image_obj.concatenate_tensors((images_I1, images_I2))
        targets_tensor = image_obj.remove_buffers(image_obj.get_displacement_field())

        if self.__inference_model is not None:
            # Perform inference of the displacement field from the recorded PIV images:
            prediction_tensor = self.__inference_model.inference(images_tensor[0,:,:,:].astype(np.float32))
        else:
            # Return the true displacement field under this interrogation window:
            prediction_tensor = targets_tensor

        return prediction_tensor, targets_tensor

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def reset(self, imposed_camera_position=None):
        """
        Resets the environment to a random or user-imposed initial state.

        **Example:**

        .. code:: python

            # Once the environment has been initialized:
            env = PIVEnv(...)

            # We reset the environment to create its initial state:
            initial_camera_position, _, _ = env.reset()

        :param imposed_camera_position: (optional)
            ``numpy.ndarray`` of two elements specifying the initial camera position in pixels :math:`[\\text{px}]`.
            This defines the bottom-left corner of the interrogation window. Example can be ``numpy.array([10,50])`` which
            positions camera at the location :math:`10 \\text{px}` along the height dimension and :math:`50 \\text{px}`
            along the width dimension. If not specified, random camera position is selected through
            ``env.observation_space.sample()``.

        :return:
            - **camera_position** - ``numpy.ndarray`` of two elements specifying the initial camera position in
              pixels :math:`[\\text{px}]`.
            - **prediction_tensor** - ``numpy.ndarray`` specifying the tensor of predicted flow targets.
            - **prediction_tensor** - ``numpy.ndarray`` specifying the tensor of true flow targets.
        """

        # Check whether the user-specified camera position is possible given the admissible observation space
        # and the size of the interrogation window:
        if imposed_camera_position is not None:
            if (imposed_camera_position[0] < 0) or (imposed_camera_position[0] > self.__admissible_observation_space[0]):
                raise ValueError("The user-imposed camera position falls outside of the admissible virtual environment!")
            if imposed_camera_position[1] < 0 or imposed_camera_position[1] > self.__admissible_observation_space[1]:
                raise ValueError("The user-imposed camera position falls outside of the admissible virtual environment!")

        # Future functionality: Can generate a new flow field, if the user didn't specify a fixed flow field to use.

        if imposed_camera_position is None:
            # Create an initial camera position:
            camera_position = self.observation_space.sample()
            self.__camera_position = camera_position
        else:
            camera_position = imposed_camera_position
            self.__camera_position = camera_position

        # Record PIV images at that camera position:
        image_obj = self.record_particles(camera_position)
        self.__image_obj = image_obj

        # Make inference of displacement field based on the recorded PIV images:
        prediction_tensor, targets_tensor = self.make_inference(image_obj)

        # Save the prediction tensor and the targets tensor as globally-available variables:
        self.__prediction_tensor = prediction_tensor
        self.__targets_tensor = targets_tensor

        return camera_position, prediction_tensor, targets_tensor

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def step(self,
             action,
             reward_function,
             verbose=False):
        """
        Makes one step in the environment which moves the camera to a new position, and computes the associated reward
        for taking that step. The reward is computed based on the PIV images seen at that position, converted
        to a continuous displacement field recovered by an inference model (either CNN-based or WIDIM-based).

        **Example:**

        .. code:: python

            # Once the environment has been initialized:
            env = PIVEnv(...)

            # We reset the environment to create its initial state:
            initial_camera_position, _, _ = env.reset()

            # Now we can take a step in the environment by selecting one of the five actions:
            new_camera_position, reward = env.step(action=4)

        :param action:
            ``tuple`` specifying the action to be taken at the current step in the environment.
        :param reward_function:
            ``function`` specifying the dynamics of the reward construction as a function of predicted and/or true
            flow targets. It can be one of the rewards functions from the ``pykitPIV.Rewards`` class.
        :param verbose: (optional)
            ``bool`` specifying if the verbose print statements should be displayed.
        """

        # Map the action (element of {0,1,2,3,4}) to the new camera position:
        direction = self._action_to_direction[action]

        if verbose:
            print('Action ' + str(action) + ': ' + self._action_to_verbose_direction[action])

        # Take the step in the environment:
        # (We clip the camera position to make sure that we don't leave the grid bounds)
        camera_position = np.array([np.clip(self.__camera_position[0] + direction[0], 0, self.__admissible_observation_space[0]),
                                    np.clip(self.__camera_position[1] + direction[1], 0, self.__admissible_observation_space[1])])

        # Reset the camera position:
        self.__camera_position = camera_position

        # Reward construction: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Record PIV images at that camera position:
        image_obj = self.record_particles(camera_position)
        self.__image_obj = image_obj

        # Make inference of displacement field based on the recorded PIV images:
        prediction_tensor, targets_tensor = self.make_inference(image_obj)

        # Save the prediction tensor and the targets tensor as globally-available variables:
        self.__prediction_tensor = prediction_tensor
        self.__targets_tensor = targets_tensor

        reward = reward_function(prediction_tensor,
                                 targets_tensor)

        return camera_position, reward

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def render(self,
               camera_position,
               c='white',
               s=10,
               lw=2,
               figsize=None,
               normalize_cbars=False,
               cmap='viridis',
               add_streamplot=False,
               streamplot_density=1,
               streamplot_color='k',
               streamplot_linewidth=1,
               filename=None):
        """
        Renders the virtual wind tunnel with the current interrogation window.

        **Example:**

        .. code:: python

            # Once the environment has been initialized:
            env = PIVEnv(...)

            # We can sample an observation which is the camera position:
            camera_position = env.observation_space.sample()

            # And we can render the virtual wind tunnel with the current interrogation window:
            env.render(camera_position,
                       c='white',
                       s=20,
                       lw=1,
                       figsize=(12,6))

        One example rendering of the virtual wind tunnel can look like this:

        .. image:: ../images/ml_PIVEnv_render.png
            :width: 800
        """

        __visualize = ['displacement', 'velocity']

        fontsize = 14

        if figsize is not None:
            figure = plt.figure(figsize=figsize)
        else:
            figure = plt.figure(figsize=(25, 10))

        spec = figure.add_gridspec(ncols=9, nrows=3, width_ratios=[1, 0.2, 1, 0.2, 1, 0.2, 1, 0.2, 1], height_ratios=[2, 0.2, 1])

        # Visualize a rectangle that defines the virtual wind tunnel:
        figure.add_subplot(spec[0, 0:9])
        ims = plt.imshow(self.flowfield.velocity_field_magnitude[0,0,:,:], cmap=cmap, origin='lower', zorder=0)
        plt.colorbar(ims)

        if add_streamplot:
            X = np.arange(0, self.flowfield.size[1], 1)
            Y = np.arange(0, self.flowfield.size[0], 1)

            plt.streamplot(X, Y,
                           self.flowfield.velocity_field[0, 0, :, :],
                           self.flowfield.velocity_field[0, 1, :, :],
                           density=streamplot_density,
                           color=streamplot_color,
                           linewidth=streamplot_linewidth,
                           zorder=1)

        plt.scatter(camera_position[1]-0.5, camera_position[0]-0.5, c=c, s=s, zorder=2)

        if normalize_cbars:
            vmin = np.min(self.flowfield.velocity_field_magnitude[0,0,:,:])
            vmax = np.max(self.flowfield.velocity_field_magnitude[0,0,:,:])
        else:
            vmin = None
            vmax = None

        # Visualize a rectangle that defines the current interrogation window:
        rect = patches.Rectangle((camera_position[1]-0.5, camera_position[0]-0.5),
                                 self.__interrogation_window_size_with_buffer[1],
                                 self.__interrogation_window_size_with_buffer[0],
                                 linewidth=lw, edgecolor=c, facecolor='none', zorder=2)
        ax = plt.gca()
        ax.add_patch(rect)
        plt.title('Virtual wind tunnel', fontsize=fontsize)

        # Visualize the target under the interrogation window:
        figure.add_subplot(spec[2, 0])
        targets_magnitude = np.sqrt(self.__targets_tensor[0,0,:,:]**2 + self.__targets_tensor[0,1,:,:]**2)
        ims = plt.imshow(targets_magnitude, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, zorder=0)
        plt.colorbar(ims)

        if add_streamplot:
            X = np.arange(0, self.__interrogation_window_size[1], 1)
            Y = np.arange(0, self.__interrogation_window_size[0], 1)

            plt.streamplot(X, Y,
                           self.__targets_tensor[0,0,:,:],
                           self.__targets_tensor[0,1,:,:],
                           density=streamplot_density/5,
                           color=streamplot_color,
                           linewidth=streamplot_linewidth,
                           zorder=1)

        plt.title('Target', fontsize=fontsize)

        # Visualize I1:
        figure.add_subplot(spec[2, 2])
        images_I1 = self.__image_obj.remove_buffers(self.__image_obj.images_I1)
        ims = plt.imshow(images_I1[0,0,:,:], origin='lower', cmap='Greys_r')
        plt.colorbar(ims)
        plt.title(r'$I_1$', fontsize=fontsize)

        # Visualize I2:
        figure.add_subplot(spec[2, 4])
        images_I2 = self.__image_obj.remove_buffers(self.__image_obj.images_I2)
        ims = plt.imshow(images_I2[0,0,:,:], origin='lower', cmap='Greys_r')
        plt.colorbar(ims)
        plt.title(r'$I_2$', fontsize=fontsize)

        # Visualize inference under the interrogation window:
        figure.add_subplot(spec[2, 6])
        prediction_magnitude = np.sqrt(self.__prediction_tensor[0, 0, :, :] ** 2 + self.__prediction_tensor[0, 1, :, :] ** 2)
        ims = plt.imshow(prediction_magnitude, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(ims)

        if add_streamplot:
            X = np.arange(0, self.__interrogation_window_size[1], 1)
            Y = np.arange(0, self.__interrogation_window_size[0], 1)

            plt.streamplot(X, Y,
                           self.__prediction_tensor[0,0,:,:],
                           self.__prediction_tensor[0,1,:,:],
                           density=streamplot_density/5,
                           color=streamplot_color,
                           linewidth=streamplot_linewidth,
                           zorder=1)

        plt.title('Inference', fontsize=fontsize)

        # Visualize the error map between the target and the inference:
        figure.add_subplot(spec[2, 8])
        prediction_error = np.abs(targets_magnitude - prediction_magnitude)
        normalizer = np.max(targets_magnitude) - np.min(targets_magnitude)
        error_map = prediction_error/normalizer*100
        ims = plt.imshow(error_map, origin='lower', cmap='Greys')
        plt.colorbar(ims)
        plt.title('Error %: ' + str(round(np.mean(error_map), 1)), fontsize=fontsize)

        plt.savefig(filename, dpi=300, bbox_inches='tight')

        return plt

########################################################################################################################
########################################################################################################################
####
####    Class: CameraAgent
####
########################################################################################################################
########################################################################################################################

class CameraAgent:
    """
    Creates a reinforcement learning agent that operates a virtual camera in a PIV experimental setting
    and provides a training loop for Q-learning. Q-learning uses a deep neural network (DNN) model.

    **Example:**

    .. code:: python

        from pykitPIV.ml import PIVEnv
        from pykitPIV.ml import CameraAgent
        import tensorflow as tf

        # Initialize the environment:
        env = PIVEnv(...)

        # Create a simple neural network model for the Q-network:
        class QNetwork(tf.keras.Model):

            def __init__(self, n_actions):

                super(QNetwork, self).__init__()

                self.dense1 = tf.keras.layers.Dense(10, activation='linear', kernel_initializer=tf.keras.initializers.Ones)
                self.dense2 = tf.keras.layers.Dense(10, activation='linear', kernel_initializer=tf.keras.initializers.Ones)
                self.output_layer = tf.keras.layers.Dense(n_actions, activation='linear', kernel_initializer=tf.keras.initializers.Ones)

            def call(self, state):

                x = self.dense1(state)
                x = self.dense2(x)

                return self.output_layer(x)

        # Initialize the camera agent:
        ca = CameraAgent(env=env,
                         target_q_network=QNetwork(env.n_actions),
                         selected_q_network=QNetwork(env.n_actions),
                         memory_size=1000,
                         batch_size=10,
                         n_epochs=10,
                         learning_rate=0.001,
                         optimizer='RMSprop',
                         discount_factor=0.95)

    :param env:
        ``gym.Env`` specifying the virtual environment.
    :param target_q_network:
        ``tf.keras.Model`` specifying the deep neural network that will be the target network for Q-learning.
    :param selected_q_network:
        ``tf.keras.Model`` specifying the deep neural network that will be the temporary network for Q-learning.
    :param memory_size:  (optional)
        ``int`` specifying the size of the memory bank.
    :param batch_size:  (optional)
        ``int`` specifying the batch size for training the Q-network for after each step in the environment.
    :param n_epochs:  (optional)
        ``int`` specifying the number of epochs to train the Q-network for after each step in the environment.
    :param learning_rate:  (optional)
        ``float`` specifying the learning rate.
    :param optimizer:  (optional)
        ``str`` specifying the gradient descent optimizer to use.
    :param discount_factor:  (optional)
        ``float`` specifying the discount factor, :math:`\gamma`.
    """

    def __init__(self,
                 env,
                 target_q_network,
                 selected_q_network,
                 memory_size=1000,
                 batch_size=10,
                 n_epochs=10,
                 learning_rate=0.001,
                 optimizer='RMSprop',
                 discount_factor=0.95):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:

        # Define the environment for the camera agent:
        self.env = env
        self.n_actions = self.env.n_actions

        print('The uploaded environment has ' + str(self.n_actions) + ' actions.')

        # We have two Q-networks, the target Q-network:
        self.target_q_network = target_q_network

        # But the target network will be synchronized with the selected Q-network only once every few episodes:
        self.selected_q_network = selected_q_network

        # This prevents the Q-network from competing against itself.

        # Parameters of training the DNN:
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        if optimizer == 'RMSprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        self.target_q_network.compile(self.optimizer, loss=tf.keras.losses.MeanSquaredError())
        self.selected_q_network.compile(self.optimizer, loss=tf.keras.losses.MeanSquaredError())
        self.MSE_losses = []

        # Reinforcement learning parameters:
        self.discount_factor = discount_factor

        # Memory replay parameters:

        class ReplayBuffer:
            def __init__(self, max_size):
                self.buffer = deque(maxlen=max_size)

            def add(self, experience):
                self.buffer.append(experience)

            def sample(self, batch_size):
                return random.sample(self.buffer, batch_size)

        self.memory_size = memory_size
        self.memory = ReplayBuffer(self.memory_size)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def choose_action(self,
                      camera_position,
                      epsilon,
                      centers=None,
                      scales=None):
        """
        Defines an epsilon-greedy choice of the next best action to select.
        """

        # Select random action with probability epsilon:
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)

        # Select the currently best action with probability (1 - epsilon):
        else:
            if (centers is not None) and (scales is not None):
                preprocessed_camera_position = ((camera_position[0] - centers[0]) / scales[0], (camera_position[1] - center[1]) / scales[1])
                preprocessed_camera_position = np.reshape(preprocessed_camera_position, (1, -1))

            else:
                preprocessed_camera_position = np.reshape(camera_position, (1, -1))

            q_values = self.target_q_network.predict(preprocessed_camera_position, verbose=0)

            return np.argmax(q_values)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def remember(self, camera_position, action, reward, next_camera_position):
        """
        Adds a step in the environment to the memory bank.
        """

        self.memory.add((camera_position, action, reward, next_camera_position))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def train(self, current_lr=0.001):
        """
        Trains the Q-network with the outcome of a single step in the environment.

        **Example:**

        .. code:: python

            # Once the camera agent has been initialized:
            ca = CameraAgent(...)

            # We can train the agent with a single pass of a batch of training data:
            ca.train(current_lr=0.001)

        :param current_lr: (optional)
            ``float`` specifying the learning rate to use in the current pass over the minibatch.
        """

        # Before the buffer fills with actual data, we're not training yet:
        if len(self.memory.buffer) < self.batch_size:
            return

        # Once the memory buffer holds enough observations...

        # Now we can start sampling batches from the memory:
        minibatch = self.memory.sample(self.batch_size)

        batch_camera_positions = np.zeros((self.batch_size, 2))
        batch_q_values = np.zeros((self.batch_size, self.n_actions))

        for i, batch_content in enumerate(minibatch):

            camera_position, action, reward, next_camera_position = batch_content

            camera_position = np.reshape(camera_position, (1, -1))
            next_camera_position = np.reshape(next_camera_position, (1, -1))

            # Compute the Q-value we want to have for this (camera_position, action) pair:
            next_q_values = self.target_q_network.predict(next_camera_position, verbose=0)
            target_q_value = reward + self.discount_factor * np.max(next_q_values, axis=-1)

            # Compute the Q-value we actually have for this (camera_position, action) pair:
            q_values = self.selected_q_network.predict(camera_position, verbose=0)

            # Swap the Q-value we actually have for the target Q-value for this action:
            q_values[0][action] = target_q_value[0]

            # Create a batch:
            batch_camera_positions[i, :] = camera_position
            batch_q_values[i, :] = q_values

        # Update the optimizer with the current learning rate:
        self.optimizer.learning_rate.assign(current_lr)

        # Teach the Q-network to predict the target Q-values over the current batch:
        history = self.selected_q_network.fit(batch_camera_positions,
                                              batch_q_values,
                                              epochs=self.n_epochs,
                                              verbose=0)

        # Append the losses:
        self.MSE_losses.append(history.history['loss'])

    def update_target_network(self):
        """
        Synchronizes the target Q-network with the selected Q-network.

        This function should be called once every a couple of episodes.

        Too frequent synchronizations can make the target Q-network to compete with itself and can slow down learning.

        **Example:**

        .. code:: python

            # The general abstraction for synchronizing the two Q-networks in a training loop can be the following:
            for episode in range(0,n_episodes):

                ...

                # Synchronize the networks once every 10 episodes:
                if (episode+1) % 10 == 0:
                    ca.update_target_network()

        """

        self.target_q_network.set_weights(self.selected_q_network.get_weights())

    def view_weights(self):
        """
        Returns the latest weights and biases of the target Q-network.

        **Example:**

        .. code:: python

            # Once the camera agent has been initialized:
            ca = CameraAgent(...)

            # View the current target Q-network weights:
            ca.view_weights()
        """

        return self.target_q_network.get_weights()







########################################################################################################################
########################################################################################################################
####
####    Class: Rewards
####
########################################################################################################################
########################################################################################################################


class Rewards:
    """
    Provides custom reward functions that are applicable to various tasks related to navigating a virtual wind tunnel
    and a virtual PIV experiment.

    Each function returns the reward, :math:`R`.

    **Example:**

    .. code:: python

        from pykitPIV.ml import Rewards

        # Instantiate an object of the Rewards class:
        rewards = Rewards()

    :param verbose: (optional)
        ``bool`` specifying if the verbose print statements should be displayed.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``.
        If specified, all image generation is reproducible.
    """

    def __init__(self,
                 verbose=False,
                 random_seed=None):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:
        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be of type 'bool'.")

        if random_seed is not None:
            if type(random_seed) != int:
                raise ValueError("Parameter `random_seed` has to be of type 'int'.")
            else:
                np.random.seed(seed=random_seed)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:
        self.__verbose = verbose
        self.__random_seed = random_seed

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Properties coming from user inputs:
    @property
    def random_seed(self):
        return self.__random_seed

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def q_criterion(self,
                    velocity_field,
                    transformation=None):
        """
        Computes the reward based on the Q-criterion.

        **Example:**

        .. code:: python

            from pykitPIV.ml import Rewards

            # Once we have a velocity field specified:
            velocity_field = ...

            # Instantiate an object of the Rewards class:
            rewards = Rewards()

            # Compute the reward based on the Q-criterion for the present velocity field:
            reward = rewards.q_criterion(velocity_field=velocity_field)

        :param velocity_field:
            ``numpy.ndarray`` specifying the velocity components under the interrogation window.
            It should be of size :math:`(1, 2, H_{\\text{i}}+b, W_{\\text{i}}+b)`,
            where :math:`1` is just one, fixed flow field, :math:`2` refers to each velocity component
            :math:`u` and :math:`v` respectively,
            :math:`H_{\\text{i}}+b` is the height and :math:`W_{\\text{i}}+b` is the width of the interrogation window.
        :param transformation: (optional)
            ``function`` specifying arbitrary transformation of the Q-criterion.

        :return:
            - **reward** - ``float`` specifying the reward, :math:`R`.
        """

        if transformation is None:
            reward = q_criterion(velocity_field)
        else:
            reward = transformation(q_criterion(velocity_field))

        return reward

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -