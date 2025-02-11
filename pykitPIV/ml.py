import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gymnasium as gym
import pygame
# from tf_agents.environments import py_environment
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
    Provides a virtual PIV **Gymnasium**-based environment for a reinforcement learning (RL) agent.

    The agent is free to locate an interrogation window within the larger flow field satisfying certain condition.

    The larger flowfield can be provided by the user, or a synthetic flowfield using **pykitPIV** can be generated.

    This is a subclass of ``gymnasium.Env``.

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
        ``numpy.ndarray`` specifying the velocity components. It should be of size :math:`(1, 2, H+2b, W+2b)`,
        where :math:`1` is just one, fixed flow field, :math:`2` refers to each velocity component
        :math:`u` and :math:`v` respectively,
        :math:`H+2b` is the height and :math:`W+2b` is the width of an interrogation window.
        If not specified, the user has to provide a flow field specification
        through the ``flowfield_spec`` parameter to create a synthetic **pykitPIV**-generated flow field.
        **Future functionality can include temporal flow fields.**
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
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
            4: np.array([0, 0]),  # stay
        }

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
            ``tuple`` specifying the camera position in pixels :math:`[\\text{px}]`.
            This defines the bottom-left corner of the interrogation window.
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

        print(velocity_field_at_interrogation_window.shape)

        print(self.__interrogation_window_size)
        print(self.__interrogation_window_size_buffer)

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

    def reset(self):
        """
        Resets the environement to a random initial state.
        """

        # Can generate a new flow field, if the user didn't specify a fixed flow field to use.


        # Create an initial camera position:
        camera_position = self.observation_space.sample()

        # Record PIV images at that camera position:
        image_obj = self.record_particles(camera_position)

        # Perform LIMA inference of the displacement field from the recorded PIV images:
        # General abstraction will be:
        # prediction = LIMA(image_obj)




        return camera_position

    def step(self):
        """
        Makes one step in the environement which moves the camera to a new position.
        """

        pass


    def render(self,
               camera_position,
               c='white',
               s=10,
               lw=2,
               figsize=None,
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

        if figsize is not None:
            plt.figure(figsize=figsize)

        plt.imshow(self.flowfield.velocity_field_magnitude[0,0,:,:], origin='lower')
        plt.scatter(camera_position[1]-0.5, camera_position[0]-0.5, c=c, s=s)

        # Visualize a rectangle that defines the current interrogation window:
        rect = patches.Rectangle((camera_position[1]-0.5, camera_position[0]-0.5),
                                 self.__interrogation_window_size_with_buffer[1],
                                 self.__interrogation_window_size_with_buffer[0],
                                 linewidth=lw, edgecolor=c, facecolor='none')
        ax = plt.gca()
        ax.add_patch(rect)

        plt.colorbar()

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
    and provides a training loop for Q-learning.
    """

    def __init__(self,
                 env,
                 learning_rate=0.001,
                 initial_epsilon=1.0,
                 epsilon_decay=0.01,
                 final_epsilon=0.1,
                 discount_factor=0.95):

        pass


########################################################################################################################
########################################################################################################################
####
####    Class: PIVPyEnvironment
####
########################################################################################################################
########################################################################################################################

# class PIVPyEnvironment(py_environment.PyEnvironment):
#     """
#     Provides a virtual PIV/BOS **TF-Agents**-based environment for a reinforcement learning (RL) agent.
#
#     This is a subclass of ``tf_agents.environments.py_environment.PyEnvironment``.
#     """
#
#     def __init__(self, size=(128,128)):
#
#         # The size of the square grid
#         self.size = size