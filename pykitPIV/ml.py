import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gymnasium as gym
import pygame
from tf_agents.environments import py_environment
from pykitPIV.checks import *
from pykitPIV.flowfield import FlowField
from pykitPIV.motion import Motion
from pykitPIV.particle import Particle
from pykitPIV.image import Image
from pykitPIV.postprocess import Postprocess

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

    This is a subclass of ``gymnasium.Env``.
    """

    def __init__(self,
                 interrogation_window_size=(128,128),
                 interrogation_window_size_buffer=10,
                 flowfield_size=(512,2048),
                 user_flowfield=None):

        # Size of the interrogation window:
        self.interrogation_window_size = interrogation_window_size

        # Size of the buffer for the interrogation window:
        self.interrogation_window_size_buffer = interrogation_window_size_buffer

        self.__interrogation_window_size_with_buffer = (self.interrogation_window_size[0] + self.interrogation_window_size_buffer,
                                                        self.interrogation_window_size[1] + self.interrogation_window_size_buffer)

        # If the user did not supply their own flow field, a pykitPIV-generated flow field is used:
        if user_flowfield is None:

            # Size of entire visible flow field:
            self.flowfield_size = flowfield_size

            # Generate the flow field that is fixed throughout training:
            flowfield = FlowField(n_images=1,
                                  size=self.flowfield_size,
                                  size_buffer=0,
                                  random_seed=100)

            flowfield.generate_random_velocity_field(gaussian_filters=(30, 30),
                                                     n_gaussian_filter_iter=10,
                                                     displacement=(2, 2))

            self.flowfield = flowfield

        # Otherwise, use the flow field provided by the user:
        else:

            self.flowfield = user_flowfield

            self.flowfield_size = user.flowfield.shape[2::]


        self.__admissible_observation_space = (self.flowfield_size[0] - self.__interrogation_window_size_with_buffer[0],
                                               self.flowfield_size[1] - self.__interrogation_window_size_with_buffer[1])

        # The observation space is the camera's location in the virtual environement. In practice, this is
        # the position of the camera looking at a specific interrogation window.
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]),
                                            high=np.array([self.__admissible_observation_space[0], self.__admissible_observation_space[1]]),
                                            dtype=int)

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

    def _record_particles(self, camera_position):
        """
        Creates virtual PIV recordings based on the current interrogation window.
        """

        # Extract the velocity field under the current interrogation window:
        h_coordinates = 1
        w_coordinates = 1








    def plot_state(self,
                   camera_position,
                   c='white',
                   s=10,
                   lw=2,
                   figsize=None):

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

    def _get_obs(self):

        return self.observation_space.sample()

    def _get_info(self):

        pass

    def reset(self):


        # Can generate a new flow field, if the user didn't specify a fixed flow field to use.


        # Create an initial camera position:
        camera_position = self.observation_space.sample()









        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self):


        pass



    def render(self):


        pass


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

class PIVPyEnvironment(py_environment.PyEnvironment):
    """
    Provides a virtual PIV/BOS **TF-Agents**-based environment for a reinforcement learning (RL) agent.

    This is a subclass of ``tf_agents.environments.py_environment.PyEnvironment``.
    """

    def __init__(self, size=(128,128)):

        # The size of the square grid
        self.size = size