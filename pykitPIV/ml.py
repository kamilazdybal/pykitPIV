import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import gymnasium as gym
import random
import tensorflow as tf
from pykitPIV.checks import *
from pykitPIV.flowfield import FlowField, FlowFieldSpecs, compute_divergence, compute_vorticity, compute_q_criterion
from pykitPIV.motion import Motion, MotionSpecs
from pykitPIV.particle import Particle, ParticleSpecs
from pykitPIV.image import Image, ImageSpecs
from pykitPIV.postprocess import Postprocess
from pykitPIV.flowfield import _available_velocity_fields

########################################################################################################################
########################################################################################################################
####
####    Class: PIVDatasetPyTorch
####
########################################################################################################################
########################################################################################################################

class PIVDatasetPyTorch(Dataset):
    """
    Loads and stores the **pykitPIV**-generated dataset for **PyTorch**.

    .. note::

        The image intensities have to be indexed by ``"I"`` and the targets have to be indexed by ``"targets"``
        within the dataset dictionary.

    This is a subclass of ``torch.utils.data.Dataset``.

    **Example:**

    .. code:: python

        from pykitPIV import PIVDatasetPyTorch

        # Specify the path to the saved dataset:
        path = 'docs/data/pykitPIV-dataset-10-PIV-pairs-256-by-256.h5'

        # Load and store the dataset:
        PIV_data = PIVDatasetPyTorch(dataset=path)

    :param dataset:
        ``str`` specifying the path to the saved dataset.
        ``str`` specifying the path to the saved dataset.
        It can also be directly passed as a ``dict`` defining the **pykitPIV** dataset.
    :param transform: (optional)
        ``torchvision.transform`` specifying vision transformations to augment the training dataset.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __init__(self,
                 dataset,
                 transform=None):

        super().__init__()

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
####    Class: PIVDatasetTF
####
########################################################################################################################
########################################################################################################################

class PIVDatasetTF(tf.keras.utils.PyDataset):
    """
    Loads and stores the **pykitPIV**-generated dataset for **TensorFlow** or **Keras**.

    .. note::

        The image intensities have to be indexed by ``"I"`` and the targets have to be indexed by ``"targets"``
        within the dataset dictionary.

    This is a subclass of ``tf.keras.utils.PyDataset``.

    **Example:**

    .. code:: python

        from pykitPIV import PIVDatasetTF

        # Specify the path to the saved dataset:
        path = 'docs/data/pykitPIV-dataset-10-PIV-pairs-256-by-256.h5'

        # Load and store the dataset:
        PIV_data = PIVDatasetTF(dataset=path)

    :param dataset:
        ``str`` specifying the path to the saved dataset.
        ``str`` specifying the path to the saved dataset.
        It can also be directly passed as a ``dict`` defining the **pykitPIV** dataset.
    :param transform: (optional)
        ``callable`` specifying vision transformations to augment the training dataset.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __init__(self,
                 dataset,
                 transform=None):

        super().__init__()

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
####    Class: PIVCVAE
####
########################################################################################################################
########################################################################################################################
class PIVCVAE(tf.keras.Model):
    """
    Provides a convolutional variational autoencoder (CVAE) for generating new velocity fields,
    displacement fields, or image intensities samples that match in distribution those coming from an experiment.
    This approach can be used to extend the training data for transfer learning and can help adapt a machine learning
    model to the changing experimental conditions.

    For more information on building convolutional VAEs, the user is referred to this
    `TensorFlow's CVAE tutorial <https://www.tensorflow.org/tutorials/generative/cvae>`_.

    The general workflow for augmenting training datasets with samples that fall in the distribution
    of a given experiment is illustrated below:

    .. image:: ../images/PIVCVAE.svg
        :width: 800
        :align: center

    This is a subclass of ``tensorflow.keras.Model``.

    **Example:**

    .. code:: python

        from pykitPIV import PIVCVAE

        # Specify the shape of the images at the input of CVAE:
        input_shape = (256, 256, 2)

        # Specify the latent dimension (typically mean + log-variance):
        latent_dimension = 2

        # Instantiate a CVAE model:
        model = PIVCVAE(input_shape=input_shape,
                        latent_dimension=latent_dimension)

    .. note::

        Note that TensorFlow's convention for image size for convolutions is different from that of PyTorch.
        While PyTorch accepts :math:`(N, C, H, W)`, TensorFlow uses the convention :math:`(N, H, W, C)`
        with the channel being the last dimension. To prepare **pykitPIV**'s images for TensorFlow's convolutional layers,
        you can simply run the following transpose:

        .. code:: python

            PIV_images = np.transpose(PIV_images, (0, 2, 3, 1))

    :param input_shape:
        ``tuple`` of ``int`` specifying the path shape of the input image, or image pair. Typically, this will be
        ``(H, W, 1)`` for a single scalar quantity or ``(H, W, 2)`` for 2D vector quantities. For the moment,
        height and width should be divisible by 4 due to the architecture used.
    :param latent_dimension: (optional)
        ``int`` specifying the latent dimension of the CVAE.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __init__(self,
                 input_shape,
                 latent_dimension):

        super().__init__()

        self.latent_dimension = latent_dimension

        (H, W, _) = input_shape

        encoded_H = H // 4
        encoded_W = W // 4

        # Construct a convolutional encoder:
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=input_shape),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dimension + latent_dimension), ])

        # Construct a convolutional decoder:
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(latent_dimension,)),
            tf.keras.layers.Dense(units=encoded_H * encoded_W * 32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(encoded_H, encoded_W, 32)),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=3, strides=1, padding='same'), ])

    @tf.function
    def sample(self,
               eps=None):
        """
        Draws an image sample from decoding the latent space using the random vector, :math:`\\varepsilon`.

        :param eps: (optional)
            ``tensorflow.Tensor`` specifying the random noise, :math:`\\varepsilon`, for random new sample generation.
            If set to ``None``,
            a tensor of 100 random noise values will be generated from the normal distribution.

        :return:
            - **sample** - ``tensorflow.Tensor`` specifying the new image sample.
        """

        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dimension))

        return self.decode(eps, apply_sigmoid=True)

    def encode(self,
               x):
        """
        Encodes an image sample down to the mean, :math:`\mu`, and the log-variance, :math:`\\ln(\\sigma^2)`.

        :param x: (optional)
            ``tensorflow.Tensor`` specifying the input image, preprocessed as necessary.

        :return:
            - **mean** - ``tensorflow.Tensor`` specifying the mean of the distribution.
            - **logvar** - ``tensorflow.Tensor`` specifying the log-variance, :math:`\\ln(\\sigma^2)`, of the distribution.
        """

        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)

        return mean, logvar

    def reparameterize(self,
                       mean,
                       logvar):
        """
        Computes the reparameterization trick to generate a sample :math:`z`:

        .. math::

            z = \\mu + \\sigma \\times \\varepsilon

        where :math:`\\mu` is the mean, :math:`\\sigma` is the standard deviation of the distribution,
        and :math:`\\varepsilon` is the random noise. Note that

        .. math::

            \\sigma = \\sqrt{\\exp(\\ln(\\sigma^2))} = \\exp \\big( \\frac{1}{2} \\cdot \\ln(\\sigma^2) \\big)

        and therefore :math:`z` is computed in this function as:

        .. math::

            z = \\mu + \\exp \\big( \\frac{1}{2} \\cdot \\ln(\\sigma^2) \\big) \\times \\varepsilon

        :param mean:
            ``tensorflow.Tensor`` specifying the mean of the distribution.
        :param logvar:
            ``tensorflow.Tensor`` specifying the log-variance, :math:`\\ln(\\sigma^2)`, of the distribution.

        :return:
            - **z** - ``tensorflow.Tensor`` specifying the latent variable, :math:`z`.
        """

        eps = tf.random.normal(shape=mean.shape)

        z = eps * tf.exp(logvar * .5) + mean

        return z

    def decode(self,
               z,
               apply_sigmoid=False):
        """
        Decodes a sample of the latent space into logits or probabilities.

        :param z:
            ``tensorflow.Tensor`` specifying the latent variable.
        :param apply_sigmoid: (optional)
            ``bool`` specifying whether the sigmoid should be applied to the logits.

        :return:
            - **logits** - ``tensorflow.Tensor`` specifying the logits if ``apply_sigmoid=False`` or probabilities if ``apply_sigmoid=True``.
        """

        logits = self.decoder(z)

        if apply_sigmoid:

            probs = tf.sigmoid(logits)

            return probs

        return logits

########################################################################################################################
########################################################################################################################
####
####    Class: PIVEnv
####
########################################################################################################################
########################################################################################################################

class PIVEnv(gym.Env):
    """
    Provides a `Gymnasium <https://gymnasium.farama.org/>`_-based virtual PIV environment
    for a reinforcement learning (RL) agent.

    The environment simulates a 2D section in a wind tunnel of a user-specified size, with synthetic or user-specified
    static velocity field, :math:`\\vec{V} = [u, v]`, and provides synthetic PIV recordings under a
    (usually much smaller) interrogation window.

    The flow target for the RL agent is the displacement field,
    :math:`d\\vec{\\mathbf{s}} = [dx, dy] = [u \\Delta t, v \\Delta t]`. This flow target is the basis for computing
    sensory cues and rewards.

    The overall mechanics of this class is visualized below:

    .. image:: ../images/PIVEnv.svg
        :width: 700
        :align: center

    We refer to :math:`H_{\\text{wt}}` as the height and :math:`W_{\\text{wt}}` as the width of the virtual wind tunnel,
    and to :math:`H_{\\text{i}}` as the height and :math:`W_{\\text{i}}` as the width of the interrogation window.

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

        flowfield_spec = {'size': (500, 1000),
                          'time_separation': 5,
                          'flowfield_type': 'random smooth',
                          'gaussian_filters': (30, 30),
                          'n_gaussian_filter_iter': 5,
                          'displacement': (2, 2)}

        motion_spec = {'n_steps': 10,
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

    Below is an example of importing a user-defined flow field and using it as the flow in the virtual wind tunnel.
    This can be done by uploading an external velocity field tensor to a ``FlowField`` class object. Below, we demonstrate
    this on a velocity field generated using the ``FlowField`` class, but the methodology is such that the user can
    replace the ``velocity_field`` tensor with a custom tensor that can be upladed externally, say, from a
    `Johns Hopkins Turbulence Database (JHTD) <https://turbulence.idies.jhu.edu/home>`_.

    .. code:: python

        from pykitPIV import FlowField

        # Initialize an object of the FlowField class that will store the user-specified flow field:
        user_flowfield = FlowField(1,
                                   size=(500, 1000),
                                   size_buffer=0,
                                   time_separation=1)

        # Create a dummy velocity field:
        user_flowfield.generate_random_velocity_field(displacement=(10, 10),
                                                 gaussian_filters=(30, 30),
                                                 n_gaussian_filter_iter=6)

        velocity_field = flowfield.velocity_field

        # Upload a user-specified velocity field tensor:
        user_flowfield.upload_velocity_field(velocity_field)

        # Initialize the Gymnasium environment:
        env = PIVEnv(interrogation_window_size=(100, 200),
                     interrogation_window_size_buffer=10,
                     cues_function=cues_function,
                     particle_spec=particle_spec,
                     motion_spec=motion_spec,
                     image_spec=image_spec,
                     flowfield_spec=None,
                     user_flowfield=user_flowfield,
                     inference_model=None,
                     random_seed=100)

    Note that at any stage of training the RL agent, the time separation between two PIV image frames can be updated
    using the ``time_separation`` attribute:

    .. code:: python

        env.time_separation = 2

    :param interrogation_window_size:
        ``tuple`` of two ``int`` elements specifying the size of the interrogation window in pixels :math:`[\\text{px}]`.
        The first number is the window height, :math:`H_{\\text{i}}`,
        the second number is the window width, :math:`W_{\\text{i}}`.
    :param interrogation_window_size_buffer:
        ``int`` specifying the buffer, :math:`b`, in pixels :math:`[\\text{px}]` to add to the interrogation window size
        in the width and height direction.
        This number should be approximately equal to the maximum displacement that particles are subject to
        in order to allow new particles to arrive into the image area
        and prevent spurious disappearance of particles near image boundaries.
    :param cues_function:
        ``function`` specifying the computation of cues that the RL agent senses based on the current
        displacement field reconstructed from PIV image pairs. The cues are the input parameters
        to the Q-network, hence the goal of the RL is to learn the mapping:
        :math:`\\text{cues} \\rightarrow \\text{actions}`.
        This function has to take as an input the predicted displacement field tensor
        and return a ``numpy`` array of shape ``(1, N)`` of :math:`N` cues computed
        from the predicted displacement field tensor.
    :param particle_spec: (optional)
        ``dict`` or ``particle.ParticleSpec`` object containing specifications for constructing
        an instance of ``Particle`` class.
    :param motion_spec: (optional)
        ``dict`` or ``motion.MotionSpec`` object containing specifications for constructing
        an instance of ``Motion`` class.
    :param image_spec: (optional)
        ``dict`` or ``image.ImageSpec`` object containing specifications for constructing
        an instance of ``Image`` class.
    :param flowfield_spec: (optional)
        ``dict`` or ``flowfield.FlowFieldSpec`` object containing specifications for constructing
        an instance of ``FlowField`` class. If not specified, the user has to provide their own flow field
        through the ``user_flowfield`` parameter.
    :param user_flowfield: (optional)
        object of ``pykitPIV.flowfield.FlowField`` class specifying the user-uploaded flow field
        for the entire virtual wind tunnel.
        If not specified, the user has to provide a flow field specification
        through the ``flowfield_spec`` parameter to create a synthetic **pykitPIV**-generated flow field.
        **Future functionality will include temporally-evolving flow fields.**
    :param inference_model: (optional)
        object of a custom, user-defined class specifying the inference model for predicting flow targets from
        PIV images. It can be a CNN-based or a WIDIM-based model.
        If set to ``None``, PIV images are not recorded and inference from them is not done,
        instead the true flow target within the interrogation window is
        returned. The inference model has to have a method ``inference_model.inference()``
        that returns the predicted flow targets tensor of size :math:`(1, 2, H_{\\text{i}}+2b, W_{\\text{i}}+2b)`.
        The ``inference_model.inference()`` method must be able to take as an input the raw PIV images and
        pre-process them as needed.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``.
        If specified, all operations are reproducible.

    **Attributes:**

    - **flowfield** - (read-only) as per user input.
    - **flowfield_type** - (read-only) as per user input.
    - **flowfield_size** - (read-only) as per user input.
    - **time_separation** - (can be re-set) as per user input.
    - **admissible_observation_space** - (read-only) ``tuple`` specifying the admissible observation space along
      height and width.
    - **camera_position** - (read-only) ``tuple`` specifying the camera position.
    - **action_to_direction** - (read-only) ``dict`` specifying how action numbers translate to directions in
      the 2D virtual environment.
    - **action_to_verbose_direction** - (read-only) ``dict`` specifying how action numbers translate to verbose
      direction strings in the 2D virtual environment.
    - **n_actions** - (read-only) ``int`` specifying the total number of actions possible in this environment.
    - **prediction_tensor** - (read-only) ``numpy.ndarray`` specifying the predicted displacement field tensor.
    - **targets_tensor** - (read-only) ``numpy.ndarray`` specifying the ground truth displacement field tensor.
    """

    def __init__(self,
                 interrogation_window_size,
                 interrogation_window_size_buffer,
                 cues_function,
                 particle_spec=None,
                 motion_spec=None,
                 image_spec=None,
                 flowfield_spec=None,
                 user_flowfield=None,
                 inference_model=None,
                 random_seed=None):

        super().__init__()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        check_two_element_tuple(interrogation_window_size, 'interrogation_window_size')

        if not isinstance(interrogation_window_size_buffer, int):
            raise ValueError("Parameter `interrogation_window_size_buffer` has to be of type 'int'.")

        if interrogation_window_size_buffer < 0:
            raise ValueError("Parameter `interrogation_window_size_buffer` has to non-negative.")

        if not callable(cues_function):
            raise ValueError("Parameter `cues_function` has to be a callable.")

        if particle_spec is not None:
            if isinstance(particle_spec, dict):
                particle_spec = ParticleSpecs(**particle_spec)
            elif not isinstance(particle_spec, ParticleSpecs):
                raise TypeError("Particle specifications have to be of type 'dict' or 'pykitPIV.particle.ParticleSpecs'.")

        if motion_spec is not None:
            if isinstance(motion_spec, dict):
                motion_spec = MotionSpecs(**motion_spec)
            elif not isinstance(motion_spec, MotionSpecs):
                raise TypeError("Motion specifications have to be of type 'dict' or 'pykitPIV.motion.MotionSpecs'.")

        if image_spec is not None:
            if isinstance(image_spec, dict):
                image_spec = ImageSpecs(**image_spec)
            elif not isinstance(image_spec, ImageSpecs):
                raise TypeError("Image specifications have to be of type 'dict' or 'pykitPIV.image.ImageSpecs'.")

        if (flowfield_spec is None) and (user_flowfield is None):
            raise ValueError('At least one has to be specified: `user_flowfield` or `flowfield_spec`.')

        if (flowfield_spec is not None) and (user_flowfield is not None):
            raise ValueError('Only one has to be specified: `user_flowfield` or `flowfield_spec`.')

        if flowfield_spec is not None:
            if isinstance(flowfield_spec, dict):
                flowfield_spec = FlowFieldSpecs(**flowfield_spec)
            elif not isinstance(flowfield_spec, FlowFieldSpecs):
                raise TypeError("Flow field specifications have to be of type 'dict' or 'pykitPIV.flowfield.FlowFieldSpecs'.")

        if user_flowfield is not None:
            if not isinstance(user_flowfield, FlowField):
                raise ValueError("Parameter `user_flowfield` has to be of type 'pykitPIV.flowfield.FlowField'.")

        if random_seed is not None:
            if type(random_seed) != int:
                raise ValueError("Parameter `random_seed` has to be of type 'int'.")
            else:
                np.random.seed(seed=random_seed)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:

        # Size of the interrogation window:
        self.__interrogation_window_size = interrogation_window_size

        # Size of the buffer for the interrogation window:
        self.__interrogation_window_size_buffer = interrogation_window_size_buffer

        # Function that will compute cues for the RL agent based on the reconstructed displacement field:
        self.cues_function = cues_function

        # Specifications for Particle class:
        self.__particle_spec = particle_spec

        # Specifications for Motion class:
        self.__motion_spec = motion_spec

        # Specifications for Image class:
        self.__image_spec = image_spec

        # User flowfield:
        self.__user_flowfield = user_flowfield

        # Inference model that is capable of predicting the displacement fields from the PIV images.
        # This can be a CNN-based or WIDIM-based model.
        self.__inference_model = inference_model

        self.__random_seed = random_seed

        # Compute the total size of the interrogation window:
        self.__interrogation_window_size_with_buffer = (self.__interrogation_window_size[0] + 2 * self.__interrogation_window_size_buffer,
                                                        self.__interrogation_window_size[1] + 2 * self.__interrogation_window_size_buffer)

        # If the user did not supply their own flow field, a pykitPIV-generated flow field is used:
        if user_flowfield is None:

            # Specifications for Flowfield class:
            self.__flowfield_spec = flowfield_spec

            # Size of the entire wind tunnel flow field:
            self.__flowfield_size = self.__flowfield_spec.size

            # Type of the entire wind tunnel flow field:
            self.__flowfield_type = self.__flowfield_spec.flowfield_type

            # Generate the flow field that is fixed throughout training:
            flowfield = FlowField(n_images=1,
                                  size=self.__flowfield_size,
                                  size_buffer=0,
                                  time_separation=self.__flowfield_spec.time_separation,
                                  random_seed=self.__random_seed)

            if self.__flowfield_type == 'random smooth':

                flowfield.generate_random_velocity_field(gaussian_filters=self.__flowfield_spec.gaussian_filters,
                                                         n_gaussian_filter_iter=self.__flowfield_spec.n_gaussian_filter_iter,
                                                         displacement=self.__flowfield_spec.displacement)

            elif self.__flowfield_type == 'radial':

                flowfield.generate_radial_velocity_field(source=self.__flowfield_spec.radial_source,
                                                         displacement=self.__flowfield_spec.displacement,
                                                         imposed_source_location=self.__flowfield_spec.radial_imposed_source_location,
                                                         sigma=self.__flowfield_spec.radial_sigma,
                                                         epsilon=self.__flowfield_spec.radial_epsilon)

            elif self.__flowfield_type == 'constant':

                flowfield.generate_constant_velocity_field(u_magnitude=self.__flowfield_spec.constant_u_magnitude,
                                                           v_magnitude=self.__flowfield_spec.constant_v_magnitude)

            if self.__flowfield_spec.apply_SLM:

                # Solve the simplified Langevin model (SLM) for the mean velocity fields:
                flowfield.generate_langevin_velocity_field(mean_field=flowfield.velocity_field,
                                                           integral_time_scale=self.__flowfield_spec.integral_time_scale,
                                                           sigma=self.__flowfield_spec.sigma,
                                                           n_stochastic_particles=self.__flowfield_spec.n_stochastic_particles,
                                                           n_iterations=self.__flowfield_spec.n_iterations,
                                                           verbose=False)

            # This is an object of the pykitPIV.flowfield.FlowField class:
            self.__flowfield = flowfield

            # Set the initial time separation to the value coming from the FlowFieldSpec.
            # This value can be overwritten later during RL training.
            self.__time_separation = self.__flowfield_spec.time_separation

        # Otherwise, use the flow field provided by the user:
        else:

            # This too is an object of the pykitPIV.flowfield.FlowField class:
            self.__flowfield = user_flowfield

            self.__flowfield_type = 'user'

            self.__flowfield_size = user_flowfield.velocity_field_magnitude.shape[2::]

            # Set the initial time separation to the value coming from the FlowFieldSpec.
            # This value can be overwritten later during RL training.
            self.__time_separation = user_flowfield.time_separation

        # Observation space for the RL agent: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Calculate the admissible observation space from which the agent can sample the interrogation windows
        self.__admissible_observation_space = (self.__flowfield_size[0] - self.__interrogation_window_size_with_buffer[0],
                                               self.__flowfield_size[1] - self.__interrogation_window_size_with_buffer[1])

        # The observation space is the camera's location in the virtual environement. In practice, this is
        # the position of the camera looking at a specific interrogation window.
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]),
                                                high=np.array([self.__admissible_observation_space[0],
                                                               self.__admissible_observation_space[1]]),
                                                dtype=int)

        # Action space for the RL agent: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Actions are the camera's movement on the pixel grid:
        self.action_space = gym.spaces.Discrete(5)

        # Dictionary that maps the abstract actions to the directions on the pixel grid:
        self.__action_to_direction = {
            0: np.array([1, 0]),  # up
            1: np.array([0, 1]),  # right
            2: np.array([-1, 0]),  # down
            3: np.array([0, -1]),  # left
            4: np.array([0, 0]),  # stay
        }

        self.__action_to_verbose_direction = {
            0: 'Up',
            1: 'Right',
            2: 'Down',
            3: 'Left',
            4: 'Stay',
        }

        self.__n_actions = 5

        # Initial camera position is undetermined. The user will have to call reset() to initialize camera position:
        self.__camera_position = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Properties coming from user inputs:
    @property
    def flowfield(self):
        return self.__flowfield

    @property
    def flowfield_type(self):
        return self.__flowfield_type

    @property
    def flowfield_size(self):
        return self.__flowfield_size

    # ^ For now, the flow field in the wind tunnel cannot be overwritten during RL training.
    # This may be modified in the future.

    @property
    def time_separation(self):
        return self.__time_separation

    # Properties computed at class init:
    @property
    def admissible_observation_space(self):
        return self.__admissible_observation_space

    @property
    def camera_position(self):
        return self.__camera_position

    @property
    def action_to_direction(self):
        return self.__action_to_direction

    @property
    def action_to_verbose_direction(self):
        return self.__action_to_verbose_direction

    @property
    def n_actions(self):
        return self.__n_actions

    @property
    def prediction_tensor(self):
        return self.__prediction_tensor

    @property
    def targets_tensor(self):
        return self.__targets_tensor

    # Setters:
    @time_separation.setter
    def time_separation(self, new_time_separation):
        if (not isinstance(new_time_separation, float)) and (not isinstance(new_time_separation, int)):
            raise ValueError("Parameter `time_separation` has to be of type `float` or `int`.")
        else:
            if new_time_separation <= 0:
                raise ValueError("Parameter `time_separation` has to be a non-zero, positive number.")
            else:
                self.__time_separation = new_time_separation
                self.__flowfield.time_separation = new_time_separation

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
            - **image** - ``pykitPIV.image.Image`` object specifying the generated PIV image pairs.
        """

        # Extract the velocity field under the current interrogation window (with the buffer!):
        h_start = camera_position[0]
        h_stop = h_start + self.__interrogation_window_size_with_buffer[0]

        w_start = camera_position[1]
        w_stop = w_start + self.__interrogation_window_size_with_buffer[1]

        velocity_field_at_interrogation_window = self.flowfield.velocity_field[:, :, h_start:h_stop, w_start:w_stop]

        # Construct virtual PIV measurements: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Initialize a particle object:
        particles = Particle(n_images=1,
                             size=self.__interrogation_window_size,
                             size_buffer=self.__interrogation_window_size_buffer,
                             diameters=self.__particle_spec.diameters,
                             distances=self.__particle_spec.distances,
                             densities=self.__particle_spec.densities,
                             diameter_std=self.__particle_spec.diameter_std,
                             seeding_mode=self.__particle_spec.seeding_mode,
                             random_seed=self.__particle_spec.random_seed)

        # Initialize a flow field object corresponding to just the flow field within the interrogation window:
        flowfield = FlowField(n_images=1,
                              size=self.__interrogation_window_size,
                              size_buffer=self.__interrogation_window_size_buffer,
                              time_separation=self.time_separation,
                              random_seed=None)

        # ^ The time separation is now taken from the class attribute, which can mean that this value has changed.

        flowfield.upload_velocity_field(velocity_field_at_interrogation_window)

        # Initialize a motion object:
        motion = Motion(particles,
                        flowfield,
                        particle_loss=self.__motion_spec.particle_loss,
                        particle_gain=self.__motion_spec.particle_gain,
                        random_seed=self.__motion_spec.random_seed)

        motion.runge_kutta_4th(n_steps=self.__motion_spec.n_steps)

        # Initialize an image object:
        image = Image(random_seed=self.__image_spec.random_seed)
        image.add_particles(particles)
        image.add_flowfield(flowfield)
        image.add_motion(motion)

        image.add_reflected_light(exposures=self.__image_spec.exposures,
                                  maximum_intensity=self.__image_spec.maximum_intensity,
                                  laser_beam_thickness=self.__image_spec.laser_beam_thickness,
                                  laser_over_exposure=self.__image_spec.laser_over_exposure,
                                  laser_beam_shape=self.__image_spec.laser_beam_shape,
                                  alpha=self.__image_spec.alpha,
                                  clip_intensities=self.__image_spec.clip_intensities,
                                  normalize_intensities=self.__image_spec.normalize_intensities)

        return image

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def make_inference(self, image_obj):
        """
        Makes inference of the displacement field under the interrogation window based on the recorded PIV images.

        **Example:**

        .. code:: python

            # Once the environment has been initialized:
            env = PIVEnv(...)

            # And once the PIV images have been recorded:
            camera_position = env.observation_space.sample()
            image_obj = env.record_particles(camera_position)

            # We can now make inference and predict the displacement field:
            targets_tensor, prediction_tensor = env.make_inference(image_obj)

        :param image_obj:
            object of ``pykitPIV.image.Image`` class specifying the generated PIV image pairs.

        :return:
            - **targets_tensor** - ``numpy.ndarray`` specifying the tensor of true flow targets.
            - **prediction_tensor** - ``numpy.ndarray`` specifying the tensor of predicted flow targets.
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

        return targets_tensor, prediction_tensor

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def reset(self,
              imposed_camera_position=None,
              regenerate_flowfield=False):
        """
        Resets the environment to a random or user-imposed initial state and, optionally, also
        re-generates the flow field.

        **Example:**

        .. code:: python

            import numpy as np

            # Once the environment has been initialized:
            env = PIVEnv(...)

            # We reset the environment to create its random initial state:
            initial_camera_position, cues = env.reset()

            # Optionally, we can impose an initial state:
            initial_camera_position, cues = env.reset(imposed_camera_position=np.array([10, 50]))

        :param imposed_camera_position: (optional)
            ``numpy.ndarray`` of two elements specifying the initial camera position in pixels :math:`[\\text{px}]`.
            This defines the bottom-left corner of the interrogation window.
            Example can be ``numpy.array([10, 50])``
            which positions the camera at the location :math:`10 \\text{px}` along the height dimension
            and :math:`50 \\text{px}` along the width dimension.
            If not specified, a random camera position is selected through ``env.observation_space.sample()``.
        :param regenerate_flowfield: (optional)
            ``bool`` specifying whether the larger flow field should be regenerated at reset.

        :return:
            - **camera_position** - ``numpy.ndarray`` of two elements specifying the initial camera position in
              pixels :math:`[\\text{px}]`.
            - **cues** - ``numpy.ndarray`` specifying the initial cues that the RL agent will later be sensing.
        """

        # Re-generate the flow field if needed:
        if regenerate_flowfield:

            if self.__user_flowfield is None:

                flowfield = FlowField(n_images=1,
                                      size=self.__flowfield_size,
                                      size_buffer=0,
                                      time_separation=self.time_separation,
                                      random_seed=self.__random_seed)

                if self.__flowfield_type == 'random smooth':
                    flowfield.generate_random_velocity_field(gaussian_filters=self.__flowfield_spec.gaussian_filters,
                                                             n_gaussian_filter_iter=self.__flowfield_spec.n_gaussian_filter_iter,
                                                             displacement=self.__flowfield_spec.displacement)

                elif self.__flowfield_type == 'radial':

                    flowfield.generate_radial_velocity_field(source=self.__flowfield_spec.radial_source,
                                                             displacement=self.__flowfield_spec.displacement,
                                                             imposed_source_location=self.__flowfield_spec.radial_imposed_source_location,
                                                             sigma=self.__flowfield_spec.radial_sigma,
                                                             epsilon=self.__flowfield_spec.radial_epsilon)

                if self.__flowfield_spec.apply_SLM:
                    # Solve the simplified Langevin model (SLM) for the mean velocity fields:
                    flowfield.generate_langevin_velocity_field(mean_field=flowfield.velocity_field,
                                                               integral_time_scale=self.__flowfield_spec.integral_time_scale,
                                                               sigma=self.__flowfield_spec.sigma,
                                                               n_stochastic_particles=self.__flowfield_spec.n_stochastic_particles,
                                                               n_iterations=self.__flowfield_spec.n_iterations,
                                                               verbose=False)

                # This is an object of the pykitPIV.flowfield.FlowField class:
                self.__flowfield = flowfield

            else:
                raise TypeError("Cannot re-generate flow field from user-specified field!")

        # Check whether the user-specified camera position is possible given the admissible observation space
        # and the size of the interrogation window:
        if imposed_camera_position is not None:
            if (imposed_camera_position[0] < 0) or (imposed_camera_position[0] > self.__admissible_observation_space[0]):
                raise ValueError("The user-imposed camera position falls outside of the admissible virtual environment!")
            if imposed_camera_position[1] < 0 or imposed_camera_position[1] > self.__admissible_observation_space[1]:
                raise ValueError("The user-imposed camera position falls outside of the admissible virtual environment!")

        if imposed_camera_position is None:
            # Create an initial camera position:
            camera_position = self.observation_space.sample()
            self.__camera_position = camera_position
        else:
            camera_position = imposed_camera_position
            self.__camera_position = camera_position

        # Record PIV images at that camera position, but only if the inference model is given:
        if self.__inference_model is not None:

            image_obj = self.record_particles(camera_position)
            self.__image_obj = image_obj

            # Make inference of displacement field based on the recorded PIV images:
            targets_tensor, prediction_tensor = self.make_inference(image_obj)

        # If the inference model is not specified, just return the ground truth velocity field:
        else:

            # Extract the velocity field under the current interrogation window:
            h_start = self.__camera_position[0] + self.__interrogation_window_size_buffer
            h_stop = h_start + self.__interrogation_window_size[0]

            w_start = self.__camera_position[1] + self.__interrogation_window_size_buffer
            w_stop = w_start + self.__interrogation_window_size[1]

            # ^ Note that we extract the interrogation window size without buffer because PIV images are not being
            # recorded.

            self.flowfield.compute_displacement_field()

            targets_tensor = self.flowfield.displacement_field[:, :, h_start:h_stop, w_start:w_stop]
            prediction_tensor = self.flowfield.displacement_field[:, :, h_start:h_stop, w_start:w_stop]

        # Save the prediction tensor and the targets tensor as globally-available variables:
        self.__prediction_tensor = prediction_tensor
        self.__targets_tensor = targets_tensor

        # Compute the cues based on the current prediction tensor:
        cues = self.cues_function(prediction_tensor)

        # Find out how many cues the RL agent will be looking at, this is useful information for later:
        (_, self.n_cues) = np.shape(cues)

        return camera_position, cues

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def step(self,
             action,
             reward_function,
             reward_transformation,
             magnify_step=1,
             verbose=False):
        """
        Makes one step in the environment which moves the camera to a new position, and computes the associated reward
        for taking that step. The reward is computed based on the PIV images seen at that position, converted
        to a continuous displacement field recovered by an inference model (either CNN-based or WIDIM-based).

        **Example:**

        .. code:: python

            from pykitPIV.ml import Rewards

            # Once the environment has been initialized:
            env = PIVEnv(...)

            # We reset the environment to create its initial state:
            initial_camera_position, initial_cues = env.reset()

            # We create a reward function by polling from one of the pykitPIV.Rewards methods.
            # Here, we use the reward based on the Q-criterion:
            rewards = Rewards(verbose=True)
            reward_function = rewards.q_criterion

            # We also define a function that will transform and reduce the Q-criterion
            # to provide one reward value:
            def reward_transformation(Q):
                Q = np.max(Q.clip(min=0))
                return Q

            # Now we can take a step in the environment by selecting one of the five actions:
            camera_position, cues, reward = env.step(action=4,
                                                     reward_function=reward_function,
                                                     reward_transformation=reward_transformation,
                                                     verbose=True)

        :param action:
            ``int`` specifying the action to be taken at the current step in the environment.
        :param reward_function:
            ``function`` specifying the dynamics of the reward construction as a function of predicted and/or true
            flow targets. It can be one of the rewards functions from the ``pykitPIV.Rewards`` class.
        :param reward_transformation:
            ``function`` specifying an arbitrary transformation of the reward function
            and an arbitrary compression of the reward function to a single value.
        :param magnify_step: (optional)
            ``int`` specifying the factor that multiplies each unit step in the environment.
        :param verbose: (optional)
            ``bool`` specifying if the verbose print statements should be displayed.

        :return:
            - **camera_position** - ``numpy.ndarray`` of two elements specifying the new camera position
              in pixels :math:`[\\text{px}]` after taking this step.
            - **cues** - ``numpy.ndarray`` specifying the cues that the RL agent senses after taking this step.
            - **reward** - ``float`` specifying the reward received after taking this step.
        """

        # Map the action (element of {0,1,2,3,4}) to the new camera position:
        direction = self.action_to_direction[action]

        if verbose:
            print('Action ' + str(action) + ': ' + self.action_to_verbose_direction[action])

        # Take the step in the environment:
        # (We clip the camera position to make sure that we don't leave the grid bounds)
        camera_position = np.array([np.clip(self.__camera_position[0] + magnify_step * direction[0], 0, self.__admissible_observation_space[0]),
                                    np.clip(self.__camera_position[1] + magnify_step * direction[1], 0, self.__admissible_observation_space[1])])

        # Reset the camera position:
        self.__camera_position = camera_position

        if verbose:
            print('Camera position:')
            print(camera_position)

        # Record PIV images at that camera position, but only if the inference model is given:
        if self.__inference_model is not None:

            image_obj = self.record_particles(camera_position)
            self.__image_obj = image_obj

            # Make inference of displacement field based on the recorded PIV images:
            targets_tensor, prediction_tensor = self.make_inference(image_obj)

        # If the inference model is not specified, just return the ground truth velocity field, bypassing the PIV:
        else:

            # Extract the velocity field under the current interrogation window:
            h_start = self.__camera_position[0] + self.__interrogation_window_size_buffer
            h_stop = h_start + self.__interrogation_window_size[0]

            w_start = self.__camera_position[1] + self.__interrogation_window_size_buffer
            w_stop = w_start + self.__interrogation_window_size[1]

            # ^ Note that we extract the interrogation window size without buffer because PIV images are not being
            # recorded.

            self.flowfield.compute_displacement_field()

            targets_tensor = self.flowfield.displacement_field[:, :, h_start:h_stop, w_start:w_stop]
            prediction_tensor = self.flowfield.displacement_field[:, :, h_start:h_stop, w_start:w_stop]

        # Save the prediction tensor and the targets tensor as globally-available variables:
        self.__prediction_tensor = prediction_tensor
        self.__targets_tensor = targets_tensor

        # Reward construction:
        reward = reward_function(vector_field=prediction_tensor,
                                 transformation=reward_transformation)

        # Compute the cues based on the current prediction tensor:
        cues = self.cues_function(prediction_tensor)

        if verbose:
            print('Cues:')
            print(cues)
            print('Reward:')
            print(reward)

        return camera_position, cues, reward

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def render(self,
               quantity=None,
               camera_position=None,
               wind_tunnel_only=False,
               c='white',
               s=10,
               lw=2,
               xlabel=None,
               ylabel=None,
               xticks=True,
               yticks=True,
               cmap='viridis',
               normalize_cbars=False,
               add_quiver=False,
               quiver_step=10,
               quiver_color='k',
               add_streamplot=False,
               streamplot_density=1,
               streamplot_color='k',
               streamplot_linewidth=1,
               figsize=(10, 5),
               dpi=300,
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
            env.render(quantity,
                       camera_position,
                       wind_tunnel_only=False,
                       c='white',
                       s=20,
                       lw=1,
                       xlabel=None,
                       ylabel=None,
                       xticks=True,
                       yticks=True,
                       cmap='viridis',
                       normalize_cbars=False,
                       add_quiver=False,
                       quiver_step=10,
                       quiver_color='k',
                       add_streamplot=False,
                       streamplot_density=1,
                       streamplot_color='k',
                       streamplot_linewidth=1,
                       figsize=(10, 5),
                       dpi=300,
                       filename=None)

        One example rendering of the virtual wind tunnel can look like this:

        .. image:: ../images/ml_PIVEnv_render.png
            :width: 800

        :param quantity: (optional)
            ``numpy.ndarray`` specifying the quantity to plot within the virtual wind tunnel section.
            It should have size :math:`(H_{\\text{wt}}, W_{\\text{wt}})`.
            If set to ``None``, displacement field magnitude, :math:`|d\\vec{\\mathbf{s}}|`, is plotted.
        :param camera_position: (optional)
            ``numpy.ndarray`` specifying the camera position in pixels :math:`[\\text{px}]`.
            This defines the bottom-left corner of the interrogation window.
            Example can be ``numpy.array([10,50])`` which positions camera at the location :math:`10 \\text{px}`
            along the height dimension and :math:`50 \\text{px}` along the width dimension.
        :param wind_tunnel_only: (optional)
            ``bool`` specifying if only the wind tunnel should be plotted.
        :param c: (optional)
            ``str`` specifying the color for the interrogation window outline.
        :param s: (optional)
            ``int`` or ``float`` specifying the size of the dot that represents the camera position.
        :param lw: (optional)
            ``int``  or ``float`` specifying the line width for the interrogation window outline.
        :param xlabel: (optional)
            ``str`` specifying :math:`x`-label.
        :param ylabel: (optional)
            ``str`` specifying :math:`y`-label.
        :param xticks: (optional)
            ``bool`` specifying if ticks along the :math:`x`-axis should be plotted.
        :param yticks: (optional)
            ``bool`` specifying if ticks along the :math:`y`-axis should be plotted.
        :param cmap: (optional)
            ``str`` or an object of `matplotlib.colors.ListedColormap <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html>`_ specifying the color map to use.
        :param normalize_cbars: (optional)
            ``bool`` specifying if the colorbar for the interrogation window should be normalized to the colorbar for
            the entire wind tunnel.
        :param add_quiver: (optional)
            ``bool`` specifying if vector field should be plotted on top of the scalar magnitude field.
        :param quiver_step: (optional)
            ``int`` specifying the step on the pixel grid to attach a vector to. The higher this number is, the less dense the vector field is.
        :param quiver_color: (optional)
            ``str`` specifying the color of velocity vectors.
        :param add_streamplot: (optional)
            ``bool`` specifying if streamlines should be plotted on top of the scalar magnitude field.
        :param streamplot_density: (optional)
            ``float`` or ``int`` specifying the streamplot density.
        :param streamplot_color: (optional)
            ``str`` specifying the streamlines color.
        :param streamplot_linewidth: (optional)
            ``int`` or ``float`` specifying the line width for the streamplot.
        :param figsize: (optional)
            ``tuple`` of two numerical elements specifying the figure size as per ``matplotlib.pyplot``.
        :param dpi: (optional)
            ``int`` specifying the dpi for the image.
        :param filename: (optional)
            ``str`` specifying the path and filename to save an image. If set to ``None``, the image will not be saved.

        :return:
            - **plt** - ``matplotlib.pyplot`` image handle.
        """

        fontsize = 14

        figure = plt.figure(figsize=figsize)

        # Compute the displacement field:
        self.flowfield.compute_displacement_field()

        if wind_tunnel_only:

            spec = figure.add_gridspec(ncols=1,
                                       nrows=1,
                                       width_ratios=[1],
                                       height_ratios=[1])

            figure.add_subplot(spec[0, 0])

        else:

            spec = figure.add_gridspec(ncols=9,
                                       nrows=3,
                                       width_ratios=[1, 0.2, 1, 0.2, 1, 0.2, 1, 0.2, 1],
                                       height_ratios=[2, 0.2, 1])

            figure.add_subplot(spec[0, 0:9])

        # Visualize the user-provided quantity or the displacement field magnitude:
        if quantity is not None:
            ims = plt.imshow(quantity, cmap=cmap, origin='lower', zorder=0)
        else:
            ims = plt.imshow(self.flowfield.displacement_field_magnitude[0,0,:,:], cmap=cmap, origin='lower', zorder=0)

        plt.colorbar(ims)

        if add_streamplot:

            X = np.arange(0, self.flowfield.size[1], 1)
            Y = np.arange(0, self.flowfield.size[0], 1)

            plt.streamplot(X, Y,
                           self.flowfield.displacement_field[0, 0, :, :],
                           self.flowfield.displacement_field[0, 1, :, :],
                           density=streamplot_density,
                           color=streamplot_color,
                           linewidth=streamplot_linewidth,
                           zorder=1)

        if add_quiver:

            X = np.arange(0, self.flowfield.size[1], quiver_step)
            Y = np.arange(0, self.flowfield.size[0], quiver_step)

            plt.quiver(X, Y,
                       self.flowfield.displacement_field[0, 0, ::quiver_step, ::quiver_step],
                       self.flowfield.displacement_field[0, 1, ::quiver_step, ::quiver_step],
                       color=quiver_color)

        plt.title('Virtual wind tunnel', fontsize=fontsize)

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if not xticks:
            plt.xticks([])

        if not yticks:
            plt.yticks([])

        # Visualize the camera position:
        if camera_position is not None:

            plt.scatter(camera_position[1]-0.5, camera_position[0]-0.5, c=c, s=s, zorder=2)

            # Visualize a rectangle that defines the current interrogation window:
            rect = patches.Rectangle((camera_position[1]-0.5, camera_position[0]-0.5),
                                     self.__interrogation_window_size_with_buffer[1],
                                     self.__interrogation_window_size_with_buffer[0],
                                     linewidth=lw, edgecolor=c, facecolor='none', zorder=2)
            ax = plt.gca()
            ax.add_patch(rect)

            # Visualize a rectangle that defines the current interrogation window without buffer:
            rect = patches.Rectangle((camera_position[1]-0.5+self.__interrogation_window_size_buffer, camera_position[0]-0.5+self.__interrogation_window_size_buffer),
                                     self.__interrogation_window_size[1],
                                     self.__interrogation_window_size[0],
                                     linewidth=lw/2, edgecolor=c, facecolor='none', zorder=2)
            ax = plt.gca()
            ax.add_patch(rect)

        if not wind_tunnel_only:

            if normalize_cbars:
                vmin = np.min(self.flowfield.displacement_field_magnitude[0,0,:,:])
                vmax = np.max(self.flowfield.displacement_field_magnitude[0,0,:,:])
            else:
                vmin = None
                vmax = None

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

            if self.__inference_model is not None:

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
            normalizer = np.mean(targets_magnitude)
            error_map = prediction_error/normalizer*100
            ims = plt.imshow(error_map, origin='lower', cmap='Greys')
            plt.colorbar(ims)
            plt.title('Error %: ' + str(round(np.mean(error_map), 1)), fontsize=fontsize)

        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')

        return plt

########################################################################################################################
########################################################################################################################
####
####    Class: CameraAgentSingleDQN
####
########################################################################################################################
########################################################################################################################

class CameraAgentSingleDQN:
    """
    Creates a reinforcement learning (RL) agent that operates a virtual camera in a PIV experimental setting
    and provides a training loop for Q-learning. Here, we use Q-learning with a single deep neural network (DNN) model.

    The goal of the RL agent is to learn the mapping function, :math:`f`,
    from :math:`\\text{cues} \\rightarrow \\text{actions}`:

    .. math::

        \\text{action} = f(\\text{cues} )

    where :math:`f` is the trained DNN model.

    **Example:**

    .. code:: python

        from pykitPIV.ml import PIVEnv
        from pykitPIV.ml import CameraAgentSingleDQN
        import tensorflow as tf

        # Initialize the environment:
        env = PIVEnv(...)

        # Example single deep Q-network model:
        class QNetwork(tf.keras.Model):

            def __init__(self, n_actions, kernel_initializer):

                super(QNetwork, self).__init__()

                self.dense1 = tf.keras.layers.Dense(env.n_cues, activation='relu', kernel_initializer=kernel_initializer)
                self.dense2 = tf.keras.layers.Dense(size_of_hidden_unit, activation='relu', kernel_initializer=kernel_initializer)
                self.dense3 = tf.keras.layers.Dense(size_of_hidden_unit, activation='relu', kernel_initializer=kernel_initializer)
                self.output_layer = tf.keras.layers.Dense(n_actions, activation='relu', kernel_initializer=kernel_initializer)

            def call(self, state):

                x = self.dense1(state)
                x = self.dense2(x)
                x = self.dense3(x)

                return self.output_layer(x)

        # Initialize the camera agent for single deep Q-network training:
        ca = CameraAgentSingleDQN(env=env,
                                  q_network=QNetwork(env.n_actions, tf.keras.initializers.RandomUniform),
                                  learning_rate=0.0001,
                                  optimizer='Adam',
                                  discount_factor=0.95)

    :param env:
        object of a custom environment class that is a subclass of ``gym.Env`` specifying the virtual environment.
        This can for instance be an object of the ``pykitPIV.ml.PIVEnv`` class.
    :param q_network:
        ``tf.keras.Model`` specifying the single deep neural network that will be trained for Q-learning.
    :param learning_rate:  (optional)
        ``float`` specifying the initial learning rate. The learning rate can be updated on the fly by passing a new
        value to the ``train()`` function.
    :param optimizer:  (optional)
        ``str`` specifying the gradient descent optimizer to use. It can be ``'Adam'`` or ``'RMSprop'``.
    :param discount_factor:  (optional)
        ``float`` specifying the discount factor, :math:`\gamma`.
    """

    def __init__(self,
                 env,
                 q_network,
                 learning_rate=0.0001,
                 optimizer='RMSprop',
                 discount_factor=0.95):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:

        # Define the environment for the camera agent:
        self.env = env
        self.n_actions = self.env.n_actions

        print('The uploaded environment has ' + str(self.n_actions) + ' actions.')

        # We define the single deep Q-network:
        self.q_network = q_network

        # Parameters of training the DNN:
        self.learning_rate = learning_rate
        if optimizer == 'RMSprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif optimizer == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            raise ValueError("Optimizer has to be 'RMSprop' or 'Adam'. Other optimizers will be implemented in future versions.")

        self.q_network.compile(self.optimizer, loss=tf.keras.losses.MeanSquaredError())
        self.MSE_losses = []

        # Reinforcement learning parameters:
        self.discount_factor = discount_factor

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def choose_action(self,
                      cues,
                      epsilon):
        """
        Defines an :math:`\\varepsilon`-greedy choice of the next best action to select.

        If the probability is less than :math:`\\varepsilon`, action is selected at random.

        If the probability is higher than or equal to :math:`\\varepsilon`, action is selected based on the cues that
        are the current characteristics of the flow field inside the current interrogation window at location defined by
        the camera position (``camera_position``).

        **Example:**

        .. code:: python

            # Once the camera agent has been initialized:
            ca = CameraAgentSingleDQN(...)

            # And we have the set of cues computed, e.g., from the initial interrogation window:
            camera_position, cues = ca.env.reset()

            # We can use the epsilon-greedy strategy to choose the action:
            ca.choose_action(cues=cues,
                             epsilon=0.5)

        :param cues:
            ``numpy.ndarray`` specifying :math:`N` cues that the RL agent senses. The cues are the input parameters
            to the Q-network. It has to have size ``(1, N)``.
        :param epsilon:
            ``float`` specifying the exploration probability, :math:`\\varepsilon`. It has to be between 0 and 1.

        :return:
            - **action** - ``int`` specifying the action selected.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:
        if not isinstance(epsilon, float):
            raise ValueError("Parameter `epsilon` has to be of type 'float'.")

        if epsilon < 0 or epsilon > 1:
            raise ValueError("Parameter `epsilon` has to be between 0 and 1.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Select random action with probability epsilon:
        if np.random.rand() < epsilon:

            return np.random.choice(self.n_actions)

        # Select the currently best action with probability (1 - epsilon):
        else:

            q_values = self.q_network(cues, training=False)

            return np.argmax(q_values)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def train(self,
              cues,
              action,
              reward,
              next_cues,
              new_learning_rate=0.001):
        """
        Trains the deep Q-network (``q_network``) with the outcome of a single step in the environment.

        .. note::

            This function uses `tf.GradientTape() <https://www.tensorflow.org/api_docs/python/tf/GradientTape>`_
            to record operations done on Q-network's trainable parameters
            and to apply gradients.

        **Example:**

        .. code:: python

            # Once the camera agent has been initialized:
            ca = CameraAgentSingleDQN(...)

            # And we have access to cues, action, reward, and next_cues
            # coming from the current step in the environment:
            cues, action, reward, next_cues = ...

            # We can train the agent with information about the current step in the environment:
            ca.train(cues=cues,
                     action=action,
                     reward=reward,
                     next_cues=next_cues,
                     new_learning_rate=0.001)

        To save the trained Q-network, you can run:

        .. code:: python

            ca.q_network.save('SingleDQN.keras')

        :param cues:
            ``numpy.ndarray`` specifying the current cues.
        :param action:
            ``int`` specifying the current action selected.
        :param reward:
            ``float`` specifying the current reward received.
        :param next_cues:
            ``numpy.ndarray`` specifying the next cues seen after taking the current action in the environment.
        :param new_learning_rate: (optional)
            ``float`` specifying the new learning rate to use in the current pass over the minibatch.
        """

        self.optimizer.learning_rate.assign(new_learning_rate)

        # Compute the target Q-value (that we should have):
        next_q_values = self.q_network(next_cues, training=False)
        max_next_q = tf.reduce_max(next_q_values, axis=1)[0]
        target_q_value = reward + self.discount_factor * max_next_q

        with tf.GradientTape() as tape:

            # Compute the Q-value that we actually have:
            current_q_values = self.q_network(cues, training=True)
            q_value_for_action = tf.gather(current_q_values[0], action)

            # Compute the loss based on the difference between what the Q-value should be and what it actually is:
            loss = tf.reduce_mean(tf.square(target_q_value - q_value_for_action))

        # Compute and apply the gradients to the trainable variables of the deep Q-network:
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # Append the losses:
        self.MSE_losses.append(loss)

    def view_weights(self):
        """
        Returns the latest weights and biases of the deep Q-network.

        **Example:**

        .. code:: python

            # Once the camera agent has been initialized:
            ca = CameraAgentSingleDQN(...)

            # View the current target Q-network weights:
            ca.view_weights()

        :return:
            - **weights_and_biases** - ``list`` of ``numpy.ndarray`` specifying the current trainable parameters
              (weights and biases) of the deep Q-network.
        """

        return self.q_network.get_weights()

########################################################################################################################
########################################################################################################################
####
####    Class: CameraAgentDoubleDQN
####
########################################################################################################################
########################################################################################################################

class CameraAgentDoubleDQN:
    """
    Creates a reinforcement learning (RL) agent that operates a virtual camera in a PIV experimental setting
    and provides a training loop for Q-learning. We use double Q-learning
    (see `Hasselt et al. <https://arxiv.org/abs/1509.06461>`_ for more info) and use
    two deep neural network (DNN) models with memory replay.
    Having two Q-networks separates the process of finding which action has the maximum Q-value from
    the process of learning precisely what that maximum Q-value should be.

    The goal of the RL agent is to learn the mapping function, :math:`f`,
    from :math:`\\text{cues} \\rightarrow \\text{actions}`:

    .. math::

        \\text{action} = f(\\text{cues} )

    where :math:`f` is the trained DNN model.

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

                self.dense1 = tf.keras.layers.Dense(10,
                                                    activation='linear',
                                                    kernel_initializer=tf.keras.initializers.Ones)
                self.dense2 = tf.keras.layers.Dense(10,
                                                    activation='linear',
                                                    kernel_initializer=tf.keras.initializers.Ones)
                self.output = tf.keras.layers.Dense(n_actions,
                                                    activation='linear',
                                                    kernel_initializer=tf.keras.initializers.Ones)

            def call(self, state):

                x = self.dense1(state)
                x = self.dense2(x)

                return self.output(x)

        # Initialize the camera agent:
        ca = CameraAgent(env=env,
                         target_q_network=QNetwork(env.n_actions),
                         online_q_network=QNetwork(env.n_actions),
                         memory_size=10000,
                         batch_size=256,
                         n_epochs=100,
                         learning_rate=0.0001,
                         optimizer='RMSprop',
                         discount_factor=0.95)

    :param env:
        object of a custom environment class that is a subclass of ``gym.Env`` specifying the virtual environment.
        This can for instance be an object of the ``pykitPIV.ml.PIVEnv`` class.
    :param target_q_network:
        ``tf.keras.Model`` specifying the deep neural network that will be the target (stable) network for Q-learning.
    :param online_q_network:
        ``tf.keras.Model`` specifying the deep neural network that will be the online (temporary) network for Q-learning.
    :param memory_size:  (optional)
        ``int`` specifying the size of the memory bank.
    :param batch_size:  (optional)
        ``int`` specifying the batch size for training the Q-network for after each step in the environment.
    :param n_epochs:  (optional)
        ``int`` specifying the number of epochs to train the Q-network for after each step in the environment.
    :param learning_rate:  (optional)
        ``float`` specifying the initial learning rate. The learning rate can be updated on the fly by passing a new
        value to the ``train()`` function.
    :param optimizer:  (optional)
        ``str`` specifying the gradient descent optimizer to use. It can be ``'Adam'`` or ``'RMSprop'``.
    :param discount_factor:  (optional)
        ``float`` specifying the discount factor, :math:`\gamma`.
    """

    def __init__(self,
                 env,
                 target_q_network,
                 online_q_network,
                 memory_size=10000,
                 batch_size=256,
                 n_epochs=100,
                 learning_rate=0.0001,
                 optimizer='RMSprop',
                 discount_factor=0.95):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:

        # Define the environment for the camera agent:
        self.env = env
        self.n_actions = self.env.n_actions

        print('The uploaded environment has ' + str(self.n_actions) + ' actions.')

        # We have two Q-networks, the target Q-network, which is the "stable" network:
        self.target_q_network = target_q_network

        # But the target network will be synchronized with the online Q-network only once every few episodes:
        self.online_q_network = online_q_network

        # ^ This prevents overshoots in learning the Q-value as shown by Hasselt et al. (2015)
        # These two Q-networks will separate the process of finding which action has the maximum Q-value from
        # the process of learning precisely what that maximum Q-value should be.

        # Parameters of training the DNN:
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        if optimizer == 'RMSprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif optimizer == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            raise ValueError("Optimizer has to be 'RMSprop' or 'Adam'. Other optimizers will be implemented in future versions.")

        self.target_q_network.compile(self.optimizer, loss=tf.keras.losses.MeanSquaredError())
        self.online_q_network.compile(self.optimizer, loss=tf.keras.losses.MeanSquaredError())
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
                      cues,
                      epsilon):
        """
        Defines an :math:`\\varepsilon`-greedy choice of the next best action to select.

        If the probability is less than :math:`\\varepsilon`, action is selected at random.

        If the probability is higher than or equal to :math:`\\varepsilon`, action is selected based on the cues that
        are the characteristic of the flow field inside the current interrogation window at location defined by
        the camera position (``camera_position``).

        **Example:**

        .. code:: python

            # Once the camera agent has been initialized:
            ca = CameraAgent(...)

            # And we have the set of cues computed, e.g., from the initial interrogation window:
            camera_position, cues = ca.env.reset()

            # We can use the epsilon-greedy strategy to choose the action:
            ca.choose_action(cues=cues,
                             epsilon=0.5)

        :param cues:
            ``numpy.ndarray`` specifying :math:`N` cues that the RL agent senses. The cues are the input parameters
            to the Q-network. It has to have size ``(1, N)``.
        :param epsilon:
            ``float`` specifying the exploration probability, :math:`\\varepsilon`. It has to be between 0 and 1.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:
        if not isinstance(epsilon, float):
            raise ValueError("Parameter `epsilon` has to be of type 'float'.")

        if epsilon < 0 or epsilon > 1:
            raise ValueError("Parameter `epsilon` has to be between 0 and 1.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Select random action with probability epsilon:
        if np.random.rand() < epsilon:

            return np.random.choice(self.n_actions)

        # Select the currently best action with probability (1 - epsilon):
        else:

            # Not sure for the moment, which one should I be using?
            # q_values = self.target_q_network(cues)
            q_values = self.online_q_network(cues)

            return np.argmax(q_values)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def remember(self, cues, action, reward, next_cues):
        """
        Adds the complete outcome of taking a step in the environment to the memory bank.
        That outcome is represented by a tuple:

        .. math::

            (\\text{cues}, \\text{action}, \\text{reward}, \\text{next cues})

        where the cues change from :math:`\\text{cues}` to :math:`\\text{next cues}` after taking an action,
        and there is a numerical value of the reward associated with that action.

        **Example:**

        .. code:: python

            # Once the camera agent has been initialized:
            ca = CameraAgent(...)

            # We can reset the environment to initialize the interrogation window:
            camera_position, cues = ca.env.reset()

            # The typical interaction with the environment will be choosing an action and executing it,
            # and remembering the outcome of that step by adding it to the memory bank.
            # The code below can represent one episode:
            for _ in range(0,100):

                action = ca.choose_action(cues,
                                          epsilon=epsilon)

                next_camera_position, next_cues, reward = ca.env.step(action,
                                                                      reward_function=reward_function,
                                                                      reward_transformation=reward_transformation,
                                                                      verbose=False)

                # We add the outcomes of the current step to the memory bank:
                ca.remember(cues,
                            action,
                            reward,
                            next_cues)

                cues = next_cues

        :param cues:
            ``numpy.ndarray`` specifying :math:`N` cues that the RL agent senses
            **before** taking a step in the environment. The cues are the input parameters
            to the Q-network. It has to have size ``(1, N)``.
        :param action:
            ``int`` specifying the action to be taken at the current step in the environment.
        :param reward:
            ``float`` specifying the reward received after taking a step.
        :param next_cues:
            ``numpy.ndarray`` specifying :math:`N` cues that the RL agent senses
            **after** taking a step in the environment. The cues are the input parameters
            to the Q-network. It has to have size ``(1, N)``.
        """

        self.memory.add((cues, action, reward, next_cues))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def train(self,
              new_learning_rate=0.001):
        """
        Trains the temporary Q-network (``online_q_network``) with the outcome of a single step in the environment.

        **Example:**

        .. code:: python

            # Once the camera agent has been initialized:
            ca = CameraAgent(...)

            # We can train the agent with a single pass over a batch of training data:
            ca.train(new_learning_rate=0.001)

        :param new_learning_rate: (optional)
            ``float`` specifying the new learning rate to use in the current pass over the minibatch.
        """

        # Before the buffer fills with actual data, we're not training yet:
        if len(self.memory.buffer) < self.batch_size:
            return

        # Once the memory buffer holds enough observations...

        # Now we can start sampling batches from the memory:
        minibatch = self.memory.sample(self.batch_size)

        batch_cues = np.zeros((self.batch_size, self.env.n_cues))
        batch_q_values = np.zeros((self.batch_size, self.n_actions))

        for i, batch_content in enumerate(minibatch):

            cues, action, reward, next_cues = batch_content

            # Compute the next Q-values using the online network to choose the best action:
            next_q_values_selected = self.online_q_network(next_cues).numpy()
            best_next_action = np.argmax(next_q_values_selected, axis=-1)

            # Compute the next Q-values using the target network to have the target Q-value:
            next_q_values_target = self.target_q_network(next_cues).numpy()
            target_q_value = reward + self.discount_factor * next_q_values_target[0][best_next_action[0]]

            # Get current Q-values from the online network:
            q_values = self.online_q_network(cues).numpy()

            # Swap the Q-value we have for the target Q-value that we want to have for this action:
            q_values[0][action] = target_q_value

            # Create a batch:
            batch_cues[i, :] = cues
            batch_q_values[i, :] = q_values

        # Update the optimizer with the new learning rate:
        self.optimizer.learning_rate.assign(new_learning_rate)

        # Teach the Q-network to predict the target Q-values over the current batch:
        history = self.online_q_network.fit(batch_cues,
                                            batch_q_values,
                                            epochs=self.n_epochs,
                                            verbose=0)

        # Append the losses:
        self.MSE_losses.append(history.history['loss'])

    def update_target_network(self):
        """
        Synchronizes the target Q-network with the online Q-network in double Q-learning
        (see `Hasselt et al. <https://arxiv.org/abs/1509.06461>`_ for more information).
        This function can be called once every a couple of steps in the environment,
        or even once every a couple of episodes.

        **Example:**

        .. code:: python

            # The general abstraction for synchronizing the two Q-networks in a training loop
            # can be the following:
            for episode in range(0,n_episodes):

                ...

                # Synchronize the networks once every 10 episodes:
                if (episode+1) % 10 == 0:
                    ca.update_target_network()
        """

        self.target_q_network.set_weights(self.online_q_network.get_weights())

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
####    Plot the agent's trajectory over an environment
####
########################################################################################################################
########################################################################################################################

def plot_trajectory(trajectory,
                    quantity=None,
                    vector_field=None,
                    interrogation_window_size=None,
                    interrogation_window_size_buffer=None,
                    c_path='white',
                    c_init='white',
                    c_final='black',
                    s=10,
                    lw=2,
                    xlabel=None,
                    ylabel=None,
                    xticks=True,
                    yticks=True,
                    cmap='viridis',
                    vmin=None,
                    vmax=None,
                    add_quiver=False,
                    quiver_step=10,
                    quiver_color='k',
                    quiver_linewidths=1,
                    add_streamplot=False,
                    streamplot_density=1,
                    streamplot_color='k',
                    streamplot_linewidth=1,
                    figsize=(10, 5),
                    dpi=300,
                    filename=None):
    """
    Plots the trajectory taken by the trained RL agent in a new environment.

    **Example:**

    .. code:: python

        from pykitPIV import plot_trajectory

        # Assuming that we will plot the displacement field magnitude as the scalar quantity:
        displacement_field_magnitude = ...

        # And we will plot the displacement field as the vector field:
        displacement_field = ...

        # And we have computed the trajectory matrix:
        trajectory = ...

        # We can visualize the trajectory in the virtual environment:
        plot_trajectory(trajectory,
                        quantity=displacement_field_magnitude,
                        vector_field=displacement_field,
                        interrogation_window_size=(60,60),
                        interrogation_window_size_buffer=10,
                        c_path='white',
                        c_init='white',
                        c_final='black',
                        s=10,
                        lw=2,
                        xlabel=None,
                        ylabel=None,
                        xticks=True,
                        yticks=True,
                        cmap='viridis',
                        vmin=None,
                        vmax=None,
                        add_quiver=False,
                        quiver_step=10,
                        quiver_color='k',
                        quiver_linewidths=1,
                        add_streamplot=False,
                        streamplot_density=1,
                        streamplot_color='k',
                        streamplot_linewidth=1,
                        figsize=(10, 5),
                        dpi=300,
                        filename=None)

    One example of plotted trajectory after training the agent is:

    .. image:: ../images/ml_plot_trajectory.png
        :width: 800

    :param trajectory:
        ``numpy.ndarray`` specifying the camera trajectory taken by the agent in the 2D virtual wind tunnel.
        It should be of size :math:`(n_s, 2)`, where :math:`n_s` is the number of steps taken in the environment,
        and columns represent the camera position along the height and width of the virtual wind tunnel, respectively.
    :param quantity: (optional)
        ``numpy.ndarray`` specifying the scalar quantity to visualize in the background.
        It should be of size :math:`(H_{\\text{wt}}, W_{\\text{wt}})`.
    :param vector_field: (optional)
        ``numpy.ndarray`` specifying the vector field to visualize on top of any scalar quantity.
        It should be of size :math:`(1, 2, H_{\\text{wt}}, W_{\\text{wt}})`.
    :param interrogation_window_size: (optional)
        ``tuple`` of two ``int`` elements specifying the size of the interrogation window in pixels :math:`[\\text{px}]`.
        The first number is the window height, :math:`H_{\\text{i}}`,
        the second number is the window width, :math:`W_{\\text{i}}`.
    :param interrogation_window_size_buffer: (optional)
        ``int`` specifying the buffer, :math:`b`, in pixels :math:`[\\text{px}]` to add to the interrogation window size
        in the width and height direction.
    :param c: (optional)
        ``str`` specifying the color for the interrogation window outline.
    :param s: (optional)
        ``int`` or ``float`` specifying the size of the dot that represents the camera position.
    :param lw: (optional)
        ``int``  or ``float`` specifying the line width for the interrogation window outline.
    :param xlabel: (optional)
        ``str`` specifying :math:`x`-label.
    :param ylabel: (optional)
        ``str`` specifying :math:`y`-label.
    :param xticks: (optional)
        ``bool`` specifying if ticks along the :math:`x`-axis should be plotted.
    :param yticks: (optional)
        ``bool`` specifying if ticks along the :math:`y`-axis should be plotted.
    :param cmap: (optional)
        ``str`` or an object of `matplotlib.colors.ListedColormap <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html>`_ specifying the color map to use.
    :param vmin: (optional)
        ``int`` or ``float`` specifying the minimum on the colorbar.
    :param vmax: (optional)
        ``int`` or ``float`` specifying the maximum on the colorbar.
    :param normalize_cbars: (optional)
        ``bool`` specifying if the colorbar for the interrogation window should be normalized to the colorbar for
        the entire wind tunnel.
    :param add_quiver: (optional)
        ``bool`` specifying if vector field should be plotted on top of the scalar magnitude field.
    :param quiver_step: (optional)
        ``int`` specifying the step on the pixel grid to attach a vector to. The higher this number is, the less dense the vector field is.
    :param quiver_color: (optional)
        ``str`` specifying the color of vectors.
    :param quiver_linewidths: (optional)
        ``int`` or ``float`` specifying the line widths of vectors.
    :param add_streamplot: (optional)
        ``bool`` specifying if streamlines should be plotted on top of the scalar magnitude field.
    :param streamplot_density: (optional)
        ``float`` or ``int`` specifying the streamplot density.
    :param streamplot_color: (optional)
        ``str`` specifying the streamlines color.
    :param streamplot_linewidth: (optional)
        ``int`` or ``float`` specifying the line width for the streamplot.
    :param figsize: (optional)
        ``tuple`` of two numerical elements specifying the figure size as per ``matplotlib.pyplot``.
    :param dpi: (optional)
        ``int`` specifying the dpi for the image.
    :param filename: (optional)
        ``str`` specifying the path and filename to save an image. If set to ``None``, the image will not be saved.

    :return:
        - **plt** - ``matplotlib.pyplot`` image handle.
        - **cbar** - ``matplotlib.pyplot`` colorbar handle.
    """

    fontsize = 14

    plt.figure(figsize=figsize)

    # Visualize the scalar quantity:
    if quantity is not None:
        ims = plt.imshow(quantity, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, zorder=0)
        cbar = plt.colorbar(ims)
    else:
        cbar = None

    # Visualize the vector field on top of the scalar quantity:
    if vector_field is not None:

        if add_streamplot:

            X = np.arange(0, vector_field.shape[3], 1)
            Y = np.arange(0, vector_field.shape[2], 1)

            plt.streamplot(X, Y,
                           vector_field[0, 0, :, :],
                           vector_field[0, 1, :, :],
                           density=streamplot_density,
                           color=streamplot_color,
                           linewidth=streamplot_linewidth,
                           zorder=1)

        if add_quiver:

            X = np.arange(0, vector_field.shape[3], quiver_step)
            Y = np.arange(0, vector_field.shape[2], quiver_step)

            plt.quiver(X, Y,
                       vector_field[0, 0, ::quiver_step, ::quiver_step],
                       vector_field[0, 1, ::quiver_step, ::quiver_step],
                       color=quiver_color,
                       linewidths=quiver_linewidths)

    plt.plot(trajectory[:, 1] - 0.5, trajectory[:, 0] - 0.5, c=c_path, lw=lw, zorder=2)

    # Visualize a rectangle that defines the initial interrogation window:
    plt.scatter(trajectory[0, 1] - 0.5, trajectory[0, 0] - 0.5, c=c_init, s=s, zorder=2)

    if interrogation_window_size is not None:

        # Visualize a rectangle that defines the current interrogation window:
        rect = patches.Rectangle((trajectory[0, 1] - 0.5, trajectory[0, 0] - 0.5),
                                 interrogation_window_size[1] + 2*interrogation_window_size_buffer,
                                 interrogation_window_size[0] + 2*interrogation_window_size_buffer,
                                 linewidth=lw, edgecolor=c_init, facecolor='none', zorder=2)
        ax = plt.gca()
        ax.add_patch(rect)

        # Visualize a rectangle that defines the current interrogation window without buffer:
        rect = patches.Rectangle((trajectory[0, 1] + interrogation_window_size_buffer - 0.5, trajectory[0, 0] + interrogation_window_size_buffer - 0.5),
                                 interrogation_window_size[1],
                                 interrogation_window_size[0],
                                 linewidth=lw/2, edgecolor=c_init, facecolor='none', zorder=2)
        ax = plt.gca()
        ax.add_patch(rect)

    # Visualize a rectangle that defines the final interrogation window:
    plt.scatter(trajectory[-1, 1] - 0.5, trajectory[-1, 0] - 0.5, c=c_final, s=s, zorder=2)

    if interrogation_window_size is not None:

        # Visualize a rectangle that defines the current interrogation window:
        rect = patches.Rectangle((trajectory[-1, 1] - 0.5, trajectory[-1, 0] - 0.5),
                                 interrogation_window_size[1] + 2*interrogation_window_size_buffer,
                                 interrogation_window_size[0] + 2*interrogation_window_size_buffer,
                                 linewidth=lw, edgecolor=c_final, facecolor='none', zorder=2)
        ax = plt.gca()
        ax.add_patch(rect)

        # Visualize a rectangle that defines the current interrogation window without buffer:
        rect = patches.Rectangle((trajectory[-1, 1] + interrogation_window_size_buffer - 0.5, trajectory[-1, 0] + interrogation_window_size_buffer - 0.5),
                                 interrogation_window_size[1],
                                 interrogation_window_size[0],
                                 linewidth=lw/2, edgecolor=c_final, facecolor='none', zorder=2)
        ax = plt.gca()
        ax.add_patch(rect)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if not xticks:
        plt.xticks([])

    if not yticks:
        plt.yticks([])

    plt.title('Virtual wind tunnel', fontsize=fontsize)

    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')

    return plt, cbar

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

    Functions of this class can be directly used as the ``reward_function`` parameter in ``pykitPIV.ml.PIVEnv.step()``.

    **Example:**

    .. code:: python

        from pykitPIV.ml import Rewards

        # Instantiate an object of the Rewards class:
        rewards = Rewards()

    :param verbose: (optional)
        ``bool`` specifying if the verbose print statements should be displayed.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``.
        If specified, all operations are reproducible.
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

    @property
    def verbose(self):
        return self.__verbose

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def divergence(self,
                   vector_field,
                   transformation):
        """
        Computes the reward based on the divergence of the flow field.

        **Example:**

        .. code:: python

            from pykitPIV.ml import Rewards
            import numpy as np

            # Once we have the vector field specified
            # e.g., velocity field or displacement field:
            vector_field = ...

            # Instantiate an object of the Rewards class:
            rewards = Rewards(verbose=True,
                              random_seed=None)

            # Design a custom transformation that looks for regions of high divergence (either positive or negative)
            # and computes the maximum absolute value of divergence in that region:
            def transformation(div):
                return np.max(np.abs(div))

            # Compute the reward based on the divergence for the present velocity field:
            reward = rewards.divergence(vector_field=vector_field,
                                        transformation=transformation)

        :param vector_field:
            ``numpy.ndarray`` specifying the vector field components under the interrogation window.
            It should be of size :math:`(1, 2, H_{\\text{i}}+2b, W_{\\text{i}}+2b)`,
            where :math:`1` is just one, fixed vector field, :math:`2` refers to each vector field component.
            For example, it can be the velocity field with components :math:`u` and :math:`v`, or
            the displacement field with components :math:`dx` and :math:`dy`.
            :math:`H_{\\text{i}}+2b` is the height and
            :math:`W_{\\text{i}}+2b` is the width of the interrogation window.
        :param transformation:
            ``function`` specifying an arbitrary transformation of the Q-criterion
            and an arbitrary compression of the Q-criterion field to a single value.

        :return:
            - **reward** - ``float`` specifying the reward, :math:`R`.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not callable(transformation):
            raise ValueError("Parameter `transformation` has to be a callable.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        reward = transformation(compute_divergence(vector_field))

        # Check that the reward is a single number (int or float).
        # Otherwise, it will not work well with Q-learning:
        if not isinstance(reward, int) and not isinstance(reward, float):
            raise TypeError("Parameter `transformation` has to return a single number of type 'int' or 'float' but returns " + str(type(reward)) + " instead.")

        if self.__verbose: print(reward)

        return reward

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def vorticity(self,
                  vector_field,
                  transformation):
        """
        Computes the reward based on the vorticity of the flow field.

        **Example:**

        .. code:: python

            from pykitPIV.ml import Rewards
            import numpy as np

            # Once we have the vector field specified
            # e.g., velocity field or displacement field:
            vector_field = ...

            # Instantiate an object of the Rewards class:
            rewards = Rewards(verbose=True,
                              random_seed=None)

            # Design a custom transformation that looks for regions of high vorticity (either positive or negative)
            # and computes the mean absolute value of vorticity in that region:
            def transformation(vort):
                return np.mean(np.abs(vort))

            # Compute the reward based on the divergence for the present velocity field:
            reward = rewards.vorticity(vector_field=vector_field,
                                       transformation=transformation)

        :param vector_field:
            ``numpy.ndarray`` specifying the vector field components under the interrogation window.
            It should be of size :math:`(1, 2, H_{\\text{i}}+2b, W_{\\text{i}}+2b)`,
            where :math:`1` is just one, fixed vector field, :math:`2` refers to each vector field component.
            For example, it can be the velocity field with components :math:`u` and :math:`v`, or
            the displacement field with components :math:`dx` and :math:`dy`.
            :math:`H_{\\text{i}}+2b` is the height and
            :math:`W_{\\text{i}}+2b` is the width of the interrogation window.
        :param transformation:
            ``function`` specifying an arbitrary transformation of the Q-criterion
            and an arbitrary compression of the Q-criterion field to a single value.

        :return:
            - **reward** - ``float`` specifying the reward, :math:`R`.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not callable(transformation):
            raise ValueError("Parameter `transformation` has to be a callable.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        reward = transformation(compute_vorticity(vector_field))

        # Check that the reward is a single number (int or float).
        # Otherwise, it will not work well with Q-learning:
        if not isinstance(reward, int) and not isinstance(reward, float):
            raise TypeError("Parameter `transformation` has to return a single number of type 'int' or 'float' but returns " + str(type(reward)) + " instead.")

        if self.__verbose: print(reward)

        return reward

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def q_criterion(self,
                    vector_field,
                    transformation):
        """
        Computes the reward based on the Q-criterion.

        **Example:**

        .. code:: python

            from pykitPIV.ml import Rewards
            import numpy as np

            # Once we have the vector field specified
            # e.g., velocity field or displacement field:
            vector_field = ...

            # Instantiate an object of the Rewards class:
            rewards = Rewards(verbose=True,
                              random_seed=None)

            # Design a custom transformation that looks for vorticity-dominated regions
            # (as opposed to shear-dominated regions)
            # and computes the maximum value of the Q-criterion in that region:
            def transformation(Q):
                Q = np.max(Q.clip(min=0))
                return Q

            # Compute the reward based on the Q-criterion for the present velocity field:
            reward = rewards.q_criterion(vector_field=vector_field,
                                         transformation=transformation)

        :param vector_field:
            ``numpy.ndarray`` specifying the vector field components under the interrogation window.
            It should be of size :math:`(1, 2, H_{\\text{i}}+2b, W_{\\text{i}}+2b)`,
            where :math:`1` is just one, fixed vector field, :math:`2` refers to each vector field component.
            For example, it can be the velocity field with components :math:`u` and :math:`v`, or
            the displacement field with components :math:`dx` and :math:`dy`.
            :math:`H_{\\text{i}}+2b` is the height and
            :math:`W_{\\text{i}}+2b` is the width of the interrogation window.
        :param transformation:
            ``function`` specifying an arbitrary transformation of the Q-criterion
            and an arbitrary compression of the Q-criterion field to a single value.

        :return:
            - **reward** - ``float`` specifying the reward, :math:`R`.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:

        if not callable(transformation):
            raise ValueError("Parameter `transformation` has to be a callable.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        reward = transformation(compute_q_criterion(vector_field))

        # Check that the reward is a single number (int or float).
        # Otherwise, it will not work well with Q-learning:
        if not isinstance(reward, int) and not isinstance(reward, float):
            raise TypeError("Parameter `transformation` has to return a single number of type 'int' or 'float' but returns " + str(type(reward)) + " instead.")

        if self.__verbose: print(reward)

        return reward

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

########################################################################################################################
########################################################################################################################
####
####    Class: Cues
####
########################################################################################################################
########################################################################################################################

class Cues:
    """
    Provides custom sensory cues functions that are applicable to various tasks related to navigating
    a virtual wind tunnel and a virtual PIV experiment. The cue vector, :math:`\\mathbf{c}`,
    is computed only based on the reconstructed displacement field, :math:`\\vec{d\\mathbf{s}}`,
    in the interrogation window.

    Each function returns a vector of cues, :math:`\\mathbf{c}`, that is a ``numpy.ndarray`` of :math:`N` cues and
    of size :math:`(1,N)`.

    The sensory cues can become the inputs to Q-networks when training RL agents.

    Functions of this class can be directly used as the ``cues_function`` parameter in ``pykitPIV.ml.PIVEnv``.

    **Example:**

    .. code:: python

        from pykitPIV.ml import Cues

        # Instantiate an object of the Cues class:
        cues_obj = Cues(verbose=False,
                        random_seed=None,
                        sample_every_n=10,
                        normalize_displacement_vectors=False)

    :param verbose: (optional)
        ``bool`` specifying if the verbose print statements should be displayed.
    :param random_seed: (optional)
        ``int`` specifying the random seed for random number generation in ``numpy``.
        If specified, all operations are reproducible.
    :param sample_every_n: (optional)
        ``int`` specifying the step in sampling the displacement vectors over a uniform grid.
    :param normalize_displacement_vectors: (optional)
        ``bool`` specifying whether the (sampled) displacement vectors should be normalized to unit length.
    """

    def __init__(self,
                 verbose=False,
                 random_seed=None,
                 sample_every_n=10,
                 normalize_displacement_vectors=False):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Input parameter check:
        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be of type 'bool'.")

        if random_seed is not None:
            if type(random_seed) != int:
                raise ValueError("Parameter `random_seed` has to be of type 'int'.")
            else:
                np.random.seed(seed=random_seed)

        if not isinstance(sample_every_n, int):
            raise ValueError("Parameter `sample_every_n` has to be of type 'int'.")

        if not isinstance(normalize_displacement_vectors, bool):
            raise ValueError("Parameter `normalize_displacement_vectors` has to be of type 'bool'.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Class init:
        self.__verbose = verbose
        self.__random_seed = random_seed
        self.__sample_every_n = sample_every_n
        self.__normalize_displacement_vectors = normalize_displacement_vectors

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Properties coming from user inputs:
    @property
    def verbose(self):
        return self.__verbose

    @property
    def random_seed(self):
        return self.__random_seed

    @property
    def sample_every_n(self):
        return self.__sample_every_n

    @property
    def normalize_displacement_vectors(self):
        return self.__normalize_displacement_vectors

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def sampled_vectors(self,
                        displacement_field):
        """
        Computes the cues vector that contains :math:`n` displacement vectors sampled on a uniform grid from
        the displacement field, :math:`\\vec{d\\mathbf{s}}`:

        .. math::

            \\vec{d\\mathbf{s}}_1, \\vec{d\\mathbf{s}}_2, \\dots, \\vec{d\\mathbf{s}}_n

        These are further organized as:

        .. math::

            \\mathbf{c} = [dx_1, dx_2, \\dots, dx_n, dy_1, dy_2, \\dots, dy_n]

        hence :math:`N = 2n`.

        **Example:**

        .. code:: python

            from pykitPIV.ml import Cues
            import numpy as np

            # Once we have the displacement field specified:
            displacement_field = ...

            # Instantiate an object of the Cues class:
            cues_obj = Cues(verbose=True,
                            random_seed=None,
                            sample_every_n=10,
                            normalize_displacement_vectors=False)

            # Compute the cues vector:
            cues = cues_obj.sampled_vectors(displacement_field=displacement_field)

        :param displacement_field:
            ``numpy.ndarray`` specifying the velocity components under the interrogation window.
            It should be of size :math:`(1, 2, H_{\\text{i}}+2b, W_{\\text{i}}+2b)`,
            where :math:`1` is just one, fixed flow field, :math:`2` refers to each velocity component
            :math:`u` and :math:`v` respectively,
            :math:`H_{\\text{i}}+2b` is the height and
            :math:`W_{\\text{i}}+2b` is the width of the interrogation window.

        :return:
            - **cues** - ``numpy.ndarray`` specifying the cues vector, :math:`\mathbf{c}`. It has shape :math:`(1,N)`.
        """

        # Sample on a uniform grid:
        (_, _, H, W) = displacement_field.shape
        idx_H = [i for i in range(0, H) if i % self.__sample_every_n == 0]
        idx_W = [i for i in range(0, W) if i % self.__sample_every_n == 0]

        idx_W, idx_H = np.meshgrid(idx_W, idx_H)

        dx_sample = displacement_field[0, 0, idx_H, idx_W]
        dy_sample = displacement_field[0, 1, idx_H, idx_W]

        if self.__normalize_displacement_vectors:

            magnitude = np.sqrt(dx_sample ** 2 + dy_sample ** 2)

            normalized_dx = dx_sample / magnitude
            normalized_dy = dy_sample / magnitude

            cues = np.hstack((normalized_dx.ravel()[None,:], normalized_dy.ravel()[None,:]))

        else:

            cues = np.hstack((dx_sample.ravel()[None, :], dy_sample.ravel()[None, :]))

        return cues

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def sampled_magnitude(self,
                          displacement_field):
        """
        Computes the cues vector that contains :math:`N` displacement magnitudes sampled on a uniform grid from
        the displacement field magnitude, :math:`|\\vec{d\\mathbf{s}}|`:

        .. math::

            \\mathbf{c} = [|\\vec{d\\mathbf{s}}_1|, |\\vec{d\\mathbf{s}}_2|, \\dots, |\\vec{d\\mathbf{s}}_N|]

        **Example:**

        .. code:: python

            from pykitPIV.ml import Cues
            import numpy as np

            # Once we have the displacement field specified:
            displacement_field = ...

            # Instantiate an object of the Cues class:
            cues_obj = Cues(verbose=True,
                            random_seed=None,
                            sample_every_n=10,
                            normalize_displacement_vectors=False)

            # Compute the cues vector:
            cues = cues_obj.sampled_magnitude(displacement_field=displacement_field)

        :param displacement_field:
            ``numpy.ndarray`` specifying the velocity components under the interrogation window.
            It should be of size :math:`(1, 2, H_{\\text{i}}+2b, W_{\\text{i}}+2b)`,
            where :math:`1` is just one, fixed flow field, :math:`2` refers to each velocity component
            :math:`u` and :math:`v` respectively,
            :math:`H_{\\text{i}}+2b` is the height and
            :math:`W_{\\text{i}}+2b` is the width of the interrogation window.

        :return:
            - **cues** - ``numpy.ndarray`` specifying the cues vector, :math:`\mathbf{c}`. It has shape :math:`(1,N)`.
        """

        # Sample on a uniform grid:
        (_, _, H, W) = displacement_field.shape
        idx_H = [i for i in range(0, H) if i % self.__sample_every_n == 0]
        idx_W = [i for i in range(0, W) if i % self.__sample_every_n == 0]

        idx_W, idx_H = np.meshgrid(idx_W, idx_H)

        dx_sample = displacement_field[0, 0, idx_H, idx_W]
        dy_sample = displacement_field[0, 1, idx_H, idx_W]

        magnitude = np.sqrt(dx_sample ** 2 + dy_sample ** 2)

        if self.__normalize_displacement_vectors:

            normalized_dx = dx_sample / magnitude
            normalized_dy = dy_sample / magnitude

            normalized_magnitude = np.sqrt(normalized_dx ** 2 + normalized_dy ** 2)

            cues = normalized_magnitude.ravel()[None,:]

        else:

            cues = magnitude.ravel()[None,:]

        return cues

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def sampled_divergence(self,
                           displacement_field):
        """
        Computes the cues vector that contains :math:`N` values for the divergence sampled on a uniform grid from
        the divergence of the displacement field, :math:`d = \\nabla \\cdot \\vec{d\\mathbf{s}}`:

        .. math::

            \\mathbf{c} = [d_1, d_2, \\dots, d_N]

        **Example:**

        .. code:: python

            from pykitPIV.ml import Cues
            import numpy as np

            # Once we have the displacement field specified:
            displacement_field = ...

            # Instantiate an object of the Cues class:
            cues_obj = Cues(verbose=True,
                            random_seed=None,
                            sample_every_n=10,
                            normalize_displacement_vectors=False)

            # Compute the cues vector:
            cues = cues_obj.sampled_divergence(displacement_field=displacement_field)

        :param displacement_field:
            ``numpy.ndarray`` specifying the velocity components under the interrogation window.
            It should be of size :math:`(1, 2, H_{\\text{i}}+2b, W_{\\text{i}}+2b)`,
            where :math:`1` is just one, fixed flow field, :math:`2` refers to each velocity component
            :math:`u` and :math:`v` respectively,
            :math:`H_{\\text{i}}+2b` is the height and
            :math:`W_{\\text{i}}+2b` is the width of the interrogation window.

        :return:
            - **cues** - ``numpy.ndarray`` specifying the cues vector, :math:`\mathbf{c}`. It has shape :math:`(1,N)`.
        """

        # Sample on a uniform grid:
        (_, _, H, W) = displacement_field.shape
        idx_H = [i for i in range(0, H) if i % self.__sample_every_n == 0]
        idx_W = [i for i in range(0, W) if i % self.__sample_every_n == 0]

        idx_W, idx_H = np.meshgrid(idx_W, idx_H)

        if self.__normalize_displacement_vectors:

            magnitude = np.sqrt(displacement_field[0, 0, :, :] ** 2 + displacement_field[0, 1, :, :] ** 2)

            divergence = compute_divergence(displacement_field / magnitude)

        else:

            divergence = compute_divergence(displacement_field)

        divergence_sample = divergence[0, idx_H, idx_W]

        cues = divergence_sample.ravel()[None,:]

        return cues

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def sampled_vorticity(self,
                          displacement_field):
        """
        Computes the cues vector that contains :math:`N` values for the vorticity sampled on a uniform grid from
        the vorticity of the displacement field, :math:`\\omega = \\nabla \\times \\vec{d\\mathbf{s}}`:

        .. math::

            \\mathbf{c} = [\\omega_1, \\omega_2, \\dots, \\omega_N]

        **Example:**

        .. code:: python

            from pykitPIV.ml import Cues
            import numpy as np

            # Once we have the displacement field specified:
            displacement_field = ...

            # Instantiate an object of the Cues class:
            cues_obj = Cues(verbose=True,
                            random_seed=None,
                            sample_every_n=10,
                            normalize_displacement_vectors=False)

            # Compute the cues vector:
            cues = cues_obj.sampled_vorticity(displacement_field=displacement_field)

        :param displacement_field:
            ``numpy.ndarray`` specifying the velocity components under the interrogation window.
            It should be of size :math:`(1, 2, H_{\\text{i}}+2b, W_{\\text{i}}+2b)`,
            where :math:`1` is just one, fixed flow field, :math:`2` refers to each velocity component
            :math:`u` and :math:`v` respectively,
            :math:`H_{\\text{i}}+2b` is the height and
            :math:`W_{\\text{i}}+2b` is the width of the interrogation window.

        :return:
            - **cues** - ``numpy.ndarray`` specifying the cues vector, :math:`\mathbf{c}`. It has shape :math:`(1,N)`.
        """

        # Sample on a uniform grid:
        (_, _, H, W) = displacement_field.shape
        idx_H = [i for i in range(0, H) if i % self.__sample_every_n == 0]
        idx_W = [i for i in range(0, W) if i % self.__sample_every_n == 0]

        idx_W, idx_H = np.meshgrid(idx_W, idx_H)

        if self.__normalize_displacement_vectors:

            magnitude = np.sqrt(displacement_field[0, 0, :, :] ** 2 + displacement_field[0, 1, :, :] ** 2)

            vorticity = compute_vorticity(displacement_field / magnitude)

        else:

            vorticity = compute_vorticity(displacement_field)

        vorticity_sample = vorticity[0, idx_H, idx_W]

        cues = vorticity_sample.ravel()[None,:]

        return cues

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def sampled_q_criterion(self,
                            displacement_field):
        """
        Computes the cues vector that contains :math:`N` values for the Q-criterion sampled on a uniform grid from
        the Q-criterion of the displacement field, :math:`Q`:

        .. math::

            \\mathbf{c} = [Q_1, Q_2, \\dots, Q_N]

        **Example:**

        .. code:: python

            from pykitPIV.ml import Cues
            import numpy as np

            # Once we have the displacement field specified:
            displacement_field = ...

            # Instantiate an object of the Cues class:
            cues_obj = Cues(verbose=True,
                            random_seed=None,
                            sample_every_n=10,
                            normalize_displacement_vectors=False)

            # Compute the cues vector:
            cues = cues_obj.sampled_q_criterion(displacement_field=displacement_field)

        :param displacement_field:
            ``numpy.ndarray`` specifying the velocity components under the interrogation window.
            It should be of size :math:`(1, 2, H_{\\text{i}}+2b, W_{\\text{i}}+2b)`,
            where :math:`1` is just one, fixed flow field, :math:`2` refers to each velocity component
            :math:`u` and :math:`v` respectively,
            :math:`H_{\\text{i}}+2b` is the height and
            :math:`W_{\\text{i}}+2b` is the width of the interrogation window.

        :return:
            - **cues** - ``numpy.ndarray`` specifying the cues vector, :math:`\mathbf{c}`. It has shape :math:`(1,N)`.
        """

        # Sample on a uniform grid:
        (_, _, H, W) = displacement_field.shape
        idx_H = [i for i in range(0, H) if i % self.__sample_every_n == 0]
        idx_W = [i for i in range(0, W) if i % self.__sample_every_n == 0]

        idx_W, idx_H = np.meshgrid(idx_W, idx_H)

        if self.__normalize_displacement_vectors:

            magnitude = np.sqrt(displacement_field[0, 0, :, :] ** 2 + displacement_field[0, 1, :, :] ** 2)

            Q = compute_q_criterion(displacement_field / magnitude)

        else:

            Q = compute_q_criterion(displacement_field)

        Q_sample = Q[0, idx_H, idx_W]

        cues = Q_sample.ravel()[None,:]

        return cues

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -