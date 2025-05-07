.. module:: ml

#######################
Module: ``ml``
#######################

*********************************************
Data primitives and data loaders
*********************************************

**PyTorch**-based
====================================

For more information on **PyTorch** data primitives visit this `PyTorch documentation website <https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_.

.. autoclass:: pykitPIV.ml.PIVDatasetPyTorch

**TensorFlow**-based
====================================

For more information on **TensorFlow** data primitives visit this `TensorFlow documentation website <https://www.tensorflow.org/guide/data>`_.

.. autoclass:: pykitPIV.ml.PIVDatasetTF

**Keras**-based
====================================

For more information on **Keras** data primitives visit this `Keras documentation website <https://keras.io/api/data_loading/>`_.

.. autoclass:: pykitPIV.ml.PIVDatasetKeras

*********************************************
Reinforcement learning environments
*********************************************

**Gymnasium**-based
====================================

Below, you will find reinforcement learning (RL) environments that are subclasses of
`gymnasium.Env <https://gymnasium.farama.org/api/env/#gymnasium.Env>`_ -
a class coming from OpenAI's `Gymnasium <https://gymnasium.farama.org/>`_ library that provides the basic
structure for any RL environment.

.. autoclass:: pykitPIV.ml.PIVEnv

.. autofunction:: pykitPIV.ml.PIVEnv.reset

.. autofunction:: pykitPIV.ml.PIVEnv.step

.. autofunction:: pykitPIV.ml.PIVEnv.render

.. autofunction:: pykitPIV.ml.PIVEnv.record_particles

.. autofunction:: pykitPIV.ml.PIVEnv.make_inference

.. autoclass:: pykitPIV.ml.CameraAgent

.. autofunction:: pykitPIV.ml.CameraAgent.choose_action

.. autofunction:: pykitPIV.ml.CameraAgent.remember

.. autofunction:: pykitPIV.ml.CameraAgent.train

.. autofunction:: pykitPIV.ml.CameraAgent.update_target_network

.. autofunction:: pykitPIV.ml.CameraAgent.view_weights

.. autofunction:: pykitPIV.ml.plot_trajectory

**TF-Agents**-based
====================================




*********************************************
Custom reinforcement learning rewards
*********************************************

.. autoclass:: pykitPIV.ml.Rewards

.. autofunction:: pykitPIV.ml.Rewards.q_criterion

.. autofunction:: pykitPIV.ml.Rewards.divergence

*********************************************
Custom reinforcement learning cues
*********************************************

.. autoclass:: pykitPIV.ml.Cues

.. autofunction:: pykitPIV.ml.Cues.sampled_vectors

