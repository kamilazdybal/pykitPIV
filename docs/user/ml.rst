.. module:: ml

#######################
Module: ``ml``
#######################

*********************************************
**PyTorch** primitives
*********************************************

.. autoclass:: pykitPIV.ml.PIVDataset

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

.. autofunction:: pykitPIV.ml.PIVEnv.record_particles

.. autofunction:: pykitPIV.ml.PIVEnv.make_inference

.. autofunction:: pykitPIV.ml.PIVEnv.reset

.. autofunction:: pykitPIV.ml.PIVEnv.step

.. autofunction:: pykitPIV.ml.PIVEnv.render

.. autoclass:: pykitPIV.ml.CameraAgent

.. autofunction:: pykitPIV.ml.CameraAgent.choose_action

.. autofunction:: pykitPIV.ml.CameraAgent.remember

.. autofunction:: pykitPIV.ml.CameraAgent.train

.. autofunction:: pykitPIV.ml.CameraAgent.update_target_network

.. autofunction:: pykitPIV.ml.CameraAgent.view_weights

**TF-Agents**-based
====================================

