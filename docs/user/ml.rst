.. module:: ml

#######################
Module: ``ml``
#######################

*********************************************
Data primitives and data loaders
*********************************************

.. warning::

    **pykitPIV**'s PIV images, and the associated targets, are generated assuming that the origin of the coordinate
    system is the lower-left corner (equivalent to setting ``origin='lower'`` in ``plt.imshow()``).
    The *bottom* boundary of a PIV image corresponds to ``[0,:]`` rows from the raw ``numpy`` arrays that store, *e.g.*, the image intensities
    or the velocity components.
    The *top* boundary of a PIV image corresponds to ``[-1,:]`` rows from the raw ``numpy`` arrays.

    Whenever the PIV images need to be interpreted with the origin in the upper-left corner
    (equivalent to setting ``origin='upper'`` in ``plt.imshow()``), the :math:`v`-component of velocity has to be multiplied by :math:`-1`.
    This is the case when using the generated **pykitPIV** dataset as raw ``numpy`` arrays, such as in an input/output
    to a convolutional neural network (CNN). A CNN processes arrays assuming that the *top* boundary is ``[0,:]`` and the *bottom*
    boundary is ``[-1,:]``. With this flip, the :math:`v`-component of velocity has to swap sign, such that whatever was a positive
    :math:`v`-component (from the old *bottom* to *top*) now is a negative :math:`v`-component (from the new *top* to *bottom*).

    This preserves the effect (divergence or convergence) that the :math:`v`-component of velocity has on particles.

    .. image:: ../images/pykitPIV-dataloader-warning.svg
        :width: 700
        :align: center

**PyTorch**-based
====================================

For more information on **PyTorch** data primitives visit this `PyTorch documentation website <https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_.

.. autoclass:: pykitPIV.ml.PIVDatasetPyTorch

**TensorFlow**-based or **Keras**-based
=======================================

For more information on **TensorFlow** data primitives visit this `TensorFlow documentation website <https://www.tensorflow.org/guide/data>`_.

.. autoclass:: pykitPIV.ml.PIVDatasetTF

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

**TF-Agents**-based
====================================

.. note::

    The future version will include environments that can be directly used with `TF-Agents <https://www.tensorflow.org/agents>`_.

*********************************************
Reinforcement learning agents
*********************************************

Single deep Q-network training
====================================

.. autoclass:: pykitPIV.ml.CameraAgentSingleDQN

.. autofunction:: pykitPIV.ml.CameraAgentSingleDQN.choose_action

.. autofunction:: pykitPIV.ml.CameraAgentSingleDQN.train

.. autofunction:: pykitPIV.ml.CameraAgentSingleDQN.view_weights

Double deep Q-network training with memory replay
==========================================================

.. autoclass:: pykitPIV.ml.CameraAgentDoubleDQN

.. autofunction:: pykitPIV.ml.CameraAgentDoubleDQN.choose_action

.. autofunction:: pykitPIV.ml.CameraAgentDoubleDQN.remember

.. autofunction:: pykitPIV.ml.CameraAgentDoubleDQN.train

.. autofunction:: pykitPIV.ml.CameraAgentDoubleDQN.update_target_network

.. autofunction:: pykitPIV.ml.CameraAgentDoubleDQN.view_weights

*********************************************
Reinforcement learning rewards
*********************************************

.. note::

    The future version will also include unsupervised rewards computed directly on PIV image pairs, such as image
    frame consistency, smoothness, Charbonier loss, etc.

.. autoclass:: pykitPIV.ml.Rewards

.. autofunction:: pykitPIV.ml.Rewards.divergence

.. autofunction:: pykitPIV.ml.Rewards.vorticity

.. autofunction:: pykitPIV.ml.Rewards.q_criterion

*********************************************
Reinforcement learning sensory cues
*********************************************

.. note::

    The future version will also include unsupervised sensory cues computed directly on PIV image pairs, such as image
    frame consistency, smoothness, Charbonier loss, etc.

.. autoclass:: pykitPIV.ml.Cues

.. autofunction:: pykitPIV.ml.Cues.sampled_vectors

.. autofunction:: pykitPIV.ml.Cues.sampled_magnitude

.. autofunction:: pykitPIV.ml.Cues.sampled_divergence

.. autofunction:: pykitPIV.ml.Cues.sampled_vorticity

.. autofunction:: pykitPIV.ml.Cues.sampled_q_criterion

*********************************************
Reinforcement learning utilities
*********************************************

.. autofunction:: pykitPIV.ml.plot_trajectory