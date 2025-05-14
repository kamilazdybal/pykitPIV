############################################################################################
Create a **Gymnasium** reinforcement learning environment
############################################################################################

This tutorial can be accessed from the Jupyter notebook: `demo-pykitPIV-12-construct-RL-environment.ipynb <https://gitlab.empa.ch/kamila.zdybal/pykitPIV/-/blob/main/jupyter-notebooks/demo-pykitPIV-12-construct-RL-environment.ipynb>`_.




************************************************************
Introduction
************************************************************



************************************************************
The ``PIVEnv`` class
************************************************************

**pykitPIV** provides the ``PIVEnv`` class that is a subclass of the ``gymnasium.Env`` class.
This class implements the reinforcement learning (RL) environment and provides a virtual PIV/BOS setup
for the RL agent to interact with. The RL agent has the ability to interact with any of the **pykitPIV** parameters.
This class can be readily accessed from the ``ml`` module.

.. code:: python

    import gymnasium as gym

    class PIVEnv(gym.Env):

