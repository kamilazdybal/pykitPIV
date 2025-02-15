.. image:: images/pykitPIV-logo-large.svg
    :width: 700
    :align: center

**pykitPIV** is a Python package that provides rich and reproducible synthetic data and virtual environments
for training machine learning algorithms in velocimetry.

**pykitPIV** gives the user, or the machine-learning agent, a lot of flexibility in selecting
various parameters that would normally be available in an experimental setting, such as seeding density,
properties of the laser plane, camera exposure, particle loss, or experimental noise.
We also provide an atlas of challenging synthetic velocity fields from analytic formulations.
The effects of particle drift and diffusion in stationary isotropic turbulence can also be added atop the
velocity fields using the simplified Langevin model (SLM).

**pykitPIV** can operate in particle image velocimetry (PIV) and background-oriented Schlieren (BOS) mode.
It exploits the idea that if the time separation between two experimental images is small,
kinematic relationship between two consecutive images is sufficient to determine particle displacement fields.

The graph below shows the possible workflows constructed from the five main classes that act as a
virtual experimental setup, or a *virtual wind tunnel*.
The ``ml`` module connects with each of the five main classes and provides integrations with machine learning
algorithms, such as **convolutional neural networks**, **reinforcement learning**, **variational approaches**,
and **active learning**. ML agents have freedom in interacting with the virtual experiment and can be trained
to perform a variety of tasks using diverse rewards.

.. image:: images/pykitPIV-modules.svg
    :width: 900
    :align: center

------------------------------------------------------------------------------------------------------------------------

.. toctree::
    :maxdepth: 5
    :caption: Installation

    installation/instruction

.. toctree::
    :maxdepth: 5
    :caption: Preliminaries

    preliminaries/classes
    preliminaries/indexing
    preliminaries/setting-spectra

.. toctree::
    :maxdepth: 2
    :caption: Quickstart

    quickstart/quickstart-scripts
    quickstart/gallery

.. toctree::
    :maxdepth: 5
    :caption: User Guide

    user/particle
    user/flowfield
    user/motion
    user/image
    user/postprocess
    user/ml

.. toctree::
    :maxdepth: 2
    :caption: Tutorials

    tutorials/demo-generate-images
    tutorials/demo-image-statistics
    tutorials/demo-PyTorch-dataloader
    tutorials/demo-Gymnasium-Env
    tutorials/demo-data-augmentation
    tutorials/demo-import-external-velocity-field
    tutorials/demo-integrate-with-LIMA

