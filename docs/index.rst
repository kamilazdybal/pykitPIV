.. image:: images/pykitPIV-logo.png
    :width: 300

**pykitPIV** (**Py**\ thon **ki**\ nematic **t**\ raining for **P**\ article **I**\ mage **V**\ elocimetry) is a
Python package that generates rich and reproducible synthetic data for training machine learning algorithms
in velocimetry, a subfield of experimental fluid dynamics.

**pykitPIV** gives the user, or the machine-learning agent, a lot of flexibility in selecting
various parameters that would normally be available in an experimental setting, such as seeding density,
properties of the laser plane, camera exposure, particle loss, or experimental noise.
We also provide an atlas of challenging synthetic velocity fields from analytic formulations.
The effects of particle drift and diffusion in stationary isotropic turbulence can also be added atop the
velocity fields using the simplified Langevin model (SLM).

**pykitPIV** can operate in particle image velocimetry (PIV) and background-oriented Schlieren (BOS) mode.
It exploits the idea that if the time separation between two experimental images is small,
kinematic relationship between two consecutive images is sufficient to determine particle displacement fields.

The graph below shows the possible workflows constructed from the five main classes:

.. image:: images/pykitPIV-workflow.svg
    :width: 900
    :align: center

The ``ml`` module provides integrations with machine learning algorithms, such as **convolutional neural networks**,
**variational approaches**, **active learning**, or **reinforcement learning**.

For more information on kinematic training of convolutional neural networks (CNNs) using synthetic PIV/BOS images, please
check the following references:

- `Kinematic training of convolutional neural networks for particle image velocimetry <https://iopscience.iop.org/article/10.1088/1361-6501/ac8fae/meta>`_

- `A lightweight neural network designed for fluid velocimetry <https://link.springer.com/article/10.1007/s00348-023-03695-8>`_

- `A lightweight convolutional neural network to reconstruct deformation in BOS recordings <https://link.springer.com/article/10.1007/s00348-023-03618-7>`_

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
    tutorials/demo-data-augmentation
    tutorials/demo-import-external-velocity-field
    tutorials/demo-integrate-with-LIMA

