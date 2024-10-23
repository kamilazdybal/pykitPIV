.. image:: images/pykitPIV-logo.png
  :width: 300

**pykitPIV** (**Py**\ thon **ki**\ nematic **t**\ raining for **P**\ article **I**\ mage **V**\ elocimetry) is a
PyTorch-compatible, Python package for synthetic PIV image generation.
It exploits the idea that if the time separation between two PIV images is small,
kinematic relationship between two consecutive PIV images is sufficient to determine particle displacement fields.
**pykitPIV** can also operate in the background-oriented Schlieren (BOS) mode.

**pykitPIV** provides images of varying complexity in order to exhaust challenging training scenarios.
The goal of this library is to give the user, or a machine learning algorithm, a lot of flexibility in setting-up
image generation. The generated image pairs, and the associated flow targets, can be directly used in training
convolutional neural networks (CNNs) for optical flow estimation.
The image tensors are compatible with PyTorch and can easily port with convolutional layers
(``torch.nn.Conv2d``) or with convolutional filters (``torch.nn.functional.conv2d``).

The graph below shows the possible workflows constructed from the five main classes:

.. image:: images/pykitPIV-workflow.svg
  :width: 900
  :align: center

For more information on kinematic training of convolutional neural networks (CNNs) using synthetic PIV/BOS images, please
check the following references:

- `Kinematic training of convolutional neural networks for particle image velocimetry <https://iopscience.iop.org/article/10.1088/1361-6501/ac8fae/meta>`_

- `A lightweight neural network designed for fluid velocimetry <https://link.springer.com/article/10.1007/s00348-023-03695-8>`_

- `A lightweight convolutional neural network to reconstruct deformation in BOS recordings <https://link.springer.com/article/10.1007/s00348-023-03618-7>`_

.. image:: images/example-temporal-evolution.gif
  :width: 700
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
   :maxdepth: 5
   :caption: User Guide

   user/particle
   user/flowfield
   user/motion
   user/image
   user/postprocess

.. toctree::
   :maxdepth: 2
   :caption: Quickstart

    quickstart/quickstart-scripts

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & Demos

   tutorials/demo-generate-images
   tutorials/demo-image-statistics
   tutorials/demo-PyTorch-dataloader
   tutorials/demo-data-augmentation
   tutorials/demo-import-external-velocity-field
   tutorials/demo-integrate-with-LIMA

