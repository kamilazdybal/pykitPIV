##############################
Welcome to ``pykitPIV``!
##############################

**Py**\ thon **ki**\ nematic **t**\ raining for particle image velocimetry (**PIV**)

.. image:: images/pykitPIV-logo.png
  :width: 300

**pykitPIV** is a Python package for synthetic PIV image generation.
The goal of this library is to give the user or a reinforcement learning (RL) agent a lot of flexibility in setting-up image generation.

The graph below shows the possible workflows constructed from the four main classes:

- The class **Particle** can be used to initialize particle properties and particle positions on an image.

- The class **FlowField** can be used to create a velocity field to advect the particles.

- The class **Motion** takes an object of class **Particle** and applies an object of class **FlowField** to it to advect the particles and generate an image pair, :math:`\mathbf{I} = (I_1, I_2)^{\top}`, at time :math:`t` and :math:`t + \Delta t` respectively.

- The class **Image** is the endpoint of the workflow and can be used to apply laser and camera properties on any standalone image, as well as on a series of images of advected particles.

At each stage, the user can enforce reproducible image generation through fixing random seeds.

.. image:: images/pykitPIV-workflow.png
  :width: 500
  :align: center

**pykitPIV** exploits the idea that if the time separation between two PIV images is small,
kinematic relationship between particles is sufficient to determine particle displacement fields.
For more information on kinematic training of convolutional neural networks (CNNs) using synthetic PIV images, please check the following references:

- `Kinematic training of convolutional neural networks for particle image velocimetry <https://iopscience.iop.org/article/10.1088/1361-6501/ac8fae/meta>`_

- `A lightweight neural network designed for fluid velocimetry <https://link.springer.com/article/10.1007/s00348-023-03695-8>`_

- `A lightweight convolutional neural network to reconstruct deformation in BOS recordings <https://link.springer.com/article/10.1007/s00348-023-03618-7>`_

.. image:: images/example-image-I1-I2.gif
  :width: 1100

.. toctree::
   :maxdepth: 5
   :caption: User Guide

   user/particle
   user/flowfield
   user/motion
   user/image

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & Demos

   tutorials/demo-generate-images

