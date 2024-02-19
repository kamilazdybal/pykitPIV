##############################
Welcome to ``pykitPIV``!
##############################

**Py** thon **ki** nematic **t** raining for particle image velocimetry (**PIV**)

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

