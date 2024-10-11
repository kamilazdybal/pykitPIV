######################################
Overview of classes
######################################

The graph below shows the possible workflows constructed from the five main classes:

.. image:: ../images/pykitPIV-workflow.svg
  :width: 900
  :align: center

- The class **Particle** can be used to initialize particle properties and particle positions on an image.

- The class **FlowField** can be used to create a velocity field to advect the particles.

- The class **Motion** takes an object of class **Particle** and applies an object of class **FlowField** to it to
  advect the particles and generate an image pair tensor, :math:`\mathbf{I} = (I_1, I_2)^{\top}`, at time :math:`t_1` and
  :math:`t_2 = t_1 + \Delta t` respectively, where :math:`\Delta t` denotes the time separation between two PIV images.

- The class **Image** can be used to apply laser and camera properties on any standalone image, as well as on a series of images of advected particles.

- The class **Postprocess** is the endpoint of the workflow and can be used to postprocess a single image or a series of images.

At each stage, the user can enforce reproducible image generation through fixing random seeds.
