.. image:: images/pypiv-logo.png
  :width: 500

--------------------------------------------------------------------------------

###############
Introduction
###############

``pypiv`` is a Python package for synthetic PIV image generation.
The graph below shows the possible workflows constructed from the four main classes talk.
The class ``Particle`` can be used to initialize particle properties and positions on an image.
The class ``FlowField`` can be used to create velocity field to advect the particles.
The class ``Motion`` takes and objects of class ``Particle`` and applies an object of class ``FlowField`` on top of it.
The class ``Image`` is the endpoint of the workflow and can be used to apply laser and camera properties on any static image as well as on the advected particles.

.. image:: images/pypiv-workflow.png
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

