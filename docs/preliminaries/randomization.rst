######################################
Randomization of experimental settings
######################################

**pykitPIV** allows for randomization of various experimental settings. The general philosophy is that the
experimental setting which can be randomized across images can be passed as a two-element ``tuple``,
where the first element of the tuple denotes the admissible minimum value
and the second element of the tuple denotes the admissible maximum value.
But it can also be forced to a fixed value across images by passing an ``int`` or a ``float`` instead of a ``tuple``.
Below, we explain the usage of both modes.

***********************
Using randomized values
***********************

For example, if we wanted to generate a spread of seeding particles' diameters
to be any random value between :math:`1 \text{px}` and :math:`4 \text{px}`, across some number of images, we could specify:

.. code:: python

    diameters = (1, 4)

which denotes that the minimum particle diameter should be :math:`1 \text{px}`
and the maximum particle diameter should be :math:`4 \text{px}`.

During image generation, the actual particle diameters for each of the :math:`N` images will be drawn
from a uniform distribution like so:

.. code:: python

    import numpy as np

    # Number of images to be generated:
    n_images = 20

    # Generate a spread of diameters for each of the 20 images:
    diameters_per_image = np.random.rand(n_images) * (diameters[1] - diameters[0]) + diameters[0]

``diameters_per_image`` is a ``numpy`` array that provides a random uniform spread of particle diameters
between :math:`1 \text{px}` and :math:`4 \text{px}`.
If we were to print ``diameters_per_image``, we will see something like this:

.. code-block:: text

    array([3.9681588 , 2.67486177, 3.71224428, 3.67575301, 3.67462814,
           3.81316186, 1.26272335, 3.09337045, 2.75332142, 1.69280529,
           1.41234225, 1.05471252, 1.02885118, 2.17232387, 2.89499995,
           1.04699985, 1.1061524 , 3.15070309, 1.20268356, 3.06636067])

In this case, the first image will have all particles with diameter :math:`3.97 \text{px}`, the second image will have
all particles with diameter :math:`2.67 \text{px}`, and so on...

*********************
Forcing a fixed value
*********************

The same experimental setting, which by default is randomized between some minimum and maximum value, can also be
set to a fixed value for all :math:`N` images.
This can be accomplished by setting that parameter to an ``int`` or a ``float`` instead of a ``tuple``.

For example, when:

.. code:: python

    diameters = 2.0

particle diameters will be exactly :math:`2 \text{px}` for all :math:`N` images.

Of course, this will accomplish the same thing as setting:

.. code:: python

    diameters = (2, 2)

but using a single ``int`` or a ``float`` is a more elegant way
and makes it clearer in the code that the value is fixed.