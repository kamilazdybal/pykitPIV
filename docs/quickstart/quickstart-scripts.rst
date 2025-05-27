######################################
Quickstart your image generation
######################################

To quickstart your image generations, we provide a couple of generic Python scripts that can be ran from the terminal
to generate samples of PIV/BOS images. These can be found in:

.. code::

    https://github.com/kamilazdybal/pykitPIV/scripts

The scripts use ``argparse`` and can be ran from the terminal using a command like:

.. code::

    python pykitPIV-generate-images.py --n_images 100 --size_buffer 10 --image_height 256 --image_width 256