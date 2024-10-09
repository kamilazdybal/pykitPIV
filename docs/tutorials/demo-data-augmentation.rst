############################################################################################
Data augmentation
############################################################################################

************************************************************
Introduction
************************************************************

In this tutorial, we show how we can apply data augmentation to synthetic PIV image pairs
in order to imitate varying experimental conditions.

.. code:: python

    import numpy as np
    import cmcrameri.cm as cmc
    from pykitPIV import Particle, FlowField, Motion, Image
