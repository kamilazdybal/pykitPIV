############################################################################################
Generate PyTorch ``Dataset`` with PIV image pairs
############################################################################################

************************************************************
Introduction
************************************************************




************************************************************
Generate synthetic images with ``pykitPIV``
************************************************************

We define the PIV image sizes (128 px :math:`\times` 128 px)
and we specify a 10 px buffer for the image size:

.. code:: python

    image_size = (128,128)

    size_buffer = 10

Below, we define a function for generating train and test PIV image pairs and the flow targets (velocity components :math:`u` and :math:`v`).

.. code:: python

    def generate_images(n_images, random_seed):

        # Instantiate an object of the Particle class:
        particles = Particle(n_images,
                             size=image_size,
                             size_buffer=size_buffer,
                             diameters=(4,4.1),
                             distances=(1,2),
                             densities=(0.05,0.1),
                             signal_to_noise=(5,20),
                             diameter_std=0.2,
                             seeding_mode='random',
                             random_seed=random_seed)

        # Instantiate an object of the FlowField class:
        flowfield = FlowField(n_images,
                              size=image_size,
                              size_buffer=size_buffer,
                              random_seed=random_seed)

        # Generate random velocity field:
        flowfield.generate_random_velocity_field(gaussian_filters=(10,11),
                                                 n_gaussian_filter_iter=20,
                                                 displacement=(0,10))

        # Instantiate an object of the Motion class:
        motion = Motion(particles,
                        flowfield,
                        time_separation=0.1)

        # Instantiate an object of the Image class:
        image = Image(random_seed=random_seed)

        # Prepare images - - - - - - - - - - - - - - - - - -

        image.add_particles(particles)

        image.add_velocity_field(flowfield)

        motion.forward_euler(n_steps=10)

        image.add_motion(motion)

        image.add_reflected_light(exposures=(0.6,0.65),
                                  maximum_intensity=2**16-1,
                                  laser_beam_thickness=1,
                                  laser_over_exposure=1,
                                  laser_beam_shape=0.95,
                                  alpha=1/10)

        image.remove_buffers()

        return image

Training set
======================

The training set will have 100 image pairs:

.. code:: python

    n_images = 100

We fix a random seed for generating the training set of PIV images:

.. code:: python

    training_random_seed = 100

Call the function that generates image pairs:

.. code:: python

    image_train = generate_images(n_images, training_random_seed)

Finally, we convert the generated images and their corresponding targets to 4-dimensional tensors:

.. code:: python

    image_pairs_train = image_train.image_pairs_to_tensor()
    targets_train = image_train.targets_to_tensor()

Validation set
======================

The validation set will have 10 image pairs:

.. code:: python

    n_images = 10

Testing dataset can be generated with a different random seed than training dataset to assure a diverse inference from the trained model.

.. code:: python

    test_random_seed = 200

Call the function that generates image pairs:

.. code:: python

    image_test = generate_images(n_images, test_random_seed)

Convert the generated images and their targets to 4-dimensional tensors:

.. code:: python

    image_pairs_test = image_test.image_pairs_to_tensor()
    targets_test = image_test.targets_to_tensor()

************************************************************
Create ``Dataset`` class
************************************************************





