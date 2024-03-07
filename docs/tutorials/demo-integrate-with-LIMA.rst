############################################################################################
Integrate synthetic image generation with training a convolutional neural network (CNN)
############################################################################################

************************************************************
Introduction
************************************************************





************************************************************
Generate synthetic images with ``pykitPIV``
************************************************************

Below, we define a function for generating train and test PIV image pairs.
Training dataset can be generated with a different random seed than test dataset.

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
                              flow_mode='random',
                              gaussian_filters=(10,11),
                              n_gaussian_filter_iter=20,
                              sin_period=(30,300),
                              displacement=(0,10),
                              random_seed=random_seed)

        # Instantiate an object of the Motion class:
        motion = Motion(particles,
                        flowfield,
                        time_separation=0.1)

        # Instantiate an object of the Image class:
        image = Image(random_seed=random_seed)

        # Prepare images - - - - - - - - - - - - - - - - - -

        image.add_particles(particles)

        motion.forward_euler(n_steps=10)

        image.add_motion(motion)

        image.add_reflected_light(exposures=(0.6,0.65),
                                  maximum_intensity=2**16-1,
                                  laser_beam_thickness=1,
                                  laser_over_exposure=1,
                                  laser_beam_shape=0.95,
                                  alpha=1/10)

        return image


Training set
======================





Testing set
======================





