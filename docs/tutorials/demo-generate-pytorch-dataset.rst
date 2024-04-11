############################################################################################
Generate PyTorch ``Dataset`` with PIV image pairs
############################################################################################

************************************************************
Introduction
************************************************************

In this tutorial, we prepare the synthetic PIV data for training with ``torch.utils.data.DataLoader``.

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

Create datasets
=================================




************************************************************
Create ``Dataset`` class
************************************************************





Our ``PIVDataset`` class inherits after the ``torch.utils.data.Dataset`` class.
We implement three standard methods: ``__init__``, ``__len__``, and ``__getitem__``.

.. code:: python

    class PIVDataset(Dataset):

      def __init__(self, image_IDs):
            self.image_IDs = image_IDs

      def __len__(self):
            return len(self.image_IDs)

      def __getitem__(self, index):

            ID = self.image_IDs[index]

            image = Image()
            sample_dictionary = image.upload_from_h5(filename='data/PyTorch-Dataset-PIV-pairs-' + ID + '.h5')

            X = sample_dictionary['I']
            y = sample_dictionary['targets']

            return X, y

We instantiate an object of the ``PIVDataset`` class:

.. code:: python

    PIV_data = PIVDataset(image_IDs=image_IDs)

Thanks to the ``__len__`` method, we can now execute the ``len()`` command on the object:

.. code:: python

    len(PIV_data)

Also, thanks to the ``__getitem__`` method, we can access the data sample at a given index:

.. code:: python

    PIV_data[10]