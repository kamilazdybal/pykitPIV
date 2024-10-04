############################################################################################
Create a **PyTorch** data loader
############################################################################################

************************************************************
Introduction
************************************************************

In this tutorial, we code a ``DataLoader``

.. code:: python

    import numpy as np
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader

************************************************************
Generated **pykitPIV** images
************************************************************

We assume that PIV images have been generated and saved, and are stored in ``path``:

.. code:: python

    path = '../data/pykitPIV-dataset.h5'

.. warning::

    **pykitPIV** PIV images, and the associated targets, are generated assuming that the origin of the coordinate
    system is the lower-left corner (equivalent to setting ``origin='lower'`` in ``plt.imshow``).
    The *bottom* boundary of a PIV image corresponds to ``[0,:]`` rows from the raw ``numpy`` arrays that store, *e.g.*, the image intensities
    or the velocity components.
    The *top* boundary of a PIV image corresponds to ``[-1,:]`` rows from the raw ``numpy`` arrays.

    Whenever the PIV images need to be interpreted with the origin in the upper-left corner
    (equivalent to setting ``origin='upper'`` in ``plt.imshow``), the :math:`v`-component of velocity has to be multiplied by :math:`-1`.
    This is the case when using the generated **pykitPIV** dataset as raw ``numpy`` arrays, such as in an input/output
    to a convolutional neural network (CNN). A CNN processes arrays assuming that *top* boundary is ``[0,:]`` and *bottom*
    boundary is ``[-1,:]``. With this flip, the :math:`v`-component of velocity has to swap sign, such that whatever was a positive
    :math:`v`-component (from the old *bottom* to *top*) now is a negative :math:`v`-component (from the new *top* to *bottom*).

    This preserves the effect that the :math:`v`-component of velocity has on the particles.

    .. image:: ../images/pykitPIV-dataloader-warning.svg
        :width: 700
        :align: center


************************************************************
Create a **pykitPIV** ``Dataset``
************************************************************


Our ``PIVDataset`` class inherits after the ``torch.utils.data.Dataset`` class.
We implement three standard methods: ``__init__``, ``__len__``, and ``__getitem__``.

.. code:: python

    class pykitPIVDatasetFromPath(Dataset):
        """Load pykitPIV-generated dataset"""

        def __init__(self, path, transform=None, n_samples=None, pin_to_ram=False):

            f = h5py.File(path, "r")
            self.data = np.array(f["I"]).astype("float32")
            self.target = np.array(f["targets"]).astype("float32")

            # Multiply the v-component of velocity by -1:
            self.target[:,1,:,:] = -self.target[:,1,:,:]

            if n_samples:
                self.data = self.data[:n_samples]
                self.target = self.target[:n_samples]

            if pin_to_ram:
                self.data = np.array(self.data)
                self.target = np.array(self.target)
                f.close()

            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):

            if torch.is_tensor(idx):
                idx = idx.tolist()

            sample = self.data[idx], self.target[idx]

            if self.transform:
                sample = self.transform(sample)

            return sample



We instantiate an object of the ``PIVDataset`` class:

.. code:: python

    PIV_data = PIVDataset(image_IDs=image_IDs)

Thanks to the ``__len__`` method, we can now execute the ``len()`` command on the object:

.. code:: python

    len(PIV_data)

Also, thanks to the ``__getitem__`` method, we can access the data sample at a given index:

.. code:: python

    PIV_data[10]

************************************************************************
Create a **pykitPIV** ``DataLoader`` with train and test samples
************************************************************************

.. code:: python

    def get_train_test_loader(args):

        # Data loader
        transform = transforms.Compose([
            datatransform.RandomAffine(degrees=17, translate=(0.2, 0.2), scale=(0.9, 2.0)),
            datatransform.RandomHorizontalFlip(),
            datatransform.RandomVerticalFlip(),
            datatransform.ToTensor(),
            datatransform.NormalizeBounded(bit_depth=16),
            datatransform.RandomBrightness(factor=(0.5, 2)),
            datatransform.RandomNoise(std=(0, args.noise_std)),
        ])

        transformref = transforms.Compose([
                datatransform.ToTensor(),
                datatransform.NormalizeBounded(bit_depth=16),
           ])

        # Create train, test, reference datasets:
        train_dataset = pykitPIVDatasetFromPath(path = args.dataset_train_test,
                                                transform=transform)

        test_dataset = pykitPIVDatasetFromPath(path = args.dataset_train_test,
                                               transform=transform)

        ref_dataset = pykitPIVDatasetFromPath(path = args.dataset_referece,
                                              transform=transformref)

        # Create data loaders:
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)

        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size)

        ref_loader = DataLoader(ref_dataset,
                                batch_size=args.batch_size)

        return train_loader, test_loader, ref_loader