.. image:: images/pykitPIV-logo-large.svg
    :width: 700
    :align: center

**pykitPIV** is a Python package that provides rich and reproducible virtual environments
for training machine learning (ML) algorithms in velocimetry.

**pykitPIV** gives the user, or the ML agent, a lot of flexibility in selecting
various parameters that would normally be available in an experimental setting, such as seeding density,
properties of the laser plane, camera exposure, particle loss, or experimental noise.
We also provide an atlas of challenging synthetic velocity fields from analytic formulations,
including **compressible** and **incompressible** (potential) flows.
The effects of particle drift and diffusion in stationary isotropic turbulence can also be added atop the
velocity fields using the simplified Langevin model (SLM). In addition, a variational approach can
connect with the real wind tunnel experimentation and use a generative model to sample new training data
that belong to the distribution of the specific experimental setting.
This can help in re-training ML agents whenever there is a need to extend their range of operation
to new experimental scenarios.

**pykitPIV** can operate in particle image velocimetry (PIV) or background-oriented Schlieren (BOS) mode.
It exploits the idea that if the time separation between two experimental images is small, the
kinematic relationship between two consecutive images is sufficient to determine particle displacement fields.

The graph below shows the overview of functionalities and possible workflows constructed
from the five main classes that mimic an experimental setup and act as a *virtual wind tunnel*.
The ML module (``pykitPIV.ml``) connects with each of the five main classes and provides integrations
with algorithms such as **convolutional neural networks**, **reinforcement learning**,
**variational and generative approaches**, and **active learning**.
ML agents have freedom in interacting with the virtual experiment,
can assimilate data from a real experiment, and can be trained to perform a variety of
tasks using diverse custom-built rewards and state representations.

.. image:: images/pykitPIV-modules.svg
    :width: 900
    :align: center

Kamila Zdybał is **pykitPIV**'s primary developer and point of contact and can be contacted by email at ``kamila.zdybal@empa.ch``.
Other authors who contributed to **pykitPIV** are Claudio Mucignat, Stephan Kunz, and Ivan Lunati.

If you use **pykitPIV** in your work, we will appreciate the citation to the following paper:

.. code-block:: text

        @article{pykitPIV2025,
        title = "pykitPIV: Rich and reproducible virtual training of machine learning algorithms in velocimetry",
        journal = "SoftwareX",
        volume = "",
        pages = "",
        year = "2025",
        issn = "",
        doi = "",
        url = "https://github.com/kamilazdybal/pykitPIV}",
        author = "Zdybał, K. and Mucignat, C. and Kunz, S. and Lunati, I."
        }

------------------------------------------------------------------------------------------------------------------------

.. toctree::
    :maxdepth: 5
    :caption: Installation

    installation/instruction

.. toctree::
    :maxdepth: 5
    :caption: Preliminaries

    preliminaries/classes
    preliminaries/indexing
    preliminaries/randomization
    preliminaries/setting-spectra

.. toctree::
    :maxdepth: 2
    :caption: Quickstart

    quickstart/quickstart-scripts
    quickstart/gallery

.. toctree::
    :maxdepth: 5
    :caption: User Guide

    user/particle
    user/flowfield
    user/motion
    user/image
    user/postprocess
    user/ml

.. toctree::
    :maxdepth: 2
    :caption: Tutorials

    tutorials/demo-generate-images
    tutorials/demo-image-statistics
    tutorials/demo-PyTorch-dataloader
    tutorials/demo-Gymnasium-Env
    tutorials/demo-data-augmentation
    tutorials/demo-import-external-velocity-field
    tutorials/demo-integrate-with-LIMA

