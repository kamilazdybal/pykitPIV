[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/pykitPIV/badge/?version=latest)](https://pykitpiv.readthedocs.io/en/latest/?badge=latest)
[![GitHub](https://img.shields.io/badge/GitHub-pykitPIV-blue.svg)](https://github.com/kamilazdybal/pykitPIV)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kamilazdybal/pykitPIV/HEAD?urlpath=%2Fdoc%2Ftree%2Fjupyter-notebooks%2F)

# `pykitPIV`: Python kinematic training for particle image velocimetry

<p align="center">
    <img src="docs/images/pykitPIV-banner.png">
</p>

## Introduction

Check out [**pykitPIV**'s documentation](https://pykitpiv.readthedocs.io/en/latest/)!

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
The ML module (`pykitPIV.ml`) connects with each of the five main classes and provides integrations
with algorithms such as **convolutional neural networks**, **reinforcement learning**,
**variational and generative approaches**, and **active learning**.
ML agents have freedom in interacting with the virtual experiment,
can assimilate data from a real experiment, and can be trained to perform a variety of
tasks using diverse custom-built rewards and sensory cues.

<p align="center">
    <img src="docs/images/pykitPIV-modules.svg" width="800">
</p>

## Documentation

Documentation at ReadTheDocs can be found [here](https://pykitpiv.readthedocs.io/en/latest/).

## Citing pykitPIV

If you use **pykitPIV** in your work, we will appreciate the citation to the following paper:

```text
@article{pykitPIV2025,
title = "pykitPIV: A framework for flexible and reproducible virtual training of machine learning models in optical velocimetry",
journal = "SoftwareX",
volume = "",
pages = "",
year = "2025",
issn = "",
doi = "",
url = "https://github.com/kamilazdybal/pykitPIV",
author = "Zdybał, K. and Mucignat, C. and Kunz, S. and Lunati, I."
}
```

## Illustrative examples

Below, we show a few illustrative examples of the possible workflows that can be established with **pykitPIV**.

### Virtual PIV with a compressible/incompressible flow

You can find the complete tutorial here: [demo-pykitPIV-07-generate-temporal-sequence-of-images.ipynb](https://github.com/kamilazdybal/pykitPIV/blob/main/jupyter-notebooks/demo-pykitPIV-07-generate-temporal-sequence-of-images.ipynb).

Below, we show two flow cases:

- an incompressible (potential) flow
- a compressible flow

Notably, the compressible flow leads to clustering of particles in the regions of strongest divergence, 
while the incompressible flow, having zero divergence, cannot lead to clustering of particles.

<p align="center">
    <img src="docs/images/animate-incompressible-PIV.gif" width="600">
</p>

<p align="center">
    <img src="docs/images/animate-compressible-PIV.gif" width="600">
</p>

-------

### Virtual PIV environment for reinforcement learning

You can find the complete tutorial here: [demo-pykitPIV-13-SingleDQN-RL-find-sources-and-sinks.ipynb](https://github.com/kamilazdybal/pykitPIV/blob/main/jupyter-notebooks/demo-pykitPIV-13-SingleDQN-RL-find-sources-and-sinks.ipynb).

The functionalities from the machine learning module can be used to train a reinforcement learning (RL) agent 
to navigate the virtual PIV camera towards sources/sinks in a radial flow. 
The agent can perform one of the five actions:

- Move up
- Move down
- Move right
- Move left
- Stay

on the virtual camera, thereby with each step it moves the virtual PIV camera in 2D by N pixels.

<p align="center">
    <img src="docs/images/PIVEnv.svg" width="800">
</p>

-------

### Variational-generative approach for creating new training samples

You can find the complete tutorial here: [demo-pykitPIV-21-convolutional-variational-autoencoder.ipynb](https://github.com/kamilazdybal/pykitPIV/blob/main/jupyter-notebooks/demo-pykitPIV-21-convolutional-variational-autoencoder.ipynb).

The functionalities from the machine learning module can be used to train a convolutional variational autoencoder (CVAE). 
The trained CVAE model generates new velocity fields (u and v components) that belong to the distribution 
of some experimental data. New PIV snapshots can then be generated with the newly generated velocity fields. 
Hence, this approach can be used to extend the training data for transfer learning and can help adapt 
a machine learning model to the changing experimental conditions. Potentially, the simplified Langevin model (SLM) 
can be added atop the generated flow field samples to mitigate the smoothing effect that the variational autoencoder has.

<p align="center">
    <img src="docs/images/PIVCVAE.svg">
</p>

## Authors and contacts

Kamila Zdybał is **pykitPIV**'s primary developer and point of contact and can be contacted by email at ``kamila.zdybal@empa.ch``.
Other authors who contributed to **pykitPIV** are Claudio Mucignat, Stephan Kunz, and Ivan Lunati.

## Related work

For more information on kinematic training of convolutional neural networks (CNNs) using synthetic PIV images, please
check the following references:

- [Kinematic training of convolutional neural networks for particle image velocimetry](https://iopscience.iop.org/article/10.1088/1361-6501/ac8fae/meta)

- [A lightweight neural network designed for fluid velocimetry](https://link.springer.com/article/10.1007/s00348-023-03695-8)

- [A lightweight convolutional neural network to reconstruct deformation in BOS recordings](https://link.springer.com/article/10.1007/s00348-023-03618-7)
