# Synthetic PIV image generation in Python

<p align="center">
    <img src="docs/images/pypiv-logo.png" width="200">
</p>

## Introduction

**pypiv** is a Python package for synthetic PIV image generation.
The goal of this library is to give the user (or a reinforcement learning agent) a lot of flexibility in setting-up image generation.

The graph below shows the possible workflows constructed from the four main classes.
The class **Particle** can be used to initialize particle properties and particle positions on an image.
The class **FlowField** can be used to create velocity field to advect the particles.
The class **Motion** takes and objects of class **Particle** and applies an object of class **FlowField** on top of it.
The class **Image** is the endpoint of the workflow and can be used to apply laser and camera properties on any standalone image,
as well as on the series of images of advected particles. At each stage, we enforce reproducible image generation through fixing random seeds.

<p align="center">
    <img src="docs/images/pypiv-workflow.png" width="400">
</p>

## Installation

To install, run the following in the main ``pypiv/`` location:

```
python -m pip install .
```

## Example PIV image

<p align="center">
    <img src="docs/images/example-image.png" width="300">
</p>