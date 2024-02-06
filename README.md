# Synthetic PIV image generation in Python

To install:

```
python -m pip install .
```


Minimal working example:

```python
from pypiv import Particle, Image

particles = Particle(4, 
                     size=(512,512), 
                     densities=(0.05,0.1),
                     diameters=(10,10),
                     distances=(1,1),
                     seeding_mode='random', 
                     random_seed=100)

image = Image(particles)

image.add_particles()

image.plot(0, 
           cmap='Greys_r',
           figsize=(8,8));

```