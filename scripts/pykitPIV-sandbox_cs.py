import numpy as np
import time
#import cmcrameri.cm as cmc
from pykitPIV import Particle, FlowField, Motion, Image


size_buffer = 10
image_height = 128
image_width = 256
time_separation = 1
diameters_min, diameters_max = (6, 8) 
densities_min, densities_max = (0.8, 0.9)
diameter_std = 0.5   #pixel
gaussian_filters_min, gaussian_filters_max = (40, 40.1)
gaussian_filters_min, gaussian_filters_max = (1, 1)

n_gaussian_filter_iter = 1
displacement_min, displacement_max = (2, 3)
n_steps = 10
exposures_min, exposures_max = (0.9, 0.95)
laser_beam_thickness = 1
laser_beam_shape = 0.95
alpha_denominator = 20
random_seed = 100

n_images=1

image_size = (image_height, image_width)

image = Image(random_seed=random_seed)

particles = Particle(n_images, 
                     size=image_size, 
                     size_buffer=size_buffer,
                     diameters=(diameters_min, diameters_max),
                     densities=(densities_min, densities_max),
                     diameter_std=diameter_std,
                     seeding_mode="poisson",
                     #seeding_mode="random",
                     random_seed=random_seed)

particles.plot(0)

image.add_particles(particles)

image.add_reflected_light(exposures=(exposures_min, exposures_max),
                          maximum_intensity=2**16-1,
                          laser_beam_thickness=laser_beam_thickness,
                          laser_beam_shape=laser_beam_shape,
                          alpha=1/alpha_denominator)



plt=image.plot(0,
           instance=1, 
           figsize=(8,8))
plt.show()
print("done")