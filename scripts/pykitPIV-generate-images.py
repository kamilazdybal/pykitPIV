import numpy as np
import argparse
import time
from pykitPIV import Particle, FlowField, Motion, Image

#################################################################################################################################
## Argument parser
#################################################################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--n_images',           type=int,  default=10, metavar='NIMAGES', help='Number of images')
parser.add_argument('--size_buffer',        type=int,  default=10, metavar='SIZEBUFFER', help='Image buffer size')
parser.add_argument('--image_height',       type=int,  default=100, metavar='H', help='Image height')
parser.add_argument('--image_width',        type=int,  default=100, metavar='W', help='Image width')
parser.add_argument('--dt',                 type=float,  default=1.0, metavar='DT', help='Time separation between two image frames')

args = parser.parse_args()

print(args)

# Populate values:
n_images = vars(args).get('n_images')
size_buffer = vars(args).get('size_buffer')
image_height = vars(args).get('image_height')
image_width = vars(args).get('image_width')
time_separation = vars(args).get('dt')

image_size = (image_height, image_width)

#################################################################################################################################
## Generate images
#################################################################################################################################

tic = time.perf_counter()

particles = Particle(n_images, 
                     size=image_size, 
                     size_buffer=size_buffer,
                     diameters=(2,4),
                     distances=(1,2),
                     densities=(0.05,0.06),
                     diameter_std=0.5,
                     seeding_mode='random', 
                     random_seed=100)

flowfield = FlowField(n_images,
                      size=image_size,
                      size_buffer=size_buffer,
                      random_seed=100)

flowfield.generate_random_velocity_field(gaussian_filters=(10,11),
                                         n_gaussian_filter_iter=20,
                                         displacement=(0,10))

motion = Motion(particles, 
                flowfield, 
                time_separation=time_separation)

motion.runge_kutta_4th(n_steps=10)

image = Image(random_seed=100)

image.add_particles(particles)
image.add_flowfield(flowfield)
image.add_motion(motion)
image.add_reflected_light(exposures=(0.6,0.65),
                          maximum_intensity=2**16-1,
                          laser_beam_thickness=1,
                          laser_over_exposure=1,
                          laser_beam_shape=0.95,
                          alpha=1/10)

image.remove_buffers()

images_tensor = image.image_pairs_to_tensor()
targets_tensor = image.targets_to_tensor()

tensors_dictionary = {"I"      : images_tensor, 
                      "targets": targets_tensor}

num, decimal = [part for part in str(time_separation).split('.')]

image.save_to_h5(tensors_dictionary, 
                 filename='pykitPIV-dataset-' + str(n_images) + '-PIV-pairs-' + str(image_height) + '-by-' + str(image_width) + '-dt-' + str(num) + 'p' + str(decimal) + '.h5',
                 verbose=True)

toc = time.perf_counter()
print(f'Images generated and saved in {(toc - tic)/60:0.1f} minutes.\n')

#################################################################################################################################