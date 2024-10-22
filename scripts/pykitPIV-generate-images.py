########################################################################################################################
## This script generates PIV image pairs and the associated displacement fields.
########################################################################################################################

import numpy as np
import argparse
import time
from pykitPIV import Particle, FlowField, Motion, Image

########################################################################################################################
## Argument parser
########################################################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--n_images',               type=int,       default=10,                     metavar='N')
parser.add_argument('--size_buffer',            type=int,       default=10,                     metavar='b')
parser.add_argument('--image_height',           type=int,       default=256,                    metavar='H')
parser.add_argument('--image_width',            type=int,       default=256,                    metavar='W')
parser.add_argument('--dt',                     type=float,     default=1.0,                    metavar='dt')
parser.add_argument('--diameters',              type=float,     default=[1, 2], nargs="+",      metavar='D')
parser.add_argument('--densities',              type=float,     default=[0.3, 0.31], nargs="+", metavar='rho')
parser.add_argument('--diameter_std',           type=float,     default=0.5,                    metavar='D_std')
parser.add_argument('--gaussian_filters',       type=float,     default=[40, 40.1], nargs="+",  metavar='GF')
parser.add_argument('--n_gaussian_filter_iter', type=int,       default=10,                     metavar='n_GF_iter')
parser.add_argument('--displacement',           type=float,     default=[2.0, 3.0], nargs="+",  metavar='disp')
parser.add_argument('--n_steps',                type=int,       default=10,                     metavar='N_STEPS')
parser.add_argument('--exposures',              type=float,     default=[0.9, 0.95], nargs="+", metavar='EXP')
parser.add_argument('--laser_beam_thickness',   type=float,     default=1,                      metavar='LB-t')
parser.add_argument('--laser_beam_shape',       type=float,     default=0.95,                   metavar='LB-s')
parser.add_argument('--alpha_denominator',      type=float,     default=20,                     metavar='ALPHA')
parser.add_argument('--random_seed',            type=int,       default=100,                    metavar='RS')

args = parser.parse_args()

print(args)

# Populate values:
n_images = vars(args).get('n_images')
size_buffer = vars(args).get('size_buffer')
image_height = vars(args).get('image_height')
image_width = vars(args).get('image_width')
time_separation = vars(args).get('dt')
diameters_min, diameters_max = tuple(vars(args).get('diameters'))
densities_min, densities_max = tuple(vars(args).get('densities'))
diameter_std = vars(args).get('diameter_std')
gaussian_filters_min, gaussian_filters_max = tuple(vars(args).get('gaussian_filters'))
n_gaussian_filter_iter = vars(args).get('n_gaussian_filter_iter')
displacement_min, displacement_max = tuple(vars(args).get('displacement'))
n_steps = vars(args).get('n_steps')
exposures_min, exposures_max = tuple(vars(args).get('exposures'))
laser_beam_thickness = vars(args).get('laser_beam_thickness')
laser_beam_shape = vars(args).get('laser_beam_shape')
alpha_denominator = vars(args).get('alpha_denominator')
random_seed = vars(args).get('random_seed')

image_size = (image_height, image_width)

########################################################################################################################
## Generate images
########################################################################################################################

tic = time.perf_counter()

particles = Particle(n_images, 
                     size=image_size, 
                     size_buffer=size_buffer,
                     diameters=(diameters_min, diameters_max),
                     densities=(densities_min, densities_max),
                     diameter_std=diameter_std,
                     random_seed=random_seed)

flowfield = FlowField(n_images,
                      size=image_size,
                      size_buffer=size_buffer,
                      random_seed=random_seed)

flowfield.generate_random_velocity_field(gaussian_filters=(gaussian_filters_min, gaussian_filters_max),
                                         n_gaussian_filter_iter=n_gaussian_filter_iter,
                                         displacement=(displacement_min, displacement_max))

motion = Motion(particles,
                flowfield,
                time_separation=time_separation)

motion.runge_kutta_4th(n_steps=n_steps)

image = Image(random_seed=random_seed)

image.add_particles(particles)
image.add_flowfield(flowfield)
image.add_motion(motion)
image.add_reflected_light(exposures=(exposures_min, exposures_max),
                          maximum_intensity=2**16-1,
                          laser_beam_thickness=laser_beam_thickness,
                          laser_beam_shape=laser_beam_shape,
                          alpha=1/alpha_denominator)

images_I1 = image.remove_buffers(image.images_I1)
images_I2 = image.remove_buffers(image.images_I2)
images_tensor = image.concatenate_tensors((images_I1, images_I2))

targets_tensor = image.remove_buffers(image.get_displacement_field())

tensors_dictionary = {"I"      : images_tensor, 
                      "target" : targets_tensor}

num, decimal = [part for part in str(time_separation).split('.')]

image.save_to_h5(tensors_dictionary, 
                 filename='pykitPIV-dataset-n-' + str(n_images) + '-' + str(image_height) + '-by-' + str(image_width) + '-dt-' + str(num) + 'p' + str(decimal) + '-rs-' + str(random_seed) + '.h5',
                 verbose=True)

toc = time.perf_counter()
print(f'Images generated and saved in {(toc - tic)/60:0.1f} minutes.\n')

########################################################################################################################