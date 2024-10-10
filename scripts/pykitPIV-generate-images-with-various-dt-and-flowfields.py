import numpy as np
import argparse
import time
from pykitPIV import Particle, FlowField, Motion, Image

#################################################################################################################################
## Argument parser
#################################################################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--n_images',           type=int,  default=83, metavar='NIMAGES', help='Number of images')
parser.add_argument('--size_buffer',        type=int,  default=80, metavar='SIZEBUFFER', help='Image buffer size')
parser.add_argument('--image_height',       type=int,  default=100, metavar='H', help='Image height')
parser.add_argument('--image_width',        type=int,  default=100, metavar='W', help='Image width')

args = parser.parse_args()

print(args)

# Populate values:
n_images = vars(args).get('n_images')
size_buffer = vars(args).get('size_buffer')
image_height = vars(args).get('image_height')
image_width = vars(args).get('image_width')

image_size = (image_height, image_width)

#################################################################################################################################
## Generate images
#################################################################################################################################

tic = time.perf_counter()

time_separations = [0.001,0.01,0.1,0.5,1,1.5,2,3,4,5,6,7,8,9,10]

for i, dt in enumerate(time_separations):

    print('Generating images for time separation of ' + str(dt) + 's...')

    for ff in [1,2,3,4]:

        particles = Particle(n_images,
                             size=image_size,
                             size_buffer=size_buffer,
                             diameters=(2, 4),
                             distances=(1, 2),
                             densities=(0.05, 0.06),
                             diameter_std=0.5,
                             seeding_mode='random',
                             random_seed=100)

        flowfield = FlowField(n_images,
                              size=image_size,
                              size_buffer=size_buffer,
                              random_seed=100)

        if ff == 1:

            flowfield.generate_random_velocity_field(gaussian_filters=(10, 11),
                                                     n_gaussian_filter_iter=20,
                                                     displacement=(0, 10))

        elif ff == 2:

            flowfield.generate_checkered_velocity_field(displacement=(0, 10),
                                                        m=6,
                                                        n=6)

        elif ff == 3:

            flowfield.generate_chebyshev_velocity_field(displacement=(0,10),
                                                        order=30)

        elif ff == 4:

            flowfield.generate_spherical_harmonics_velocity_field(displacement=(0,10),
                                                                  degree=10,
                                                                  order=10)

        motion = Motion(particles,
                        flowfield,
                        time_separation=dt)

        motion.runge_kutta_4th(n_steps=10)

        image = Image(random_seed=100)

        image.add_particles(particles)
        image.add_flowfield(flowfield)
        image.add_motion(motion)
        image.add_reflected_light(exposures=(0.6, 0.65),
                                  maximum_intensity=2 ** 16 - 1,
                                  laser_beam_thickness=1,
                                  laser_over_exposure=1,
                                  laser_beam_shape=0.95,
                                  alpha=1 / 10)

        images_I1 = image.remove_buffers(image.images_I1)
        images_I2 = image.remove_buffers(image.images_I2)
        current_images_tensor = image.concatenate_tensors((images_I1, images_I2))
        current_targets_tensor = image.remove_buffers(image.get_displacement_field())

        if i == 0 and ff == 1:

            images_tensor = current_images_tensor
            targets_tensor = current_targets_tensor

        else:

            images_tensor = np.concatenate((images_tensor, current_images_tensor), axis=0)
            targets_tensor = np.concatenate((targets_tensor, current_targets_tensor), axis=0)

tensors_dictionary = {"I"      : images_tensor, 
                      "target" : targets_tensor}

image.save_to_h5(tensors_dictionary, 
                 filename='pykitPIV-dataset-' + str(n_images*len(time_separations)*4) + '-PIV-pairs-' + str(image_height) + '-by-' + str(image_width) + '-various-dt-and-flowfields.h5',
                 verbose=True)

toc = time.perf_counter()
print(f'Images generated and saved in {(toc - tic)/60:0.1f} minutes.\n')

#################################################################################################################################