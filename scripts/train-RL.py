from pykitPIV.ml import PIVEnv, CameraAgent, Rewards, Cues, plot_trajectory
from pykitPIV.flowfield import compute_q_criterion, compute_divergence
from pykitPIV import ParticleSpecs, FlowFieldSpecs, MotionSpecs, ImageSpecs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cmcrameri.cm as cmc
import numpy as np
import onnxruntime
import tensorflow as tf
import torch
import sys, os
import time
import h5py

# Argparse - - - - - - - - - - - - - - - - - - - - - - - - - 

n_episodes = 1000
n_iterations = 20

epsilon_start = 0.8
discount_factor=0.95

batch_size = 256
n_epochs = 1

memory_size = n_episodes * n_iterations

initial_learning_rate = 0.001
alpha_lr = 0.001

interrogation_window_size = (40,40)

# Specifications for PIV - - - - - - - - - - - - - - - - - -

particle_spec = ParticleSpecs(diameters=(1, 1),
                              distances=(2, 2),
                              densities=(0.4, 0.4),
                              diameter_std=0,
                              seeding_mode='random')

print(particle_spec)

flowfield_spec = FlowFieldSpecs(size=(200, 300),
                                flowfield_type='random smooth',
                                gaussian_filters=(10, 10),
                                n_gaussian_filter_iter=10,
                                displacement=(2, 2))

print(flowfield_spec)

motion_spec = MotionSpecs(n_steps=10,
                          time_separation=1,
                          particle_loss=(0, 0),
                          particle_gain=(0, 0))

print(motion_spec)

image_spec = ImageSpecs(exposures=(0.98, 0.98),
                        maximum_intensity=2**16-1,
                        laser_beam_thickness=1,
                        laser_over_exposure=1,
                        laser_beam_shape=0.95,
                        alpha=1/8,
                        clip_intensities=True,
                        normalize_intensities=False)

print(image_spec)

# Specify cues:
cues_obj = Cues(sample_every_n=10)
cues_function = cues_obj.sampled_vectors

# Specify rewards:
rewards = Rewards(verbose=False)
reward_function = rewards.divergence

def reward_transformation(div):  
    return np.max(np.abs(div))*100

env = PIVEnv(interrogation_window_size=interrogation_window_size,
             interrogation_window_size_buffer=5,
             cues_function=cues_function,
             particle_spec=particle_spec,
             motion_spec=motion_spec,
             image_spec=image_spec,
             flowfield_spec=flowfield_spec,
             inference_model=None,
             random_seed=None)

camera_position, cues = env.reset()

print(cues.shape)

plt = env.render(camera_position,
                 c='white',
                 s=20,
                 lw=1,
                 normalize_cbars=True,
                 cmap=cmc.roma,
                 add_streamplot=True,
                 streamplot_density=3,
                 streamplot_color='k',
                 streamplot_linewidth=0.3,
                 figsize=(10,6), 
                 filename='initial-environment.png')

np.savetxt('velocity-field-u.csv', (env.flowfield.velocity_field[0,0,:,:]), delimiter=',', fmt='%.16e')
np.savetxt('velocity-field-v.csv', (env.flowfield.velocity_field[0,1,:,:]), delimiter=',', fmt='%.16e')

# Train the RL agent - - - - - - - - - - - - - - - - - - - - 

kernel_initializer = tf.keras.initializers.RandomUniform
n_decay_steps = int(n_episodes/1.5)

def epsilon_exp_decay(epsilon_start, iter_count, n=1000):
    return epsilon_start/np.exp(iter_count/(n))

def decayed_learning_rate(step, initial_learning_rate, alpha, decay_steps):
    
    step = np.min([step, decay_steps])
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    
    return initial_learning_rate * decayed

class QNetwork(tf.keras.Model):
    
    def __init__(self, n_actions, kernel_initializer):
        
        super(QNetwork, self).__init__()
        
        self.dense1 = tf.keras.layers.Dense(10, activation='linear', kernel_initializer=kernel_initializer)
        self.dense2 = tf.keras.layers.Dense(8, activation='tanh', kernel_initializer=kernel_initializer)
        self.output_layer = tf.keras.layers.Dense(n_actions, activation='linear', kernel_initializer=kernel_initializer)

    def call(self, state):
        
        x = self.dense1(state)
        x = self.dense2(x)
        
        return self.output_layer(x)

ca = CameraAgent(env=env,
                 target_q_network=QNetwork(env.n_actions, kernel_initializer),
                 selected_q_network=QNetwork(env.n_actions, kernel_initializer),
                 memory_size=memory_size,
                 batch_size=batch_size,
                 n_epochs=n_epochs,
                 learning_rate=initial_learning_rate,
                 optimizer='RMSprop',
                 discount_factor=discount_factor)

saved_camera_trajectories_H = np.zeros((n_iterations, n_episodes))
saved_camera_trajectories_W = np.zeros((n_iterations, n_episodes))

total_tic = time.perf_counter()

print('- '*50)

tic = time.perf_counter()

iter_count = 0
total_rewards = []

for episode in range(0,n_episodes):

    camera_position, cues = ca.env.reset()
    total_reward = 0

    # Before we start training the Q-network, only exploration is allowed:
    if len(ca.memory.buffer) >= batch_size:
        # Exploration probability decreases with training time:
        epsilon = epsilon_exp_decay(epsilon_start, iter_count, n=1000)
        iter_count += 1
    else:
        epsilon = 1.0
    
    if (episode+1) % 10 == 0: print(f'Epsilon: {epsilon:0.3f}')
    
    for i in range(0,n_iterations):

        action = ca.choose_action(cues,
                                  epsilon=epsilon)

        next_camera_position, next_cues, reward = ca.env.step(action,
                                                              reward_function=reward_function,
                                                              reward_transformation=reward_transformation,
                                                              verbose=False)

        ca.remember(cues,
                    action,
                    reward,
                    next_cues)

        cues = next_cues
        total_reward += reward

        saved_camera_trajectories_H[i, episode] = next_camera_position[0]
        saved_camera_trajectories_W[i, episode] = next_camera_position[1]

    # Train the Q-network, (but hold off with training until batch_size of samples is collected):
    if len(ca.memory.buffer) >= batch_size:
    
        current_lr = decayed_learning_rate(iter_count, initial_learning_rate, alpha_lr, n_decay_steps)
        ca.train(initial_learning_rate)
    
        if (episode+1) % 1 == 0 :
            ca.update_target_network()

    else:
        print('Not training the Q-network yet...')

    if (episode+1) % 10 == 0:
        toc = time.perf_counter()
        print(f"Episode: {episode + 1}, Total Reward: {total_reward:0.1f}")
        print(f'\tThese episodes took: {(toc - tic):0.1f} sec.')
        tic = time.perf_counter()

    total_rewards.append(total_reward)

total_toc = time.perf_counter()
print(f'\n\nTotal time: {(total_toc - total_tic)/60/60:0.2f} h.\n')

MSE_losses_collected = np.array(ca.MSE_losses).ravel()
plt.figure(figsize=(20,4))
plt.semilogy(MSE_losses_collected)
plt.xlabel('Epoch #', fontsize=20)
plt.ylabel('MSE loss', fontsize=20)
plt.savefig('MSE-losses.png', bbox_inches='tight', dpi=300)

plt.figure(figsize=(20,4))
plt.plot(total_rewards, 'ko--')
plt.savefig('rewards.png', bbox_inches='tight', dpi=300)

# Save the trained Q-network:
ca.target_q_network.save("QNetwork.keras")

# Visualize the learned policy in the training environment:
(_, _, H, W) = ca.env.flowfield.velocity_field_magnitude.shape
(H_adm, W_adm) = ca.env.admissible_observation_space
idx_H = [i for i in range(0, H_adm) if i % 6 == 0]
idx_W = [i for i in range(0, W_adm) if i % 6 == 0]
print(len(idx_H) * len(idx_W))

learned_policy = np.ones((H,W)) * np.nan

for h in idx_H:
    for w in idx_W:

        camera_position = np.array([h, w])
        _, cues = ca.env.reset(imposed_camera_position=camera_position)
        q_values = ca.target_q_network.predict(cues, verbose=0)
        action = np.argmax(q_values)
        learned_policy[h, w] = action

learned_policy = learned_policy[~np.isnan(learned_policy)]
learned_policy = learned_policy.reshape(len(idx_H), len(idx_W))

cluster_colors = cmc.batlow(np.linspace(0, 1, 5))
cmap = ListedColormap(cluster_colors)

plt.figure(figsize=(20,5))
plt.imshow(learned_policy, origin='lower', cmap=cmap, vmin=0, vmax=4)
cbar = plt.colorbar()
cbar.set_ticks([4/5*(i+0.5) for i in range(0,5)])
cbar.set_ticklabels(list(ca.env.action_to_verbose_direction.values()))
plt.xticks([])
plt.yticks([])
plt.savefig('learned-policy.png', bbox_inches='tight', dpi=300)
