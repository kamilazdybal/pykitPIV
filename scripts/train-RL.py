# ######################################################################################################################
# Train the RL agent to detect sources and sinks in a virtual wind tunnel environment
# ######################################################################################################################

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

# ######################################################################################################################
# Argparse
# ######################################################################################################################

n_episodes = 4
n_iterations = 40

epsilon_start = 0.8
n_decay_steps_epsilon = n_episodes
discount_factor=0.95

batch_size = 128
n_epochs = 1

memory_size = 1000

initial_learning_rate = 0.001
current_lr = 0.001
alpha_lr = 0.001
n_decay_steps_learning_rate = n_episodes

interrogation_window_size = (40,40)

# ######################################################################################################################
# Specifications for the virtual PIV setup
# ######################################################################################################################

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

# Specify cues: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
cues_obj = Cues(sample_every_n=10)
cues_function = cues_obj.sampled_vectors

# Specify rewards: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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

_, cues = env.reset()

print('\nWe have this many cues within one interrogation window:')
print(env.n_cues)
print()

# ######################################################################################################################
# Train the RL agent
# ######################################################################################################################

kernel_initializer = tf.keras.initializers.RandomUniform

# Exploration probability decay: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def epsilon_exp_decay(epsilon_start, iter_count, n=n_decay_steps_epsilon):
    return epsilon_start/np.exp(iter_count/(n))

exploration_probabilities = []

for i in range(0,n_episodes):

    exploration_probabilities.append(epsilon_exp_decay(epsilon_start,
                                                       i,
                                                       n=n_decay_steps_epsilon))

plt.figure(figsize=(5,2))
plt.plot(exploration_probabilities, c='k', lw=3)
plt.xlabel('Episode number')
plt.savefig('epsilon-decay.png', bbox_inches='tight', dpi=300)

print('Exploration probabilities that will be used:')
print(exploration_probabilities)
print()

# Learning rate decay - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def decayed_learning_rate(step, initial_learning_rate, alpha, n=n_decay_steps_learning_rate):

    step = np.min([step, n])
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / n))
    decayed = (1 - alpha) * cosine_decay + alpha

    return initial_learning_rate * decayed

decayed_learning_rates = []

for i in range(0,n_episodes):

    decayed_learning_rates.append(decayed_learning_rate(i,
                                                        initial_learning_rate=initial_learning_rate,
                                                        alpha=alpha_lr,
                                                        n=n_decay_steps_learning_rate))

plt.figure(figsize=(5,2))
plt.semilogy(decayed_learning_rates, c='k', lw=3)
plt.xlabel('Episode number')
plt.savefig('learning-rate-decay.png', bbox_inches='tight', dpi=300)

print('Learning rates that will be used:')
print(decayed_learning_rates)
print()

# Model for the Q-networks - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class QNetwork(tf.keras.Model):

    def __init__(self, n_actions, kernel_initializer):

        super(QNetwork, self).__init__()

        self.dense1 = tf.keras.layers.Dense(env.n_cues, activation='linear', kernel_initializer=kernel_initializer)
        self.dense2 = tf.keras.layers.Dense(int(env.n_cues/2), activation='tanh', kernel_initializer=kernel_initializer)
        self.dense3 = tf.keras.layers.Dense(int(env.n_cues/3), activation='tanh', kernel_initializer=kernel_initializer)
        self.output_layer = tf.keras.layers.Dense(n_actions, activation='linear', kernel_initializer=kernel_initializer)

    def call(self, state):

        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)

        return self.output_layer(x)

print('Q-network architecture is:')
print(str(env.n_cues) + '-' + str(int(env.n_cues/2)) + '-' + str(int(env.n_cues/3))+ '-' + str(env.n_actions))
print()

# Define the RL agent - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

ca = CameraAgent(env=env,
                 target_q_network=QNetwork(env.n_actions, kernel_initializer),
                 selected_q_network=QNetwork(env.n_actions, kernel_initializer),
                 memory_size=memory_size,
                 batch_size=batch_size,
                 n_epochs=n_epochs,
                 learning_rate=initial_learning_rate,
                 optimizer='RMSprop',
                 discount_factor=discount_factor)

total_tic = time.perf_counter()

print()
print('- '*50)
print('Starting training the RL agent...\n')

tic = time.perf_counter()

iter_count = 0
total_rewards = []

log_every = 1

for episode in range(0, n_episodes):

    camera_position, cues = ca.env.reset(regenerate_flowfield=True)
    total_reward = 0

    # Before we start training the Q-network, only exploration is allowed:
    if len(ca.memory.buffer) >= batch_size:

        # Exploration probability decreases with training time:
        epsilon = epsilon_exp_decay(epsilon_start,
                                    iter_count,
                                    n=n_decay_steps_epsilon)

        # Decay the learning rate:
        current_lr = decayed_learning_rate(iter_count,
                                           initial_learning_rate,
                                           alpha_lr,
                                           n=n_decay_steps_learning_rate)

        iter_count += 1  # Only counts episodes that had Q-network trainings in them

    else:

        epsilon = 1.0

    if (episode) % log_every == 0:

        print(f'Episode: {episode + 1}')
        print(f'Epsilon: {epsilon:0.3f}')
        print('Learning rate: ' + str(current_lr))

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

        # Train the Q-network after each step, (but hold off with training until batch_size of samples is collected):
        if len(ca.memory.buffer) >= batch_size:

            ca.train(current_lr)

    # Synchronize the Q-networks only at the end of each episode:
    ca.update_target_network()

    if (episode) % log_every == 0:

        toc = time.perf_counter()

        print(f"Total Reward: {total_reward:0.1f}")
        print(f'This episode took: {(toc - tic):0.1f} sec.')
        print('- '*15)
        print()

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

# Save the trained Q-network: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
ca.target_q_network.save("QNetwork.keras")

print('- '*50)

# ######################################################################################################################
# Save quantities at the end of training
# ######################################################################################################################

# Render the final environment: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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
                 filename='final-environment.png')

np.savetxt('final-velocity-field-u.csv', (env.flowfield.velocity_field[0,0,:,:]), delimiter=',', fmt='%.16e')
np.savetxt('final-velocity-field-v.csv', (env.flowfield.velocity_field[0,1,:,:]), delimiter=',', fmt='%.16e')

# Visualize the learned policy on the final environment: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
(_, _, H, W) = ca.env.flowfield.velocity_field_magnitude.shape
(H_adm, W_adm) = ca.env.admissible_observation_space
idx_H = [i for i in range(0, H_adm) if i % 6 == 0]
idx_W = [i for i in range(0, W_adm) if i % 6 == 0]

print('Interpolating the learned policy over this many points:')
print(len(idx_H) * len(idx_W))
print()

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

print('Script done!')

# ######################################################################################################################
# End
# ######################################################################################################################