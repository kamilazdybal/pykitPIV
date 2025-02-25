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
import copy as cp
import argparse

# ######################################################################################################################
# Argparse
# ######################################################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--case_name', type=str,
                    default='virtual-PIV', metavar='CASENAME',
                    help='Case name that will be added to filenames')
parser.add_argument('--n_episodes', type=int,
                    default=1000, metavar='NEPISODES',
                    help='Number of episodes')
parser.add_argument('--n_iterations', type=int,
                    default=20, metavar='NITER',
                    help='Number of iterations per episode')
parser.add_argument('--epsilon_start', type=float,
                    default=0.8, metavar='EPSILONSTART',
                    help='Initial exploration probability')
parser.add_argument('--discount_factor', type=float,
                    default=0.95, metavar='GAMMA',
                    help='Discount factor')
parser.add_argument('--batch_size', type=int,
                    default=128, metavar='BATCHSIZE',
                    help='Batch size')
parser.add_argument('--n_epochs', type=int,
                    default=1, metavar='NEPOCHS',
                    help='Number of epochs for training the Q-network on each batch')
parser.add_argument('--memory_size', type=int,
                    default=200, metavar='MEMSIZE',
                    help='Size of the memory bank')
parser.add_argument('--initial_learning_rate', type=float,
                    default=0.001, metavar='LRINIT',
                    help='Initial learning rate')
parser.add_argument('--alpha_lr', type=float,
                    default=0.001, metavar='LRALPHA',
                    help='Alpha for the final learning rate')
parser.add_argument('--sample_every_n', type=int,
                    default=10, metavar='CUESSAMPLE',
                    help='Sample every n points to compute the cues vectors')
parser.add_argument('--normalize_displacement_vectors', type=bool,
                    default=True,
                    action=argparse.BooleanOptionalAction, metavar='NORMALIZEDSINCUES',
                    help='Normalize cues vectors from displacement fields')
parser.add_argument('--interrogation_window_size_buffer', type=int,
                    default=5, metavar='BUFFER',
                    help='Interrogation window buffer size')
parser.add_argument('--interrogation_window_size', type=int,
                    default=[40,40], nargs="+", metavar='SEEDS',
                    help='Interrogation window size')

args = parser.parse_args()

print(args)

# Populate values:
case_name = vars(args).get('case_name')
n_episodes = vars(args).get('n_episodes')
n_iterations = vars(args).get('n_iterations')
epsilon_start = vars(args).get('epsilon_start')
discount_factor = vars(args).get('discount_factor')
batch_size = vars(args).get('batch_size')
n_epochs = vars(args).get('n_epochs')
memory_size = vars(args).get('memory_size')
initial_learning_rate = vars(args).get('initial_learning_rate')
alpha_lr = vars(args).get('alpha_lr')
sample_every_n = vars(args).get('sample_every_n')
normalize_displacement_vectors = vars(args).get('normalize_displacement_vectors')
interrogation_window_size_buffer = vars(args).get('interrogation_window_size_buffer')
H_interrogation_window, W_interrogation_window = tuple(vars(args).get('interrogation_window_size'))

interrogation_window_size = (H_interrogation_window, W_interrogation_window)
n_decay_steps_epsilon = n_episodes
n_decay_steps_learning_rate = n_episodes

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
cues_obj = Cues(sample_every_n=sample_every_n,
                normalize_displacement_vectors=normalize_displacement_vectors)
cues_function = cues_obj.sampled_vectors

# Specify rewards: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
rewards = Rewards(verbose=False)
reward_function = rewards.divergence

def reward_transformation(div):
    return np.max(np.abs(div))*10

# Construct the PIV environment: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

env = PIVEnv(interrogation_window_size=interrogation_window_size,
             interrogation_window_size_buffer=interrogation_window_size_buffer,
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
plt.savefig(case_name + '-epsilon-decay.png', bbox_inches='tight', dpi=300)

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
plt.savefig(case_name + '-learning-rate-decay.png', bbox_inches='tight', dpi=300)

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
                 online_q_network=QNetwork(env.n_actions, kernel_initializer),
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
current_lr = cp.deepcopy(initial_learning_rate)

batch_q_values_collected = np.zeros((1, env.n_actions))

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

    batch_q_values = ca.online_q_network(cues).numpy()
    batch_q_values_collected = np.vstack((batch_q_values_collected, batch_q_values))

    # Synchronize the Q-networks only at the end of each episode:
    if len(ca.memory.buffer) >= batch_size:
        ca.update_target_network()
    
    if (episode) % log_every == 0:

        toc = time.perf_counter()

        print(f"Total Reward: {total_reward:0.1f}")
        print(f'This episode took: {(toc - tic):0.1f} sec.')
        print('- '*15)
        print()

        tic = time.perf_counter()

    total_rewards.append(total_reward)

batch_q_values_collected = batch_q_values_collected[1::,:]
np.savetxt(case_name + '-batches-of-q-values.csv', (batch_q_values_collected), delimiter=',', fmt='%.16e')

total_toc = time.perf_counter()
print(f'\n\nTotal time: {(total_toc - total_tic)/60/60:0.2f} h.\n')

MSE_losses_collected = np.array(ca.MSE_losses).ravel()
plt.figure(figsize=(20,4))
plt.semilogy(MSE_losses_collected)
plt.xlabel('Epoch #', fontsize=20)
plt.ylabel('MSE loss', fontsize=20)
plt.savefig(case_name + '-MSE-losses.png', bbox_inches='tight', dpi=300)

plt.figure(figsize=(20,4))
plt.plot(total_rewards, 'ko--')
plt.savefig(case_name + '-rewards.png', bbox_inches='tight', dpi=300)

plt.figure(figsize=(20,4))
for i in range(0,5):
    plt.plot(batch_q_values_collected[:,i], label='Action ' + str(i+1), c='k')
plt.xlabel('Step #', fontsize=20)
plt.ylabel('Q-value', fontsize=20)
plt.legend(frameon=False)
plt.savefig(case_name + '-Q-values-action-' + str(i+1) + '.png', bbox_inches='tight', dpi=300)

# Save the trained Q-network: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
ca.target_q_network.save(case_name + '-QNetwork.keras')

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

np.savetxt(case_name + '-final-velocity-field-u.csv', (env.flowfield.velocity_field[0,0,:,:]), delimiter=',', fmt='%.16e')
np.savetxt(case_name + '-final-velocity-field-v.csv', (env.flowfield.velocity_field[0,1,:,:]), delimiter=',', fmt='%.16e')

# Visualize the learned policy on the final environment: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
(_, _, H, W) = ca.env.flowfield.velocity_field_magnitude.shape
(H_adm, W_adm) = ca.env.admissible_observation_space
idx_H = [i for i in range(0, H_adm) if i % 3 == 0]
idx_W = [i for i in range(0, W_adm) if i % 3 == 0]

print('Interpolating the learned policy over this many points:')
print(len(idx_H) * len(idx_W))
print()

learned_policy = np.ones((H,W)) * np.nan

for h in idx_H:
    for w in idx_W:

        camera_position = np.array([h, w])
        _, cues = ca.env.reset(imposed_camera_position=camera_position)
        q_values = ca.target_q_network(cues)
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
plt.savefig(case_name + '-learned-policy.png', bbox_inches='tight', dpi=300)

print('Script done!')

# ######################################################################################################################
# End
# ######################################################################################################################