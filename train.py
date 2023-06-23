import gym
import numpy as np
import torch
import torch.nn as nn

from icm import ICM
import gym_icub_skin
import time
import yarp

from scipy.ndimage import gaussian_filter

# Define a custom function to apply the Gaussian filter at specified locations
def apply_gaussian_filter(data, center, sigma=20):
    # Create a filter with the desired size (radius)
    filter_size = int(6 * sigma)  # Adjust as needed
    
    win = np.array([center[0] - filter_size / 2, center[0] + 1 + filter_size / 2, center[1] - filter_size / 2, center[1] + 1 + filter_size / 2], dtype=np.uint16)
    filt = data[win[0] : win[1], win[2] : win[3]]

    # Apply the Gaussian filter to the data at the specified center
    filtered_data = gaussian_filter(filt, sigma=sigma, mode='constant')
    
    return filtered_data.sum()

def main():
  # Create the env and the ICM algorithm
  env = gym.make('icub_skin-v0')
  # Define the ICM
  effector_pose_size = 7
  effector_action_size = 3
  gaussian_kernels = 29

  obs_dim = effector_pose_size + gaussian_kernels #env.observation_space.shape[0]
  act_dim = effector_action_size #env.action_space.shape[0]
  hidden_size = 64
  eta = 0.1

  icm = ICM(obs_dim, act_dim, hidden_size, eta)

  num_steps = 1000  # Number of steps for online learning

  forward_losses = []
  inverse_losses = []
  policy_losses = []

  kcenters = np.load('kcenters.npy')

  for step in range(num_steps):
    done = False
    episode_reward = 0

    retry = True
    env.reset_simulation()
    time.sleep(0.5)
    
    while retry:
      retry = False
      #rospy.wait_for_service('/gazebo/reset_simulation')
      #reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
      #reset_simulation()
      
      #env = gym.make('icub_skin-v0')
      env.reset()

      modes = yarp.VectorInt(16)

      for ic in env.icontrol:
        ic.getControlModes(modes.data())
        for i in range(modes.length()):
          if modes[i] == yarp.VOCAB_CM_HW_FAULT:
            retry = True
            modes[i] = yarp.VOCAB_CM_FORCE_IDLE
        ic.setControlModes(modes.data())

        for i in range(modes.length()):
          if modes[i] == yarp.VOCAB_CM_FORCE_IDLE:
            modes[i] = yarp.VOCAB_CM_POSITION
        ic.setControlModes(modes.data())
      
      if retry:
        env.reset_simulation()
        time.sleep(0.5)

    env.set_action_cmd('effector')
    env.icontrol[1].getControlModes(modes.data())
    for i in range(7):
      modes[i] = yarp.VOCAB_CM_POSITION_DIRECT
    env.icontrol[1].setControlModes(modes.data())

    tries = int((1 - np.exp(-0.004 * step)) * 85 + 15)

    state = env.get_obs()
    state = np.concatenate([state['effector_pose'], np.zeros(gaussian_kernels)]) 

    no_reward_cnt = 0

    while True:
      # Select an action from the policy network
      action = icm.policy_net(torch.tensor(state, dtype=torch.float32)).detach().numpy() * 0.05

      # Perform the action in the env+
      delta_action = np.zeros(7)
      delta_action[:3] += action
      observation, reward, done, _ = env.step(delta_action)

      if reward == 0.0:
        no_reward_cnt += 1

      # Initialize an array to store the activations of each kernel
      sigma = 20
      filter_size = int(6 * sigma)
      kernel_activations = []
      touch_data = np.zeros([515, 515])

      for j, t in enumerate(observation['skin']['torso']):
        if t != 0.0:
          t_x, t_y = env.SKIN_COORDINATES[-1][1][j], env.SKIN_COORDINATES[-1][2][j]
          touch_data[t_x, t_y] = t / 255.0

      # Iterate over the centers of the Gaussian kernels
      for center in kcenters:
        # Apply the Gaussian filter at the specified center
        kernel_activation = apply_gaussian_filter(touch_data, center)

        # Append the activation to the array
        kernel_activations.append(kernel_activation)

      next_state = np.concatenate([observation['effector_pose'], kernel_activations]) 

      reward = reward * 0.01
      # Train the ICM algorithm with the collected data
      forward_loss, inverse_loss, policy_loss, total_reward = icm.train(state, action, next_state, reward)

      # Track the losses
      forward_losses.append(forward_loss)
      inverse_losses.append(inverse_loss)
      policy_losses.append(policy_loss)

      state = next_state
      episode_reward += total_reward

      if no_reward_cnt == 10:
        break

    # Print the training progress
    print(f"Step {step+1}/{num_steps} - Episode Reward: {episode_reward:.4f}")

  # Plot the training losses
  # import matplotlib.pyplot as plt
  # plt.plot(forward_losses, label='Forward Loss')
  # plt.plot(inverse_losses, label='Inverse Loss')
  # plt.plot(policy_losses, label='Policy Loss')
  # plt.xlabel('Training Step')
  # plt.ylabel('Loss')
  # plt.legend()
  # plt.show()

# Run the main function
if __name__ == '__main__':
  main()