import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import gym

# Define the mean and covariance matrix for the Multivariate Normal distribution
mean = torch.tensor([0.0, 0.0, 0.0])
covariance_matrix = torch.tensor([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]])

# Define the neural network architecture for the policy
class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        action_values = self.tanh(x)
        return action_values

# Define the neural network architecture for the forward and inverse models
class ForwardModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(ForwardModel, self).__init__()

        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, state_size)

    def forward(self, input):
        x = torch.relu(self.fc1(input))
        output = self.fc2(x)
        return output

class InverseModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(InverseModel, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state, next_state):
        x = torch.relu(self.fc1(state))
        output = self.fc2(x)
        return output

# Define the ICM algorithm
class ICM:
    def __init__(self, state_size, action_size, hidden_size, eta):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.eta = eta

        self.forward_model = ForwardModel(state_size, action_size, hidden_size)
        self.inverse_model = InverseModel(state_size, action_size, hidden_size)
        self.policy_net = PolicyNet(state_size, action_size, hidden_size)

        self.forward_loss_fn = nn.MSELoss()
        self.inverse_loss_fn = nn.CrossEntropyLoss()

        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=0.001)
        self.inverse_optimizer = optim.Adam(self.inverse_model.parameters(), lr=0.001)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

    def train(self, state, action, next_state, reward):
        # Compute the intrinsic reward and train the models
        forward_prediction = self.forward_model(torch.tensor(np.concatenate([state, action], axis=0), dtype=torch.float32))
        intrinsic_reward = self.eta * self.forward_loss_fn(forward_prediction, torch.tensor(next_state, dtype=torch.float32))

        # Train the forward model
        forward_loss = self.forward_loss_fn(forward_prediction, torch.tensor(next_state, dtype=torch.float32))
        self.forward_optimizer.zero_grad()
        forward_loss.backward()
        self.forward_optimizer.step()

        # Train the inverse model
        inverse_prediction = self.inverse_model(torch.tensor(state, dtype=torch.float32), torch.tensor(next_state, dtype=torch.float32))
        inverse_loss = self.inverse_loss_fn(inverse_prediction.unsqueeze(0), torch.tensor([np.argmax(action)], dtype=torch.long))
        self.inverse_optimizer.zero_grad()
        inverse_loss.backward()
        self.inverse_optimizer.step()

        intrinsic_rewards = np.ones_like(reward) * intrinsic_reward.detach().numpy()
        total_rewards = reward + intrinsic_rewards

        # Train the policy using the total rewards
        prediction = self.policy_net(torch.tensor(state, dtype=torch.float32)) * 0.05
        action_covariance = torch.eye(self.action_size)  # Assuming a diagonal covariance matrix
        multivariate_normal = distributions.MultivariateNormal(prediction, action_covariance)
        log_prob = multivariate_normal.log_prob(multivariate_normal.sample())
        
        policy_loss = log_prob * total_rewards
        #policy_loss = -prediction.log_prob(torch.tensor(action, dtype=torch.float32)) * total_rewards

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return forward_loss.detach().numpy(), inverse_loss.detach().numpy(), policy_loss.detach().numpy(), total_rewards
