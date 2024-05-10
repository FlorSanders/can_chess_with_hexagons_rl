"""
* Overview:

The provided script outlines an experimental framework (inspired by Alpha Zero's architecture) aimed at enhancing a Reinforcement Learning (RL) agent's performance for Hexagon Chess. 
The core idea behind the experiment is to integrate a memory component that allows the model to remember a sequence of past actions and states.
This memory is hypothesized to enable the model to develop more sophisticated strategies by understanding the sequence of events leading to the current state. 
The architecture includes a backbone model that processes stacked state vectors for n_previous_steps, creating a high-dimensional representation of the state. 
This representation is then fed into actor and critic heads along with action tuples for n_previous_steps. The critic head outputs the value for the given state
and its preceding actions, while the actor head determines the subsequent action. After endless experiments we decided to disregard this model architecture 
as it resulted in massive models that were not only hard to train but also did not show any significant improvement in performance.

* Class Descriptions:

1) ResidualBlock:
- This class implements a standard residual block used in convolutional neural networks. It helps in building deeper networks by alleviating the 
vanishing gradient problem, making it suitable for complex spatial hierarchies in state representations.

2) Episode:
- Manages individual episodes by storing steps, rewards, and the total rewards accumulated over the episode. This facilitates episodic learning 
where the agent learns from entire sequences of actions and outcomes.

3) PrioritizedReplayBufferEpisodes:
- An extension of the traditional replay buffer that not only stores transitions but entire episodes. It prioritizes important episodes for replay, 
potentially accelerating learning by focusing on more informative experiences.

4) StateInput:
- Handles the preprocessing and management of state inputs to the model. It maintains a history of n_previous_states, providing a comprehensive view of the 
state evolution to the model.

5) ActionInput:
- Similar to StateInput, this class manages action inputs by keeping a history of actions taken over n_previous_steps. This historical context can help 
the model in decision-making processes that consider past actions.

6) ResidualBackBone:
- Acts as the backbone of the model, processing input states through several convolutional layers and residual blocks to produce a rich feature representation 
that captures both current and historical information.

7) CriticHead and SimpleCritic:
- These classes form the critic part of the architecture, evaluating the quality of state-action pairs. CriticHead integrates action information directly, 
while SimpleCritic operates only on state information.

8) Actor:
- The actor head of the model, responsible for proposing actions based on the current state and past actions. It uses the processed features from the 
ResidualBackBone to compute the probabilities of possible actions.

9) AdvancedA2CLearning:
Orchestrates the training process, managing interactions with the environment, executing training episodes, updating model parameters, and handling the 
episodic memory buffer. It integrates learning components and manages the exploration-exploitation balance through epsilon-greedy strategies.

"""

import sys
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import torch.distributions as distributions
import logging
from collections import deque


logger = logging.getLogger(__name__)
sys.path.append("..")
from hexchess.players import Player
from .environment import HexChessEnv


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Episode:
    def __init__(self):
        self.steps = []
        self.rewards = []
        self.total_reward = 0
        self.length = 0

    def add_step(self, step, reward):
        self.steps.append(step)
        self.rewards.append(reward)
        self.total_reward += reward
        self.length += 1


class PrioritizedReplayBufferEpisodes:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.length = 0

    def add(self, episode):
        if len(self.memory) < self.capacity:
            self.memory.append(episode)
            self.priorities.append(
                max(self.priorities, default=1)
            )  # Use the maximum sampling probability for initialization
        else:
            self.memory.popleft()
            self.priorities.popleft()
            self.memory.append(episode)
            self.priorities.append(max(self.priorities, default=1))
        self.length = len(self.memory)

    def set_priorities(self, indices, priorities, offset=1e-5):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = abs(priority) + offset

    def get_probabilities(self, probability_scale):
        scaled_probabilities = np.array(self.priorities) ** probability_scale
        return scaled_probabilities / np.sum(scaled_probabilities)

    def get_importance(self, probabilities):
        importance = 1 / len(self.memory) * 1 / probabilities
        importance_normalized = importance / np.max(importance)
        return importance_normalized

    def sample(self, batch_size, probability_scale=1.0, replace_sample=True, beta=-1):

        # Actual batch size
        batch_size = min(batch_size, len(self.memory))

        # Get sampling probabilities
        sampling_probs = self.get_probabilities(probability_scale)

        # Get the minibatch indices and minibatch
        episodes_minibatch_indices = np.random.choice(
            len(self.memory), batch_size, replace=replace_sample, p=sampling_probs
        )
        episodes_minibatch = [self.memory[i] for i in episodes_minibatch_indices]

        # Get the importance weights
        if beta >= 0:
            importance_weights = self.get_importance(
                sampling_probs[episodes_minibatch_indices]
            )
            importance_weights = importance_weights**beta
            return episodes_minibatch, episodes_minibatch_indices, importance_weights
        else:
            return episodes_minibatch, episodes_minibatch_indices


class StateInput:
    def __init__(
        self,
        num_previous_states,
        state_shape=(6, 11, 11),
        start_state=None,
        end_state=None,
    ):

        # Input Shape - State : [channels (6), height (11), width (11)]
        # Output Shape - State : [num_previous_states, channels (6), height (11), width (11)]

        self.num_previous_states = num_previous_states
        self.state_shape = state_shape
        self.states = deque(maxlen=num_previous_states)

        # Initialize the start and end states if provided, else use zeros
        self.start_state = (
            start_state if start_state is not None else torch.zeros(state_shape)
        )
        self.end_state = (
            end_state if end_state is not None else torch.zeros(state_shape)
        )

        # Fill the deque with the start state initially
        for _ in range(num_previous_states):
            self.states.append(self.start_state)

    def update_state(self, new_state, done=False):

        # Convert new_state to a tensor if it isn't already
        if not isinstance(new_state, torch.Tensor):
            new_state = torch.tensor(new_state, dtype=torch.float32)

        # Check if the new state has the correct shape
        new_state = new_state.permute(
            2, 0, 1
        )  # [height, width, channels] --> [channels, height, width]

        # If the episode is done, add new state and fill the rest of the deque with end states
        if done:
            self.states.append(new_state)
            for _ in range(self.num_previous_states - 1):
                self.states.append(self.end_state)
        else:
            self.states.append(new_state)

    def get_state_tensor(self):
        # Stack the states to create a tensor of shape [num_previous_states, channels, height, width]
        return torch.stack(list(self.states), dim=0)

    def reset(self):
        # Clear the state history and fill with start states
        self.states.clear()
        for _ in range(self.num_previous_states):
            self.states.append(self.start_state)


class ActionInput:
    def __init__(
        self, num_previous_actions, action_size=2, start_state=None, end_state=None
    ):

        self.num_previous_actions = num_previous_actions
        self.action_size = action_size
        self.actions = deque(maxlen=num_previous_actions)

        # Initialize the start and end states if provided, else use zeros
        self.start_state = (
            start_state if start_state is not None else torch.zeros(action_size)
        )
        self.end_state = (
            end_state if end_state is not None else torch.zeros(action_size)
        )

        # Initialize with padding values
        for _ in range(num_previous_actions):
            self.actions.append(self.start_state)

    def update_action(self, new_action, done=False):

        # Convert new_state to a tensor if it isn't already
        if not isinstance(new_action, torch.Tensor):
            new_action = torch.tensor(new_action, dtype=torch.float32)

        # If the episode is done, add the new action and fill with end actions
        if done:
            self.actions.append(new_action)
            for _ in range(self.num_previous_actions - 1):
                self.actions.append(self.end_state)
        else:
            self.actions.append(new_action)

    def get_action_tensor(self):
        # Replace None with a tensor filled with zeros of the correct size
        return torch.stack(list(self.actions), dim=0)

    def reset(self):
        # Clear the action history and fill with padding values
        self.actions.clear()
        for _ in range(self.num_previous_actions):
            self.actions.append(self.start_state)


class ResidualBackBone(nn.Module):

    def __init__(
        self,
        input_channels=6,
        hidden_dimensions=256,
        residual_blocks=3,
        model_path=None,
    ):
        super().__init__()

        # Input Shape - State : [batch_size, 6, 11, 11]
        # Output Shape - State : [batch_size, 256, 11, 11]

        # Initial Convolutions
        self.initial_convs = nn.ModuleList()

        current_channels = 32
        while current_channels <= hidden_dimensions:
            self.initial_convs.append(
                nn.Conv2d(
                    input_channels, current_channels, kernel_size=3, padding="same"
                )
            )
            self.initial_convs.append(nn.BatchNorm2d(current_channels))
            self.initial_convs.append(nn.ReLU(inplace=True))

            input_channels = current_channels
            current_channels *= 2

        # Create residual blocks
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_dimensions, hidden_dimensions)
                for _ in range(residual_blocks)
            ]
        )

        if model_path is not None:
            self.load_state_dict(torch.load(model_path))
            print("BackBone loaded from: ", model_path)

    def forward(self, x):
        for layer in self.initial_convs:
            x = layer(x)
        x = self.residual_blocks(x)
        return x


class CriticHead(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=256, action_size=2):

        super().__init__()

        # Convolutional filter 1x1 (input_dim should be the same as the number of channels from the backbone output)
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # New layer for action processing
        self.action_layer = nn.Linear(action_size, hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(
            (hidden_dim * 11 * 11 * 2), hidden_dim
        )  # 11x11xhidden_dim (state) + hidden_dim (action)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Activation function
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x, action):

        # Apply 1x1 convolution and batch normalization
        x = self.relu(self.bn1(self.conv1(x)))

        # Process the action
        action = self.action_layer(action)
        action = action.unsqueeze(-1).unsqueeze(-1)

        # Assuming action tensor is of shape [batch_size, hidden_dim, 1, 1]
        # and we want to repeat it to match [batch_size, hidden_dim, 11, 11]
        action = action.repeat(
            1, 1, 11, 11
        )  # now action has shape [batch_size, hidden_dim, 11, 11]

        # Concatenate the state and action
        x = torch.cat([x, action], dim=1)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        value = self.tanh(self.fc2(x))
        value = value.mean(dim=0)

        return value


class SimpleCritic(nn.Module):
    def __init__(self, input_dim=256, model_path=None):

        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                input_dim * 11 * 11, 32
            ),  # Assuming the output of the backbone is [batch_size, 256, 11, 11]
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

        if model_path is not None:
            self.load_state_dict(torch.load(model_path))
            print("Critic loaded from: ", model_path)

    def forward(self, x):
        return self.fc(x)


class Critic(nn.Module):

    def __init__(
        self,
        input_dim=256,
        hidden_dim=256,
        double_critic=False,
        model_path=None,
        action_size=2,
    ):

        super().__init__()

        self.double_critic = double_critic

        if double_critic:
            self.critic1 = CriticHead(
                input_dim=input_dim, hidden_dim=hidden_dim, action_size=action_size
            )
            self.critic2 = CriticHead(
                input_dim=input_dim, hidden_dim=hidden_dim, action_size=action_size
            )
        else:
            self.critic = CriticHead(
                input_dim=input_dim, hidden_dim=hidden_dim, action_size=action_size
            )

        if model_path is not None:
            self.load_state_dict(torch.load(model_path))
            print("Critic loaded from: ", model_path)

    def forward(self, x, action):
        if self.double_critic:
            return self.critic1(x, action), self.critic2(x, action)
        else:
            return self.critic(x, action)


class Actor(nn.Module):

    def __init__(
        self,
        input_dim=256,
        hidden_dim=1024,
        model_path=None,
        action_size=2,
    ):

        super().__init__()

        # Convolutional filter 1x1 (input_dim should be the same as the number of channels from the backbone output)
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # New layer for action processing
        self.action_layer = nn.Linear(action_size, hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(
            (hidden_dim * 11 * 11 * 2), hidden_dim
        )  # 11x11xhidden_dim (state) + hidden_dim (action)
        self.fc2 = nn.Linear(hidden_dim, 8281)

        # Activation function
        self.relu = nn.ReLU(inplace=True)

        if model_path is not None:
            self.load_state_dict(torch.load(model_path))
            print("Actor loaded from: ", model_path)

    def forward(self, x, action):

        # Apply 1x1 convolution and batch normalization
        x = self.relu(self.bn1(self.conv1(x)))

        # Process the action
        action = self.action_layer(action)
        action = action.unsqueeze(-1).unsqueeze(-1)

        # Assuming action tensor is of shape [batch_size, hidden_dim, 1, 1]
        # and we want to repeat it to match [batch_size, hidden_dim, 11, 11]
        action = action.repeat(1, 1, 11, 11)
        # Concatenate the state and action
        x = torch.cat([x, action], dim=1)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        logits = logits.mean(dim=0)
        max_logits = torch.max(logits, dim=-1, keepdim=True).values
        stable_logits = logits - max_logits
        softmax = torch.exp(stable_logits) / torch.sum(
            torch.exp(stable_logits), dim=-1, keepdim=True
        )
        log_action_probs = torch.log(softmax + 1e-9)

        return log_action_probs


class AdvancedA2CLearning:
    def __init__(
        self,
        env,
        critic,
        backbone,
        actor=None,
        num_previous_states=4,
        epochs=100,
        learning_rate=1e-4,
        episodes_per_epoch=10,
        batch_size=32,
        gamma=0.99,
        memory_size=8192,
        device="cpu",
        max_steps=250,
        simple_critic=False,
    ):

        self.env = env
        self.critic = critic.to(device)
        if actor is not None:
            self.actor = actor.to(device)
        else:
            self.actor = actor
        self.backbone = backbone.to(device)
        self.epochs = epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.memory_size = memory_size
        self.max_steps = max_steps
        self.simple_critic = simple_critic

        self.memory = PrioritizedReplayBufferEpisodes(memory_size)

        self.state_input = StateInput(num_previous_states=num_previous_states)
        self.action_input = ActionInput(num_previous_actions=num_previous_states)

        self.epsilon = 0.01
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        # Combine parameters from both models
        combined_parameters = (
            list(self.critic.parameters())
            + list(self.actor.parameters())
            + list(self.backbone.parameters())
        )

        # Initialize the optimizer with the combined parameters
        self.optimizer = torch.optim.Adam(combined_parameters, lr=learning_rate)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(
        self,
        sampling_prob_scale=1.0,
        replace_samples=True,
        priority_offset=1e-5,
        importance_sampling_beta=-1,
    ):

        episode_rewards = []
        average_losses = []
        average_steps_per_episode = []

        self.initialize_memory()  # Fill the replay buffer with episodes

        for epoch in range(self.epochs):
            epoch_losses = []  # Store the losses for each epoch
            epoch_rewards = []  # Store the rewards for each epoch

            for _ in range(self.episodes_per_epoch):
                episode, step = self.generate_episode()
                self.memory.add(episode)
                average_steps_per_episode.append(step)
                epoch_rewards.append(
                    episode.total_reward
                )  # Store total reward for the episode

            # Calculate the average reward for this epoch
            epoch_avg_reward = np.mean(epoch_rewards)
            episode_rewards.append(epoch_avg_reward)

            # Get the number of mini-batches based on the buffer size
            num_mini_batches = max(1, self.memory.length // self.batch_size)

            for _ in range(num_mini_batches):

                # Get samples from the replay buffer
                if importance_sampling_beta >= 0:
                    minibatch, minibatch_indices, importance_weights = (
                        self.memory.sample(
                            self.batch_size,
                            probability_scale=sampling_prob_scale,
                            replace_sample=replace_samples,
                            beta=importance_sampling_beta,
                        )
                    )
                else:
                    minibatch, minibatch_indices = self.memory.sample(
                        self.batch_size,
                        probability_scale=sampling_prob_scale,
                        replace_sample=replace_samples,
                    )
                    importance_weights = None

                # Update the Critic
                average_loss, td_errors = self.update_critic(
                    minibatch, minibatch_indices, importance_weights
                )
                epoch_losses.append(average_loss)
                self.memory.set_priorities(
                    minibatch_indices, td_errors, offset=priority_offset
                )

            # Calculate the average loss for this epoch
            epoch_avg_loss = np.mean(epoch_losses)
            average_losses.append(
                epoch_avg_loss
            )  # Add the average loss to the overall list
            print(
                f"Epoch {epoch+1}: Average Steps per Episode: {np.mean(average_steps_per_episode):.2f}, "
                f"Average Reward: {epoch_avg_reward:.2f}, Average Loss: {epoch_avg_loss:.4f}"
            )

        return episode_rewards, average_losses

    def generate_episode(self):

        if self.actor is not None:
            self.actor.eval()
            self.backbone.eval()
            self.state_input.reset()  # Reset at the start of processing a new episode
            self.action_input.reset()

        episode = Episode()
        state = self.env.reset()
        step = 0
        state = self.env.get_state()
        done = False

        if self.actor is None:
            for _ in range(self.max_steps):

                action_mask = self.env.get_action_mask()

                # Flatten the action matrix
                flat_indices = np.flatnonzero(action_mask)

                # Randomly choose one of the valid actions
                chosen_index = np.random.choice(flat_indices)

                # Convert the flat index back into two-dimensional indices
                action_tuple = np.unravel_index(chosen_index, action_mask.shape)

                next_state, reward, done = self.env.step(action_tuple)

                # Store the step
                episode.add_step(
                    (state, action_tuple, reward, next_state, done), reward
                )
                state = next_state
                step += 1

                if done:
                    break

            return episode, step
        else:
            for _ in range(self.max_steps):

                if np.random.rand() < self.epsilon:
                    # Flatten the action matrix
                    action_mask = self.env.get_action_mask()
                    flat_indices = np.flatnonzero(action_mask)

                    # Randomly choose one of the valid actions
                    chosen_index = np.random.choice(flat_indices)

                    # Convert the flat index back into two-dimensional indices
                    action_tuple = np.unravel_index(chosen_index, action_mask.shape)
                else:
                    action_mask = self.env.get_action_mask()
                    self.state_input.update_state(state, done)
                    state_tensor = self.state_input.get_state_tensor()
                    action_tensor = self.action_input.get_action_tensor()
                    state_tensor = state_tensor.to(self.device)
                    action_tensor = action_tensor.to(self.device)

                    state_features = self.backbone(state_tensor)
                    log_action_probs = self.actor(state_features, action_tensor)
                    log_action_probs = log_action_probs.view(91, 91)
                    action_probs = torch.exp(log_action_probs)
                    action_probs = action_probs.detach().cpu().numpy()
                    legal_action_probs = action_probs * action_mask
                    sum_probs = np.sum(legal_action_probs) + 1e-6
                    legal_action_probs /= sum_probs
                    # Sample an action
                    legal_action_probs_tensor = torch.tensor(
                        legal_action_probs, dtype=torch.float32, requires_grad=False
                    )

                    legal_action_probs_tensor = legal_action_probs_tensor.to(
                        self.device
                    )
                    legal_action_probs_tensor = legal_action_probs_tensor.view(-1)

                    action_index = torch.multinomial(
                        legal_action_probs_tensor, 1
                    ).item()
                    action_tuple = np.unravel_index(action_index, action_mask.shape)

                # Decode action indices to board positions
                next_state, reward, done = self.env.step(action_tuple)

                # Store the step
                episode.add_step(
                    (state, action_tuple, reward, next_state, done), reward
                )
                state = next_state
                self.action_input.update_action(action_tuple, done)
                step += 1

                if done:
                    break

            if self.actor is not None:
                self.actor.train()
                self.backbone.train()

            self.decay_epsilon()
            return episode, step

    def update_critic(self, minibatch, minibatch_indices, importance_weights=None):

        # Initialize the losses
        critic_losses = []
        actor_losses = []
        total_losses = []
        td_errors = []
        self.critic.train()
        if self.actor is not None:
            self.actor.train()

        for episode in minibatch:
            self.state_input.reset()  # Reset at the start of processing a new episode
            self.action_input.reset()

            # Accumulate state representations for the episode
            value_predictions = []
            log_action_probs = []
            rewards = []
            dones = []

            for step in episode.steps:

                state, action_tuple, reward, next_state, done = step

                self.state_input.update_state(state, done)

                if not self.simple_critic:
                    self.action_input.update_action(action_tuple, done)

                state_tensor = self.state_input.get_state_tensor()

                if not self.simple_critic:
                    action_tensor = self.action_input.get_action_tensor()

                # Assume backbone accepts state tensor shaped as (episode_len, channels, height, width)
                state_tensor = state_tensor.to(self.device)

                if not self.simple_critic:
                    action_tensor = action_tensor.to(self.device)

                # Pass the state tensor through the backbone to get features
                state_features = self.backbone(state_tensor)
                if not self.simple_critic:
                    value_prediction = self.critic(state_features, action_tensor)
                else:
                    value_prediction = self.critic(state_features)

                if self.actor is not None:
                    log_action_prob = self.actor(state_features, action_tensor)
                    action_from, action_to = action_tuple
                    flat_actions_indices = action_from * 91 + action_to
                    log_action_prob = log_action_prob.view(-1)[flat_actions_indices]
                    log_action_prob = log_action_prob.unsqueeze(0)
                    log_action_probs.append(log_action_prob)
                value_predictions.append(value_prediction)
                rewards.append(reward)
                dones.append(done)

            # Convert the list of value predictions to a tensor
            value_predictions = torch.cat(value_predictions, dim=0)

            if self.actor is not None:
                log_action_probs = torch.cat(log_action_probs, dim=0)

            # Convert to tensors
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

            # Compute the returns
            returns = self.compute_returns(rewards, dones)

            # Compute the advantages
            td_error = returns - value_predictions
            td_error = td_error.detach()
            td_error = td_error - td_error.min()
            # td_error = (td_error - td_error.mean()) / (td_error.std() + 1e-5)
            td_errors.append(td_error.cpu().numpy().mean())

            # Compute the critic loss

            if importance_weights is not None:
                critic_loss = torch.mean(
                    td_error.pow(2)
                    * torch.tensor(importance_weights, dtype=torch.float32).to(
                        self.device
                    )
                )
            else:
                critic_loss = torch.mean(td_error.pow(2))

            if self.actor is not None:
                actor_loss = -torch.mean(log_action_probs * td_error.detach())
                actor_losses.append(actor_loss.item())

            # Accumulate the loss
            critic_losses.append(critic_loss.item())

            print("Critic Loss:", critic_loss.item())
            print("Actor Loss:", actor_loss.item())
            if self.actor is not None:
                total_loss = critic_loss + actor_loss
            else:
                total_loss = critic_loss

            total_losses.append(total_loss.item())

            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return np.mean(total_losses), td_errors

    def compute_returns(self, rewards, dones):

        returns = torch.zeros_like(rewards)
        running_return = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + (
                self.gamma * running_return * (1 - dones[t].float())
            )
            returns[t] = running_return

        return returns

    def initialize_memory(self):
        average_steps_per_episode = []
        average_reward = []
        while self.memory.length < self.memory_size:
            episode, step = self.generate_episode()
            self.memory.add(episode)
            average_steps_per_episode.append(step)
            average_reward.append(episode.total_reward)

        print(
            f"Memory initialized with {self.memory.length} episodes. Average steps per episode: {np.mean(average_steps_per_episode):.2f}, Average reward: {np.mean(average_reward):.2f}"
        )

    def save(self, critic_path=None, backbone_path=None, actor_path=None):
        """Saves the model's state dictionary to the specified path."""
        if critic_path is not None:
            torch.save(self.critic.state_dict(), critic_path)
            print(f"Critic saved to {critic_path}")
        if backbone_path is not None:
            torch.save(self.backbone.state_dict(), backbone_path)
            print(f"Backbone saved to {backbone_path}")
        if actor_path is not None:
            torch.save(self.actor.state_dict(), actor_path)
            print(f"Actor saved to {actor_path}")
        if backbone_path is None and critic_path is None and actor_path is None:
            print("No path specified. Model will not be saved.")
