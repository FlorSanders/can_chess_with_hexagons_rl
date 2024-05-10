"""
This script comprises several Python classes that collectively form an A2C reinforcement learning agent for playing HexChess. 
These classes are integrated with PyTorch and utilize neural networks for decision-making. The script is structured to support features 
like residual blocks, double critics, and delayed updates, enhancing the agent's learning stability and performance.

Classes Defined in the Script:

1) A2CPlayer: 
- Inherits from a generic Player class.
- Responsible for interacting with the HexChess environment.
- Utilizes an A2C model (A2C) to determine moves based on the current state of the board.
- Manages the game state and processes the neural network's output to make game decisions.

2) ResidualBlock (extends nn.Module):
- Implements a standard residual block for use in neural networks.
- Used within the actor and critic networks to facilitate deeper network architectures without degradation in performance, 
which is common due to vanishing gradients in deep networks.

3) Actor and Critic (both extend nn.Module):
- Define the architecture for the actor and critic networks, respectively.
- The actor network outputs action probabilities based on the current state.
- The critic network estimates the value of the state from the game's perspective, guiding how the actor's decisions should be adjusted.
- These classes support configurations with or without residual blocks and can be instantiated for shared or separate feature extraction layers.

4) A2C (extends nn.Module):
- Combines the actor and critic into a cohesive model capable of both evaluating actions and adjusting policy based on the received rewards.
- Can be configured to use a double critic (for stability it considers minimum of both models' value prediction) and 
delayed critic updates (to potentially enhance learning stability).

5) PrioritizedReplayBuffer:
- Manages a replay buffer that stores experience tuples from gameplay, allowing the model to learn from past actions.
- Uses prioritization to replay important experiences more frequently, which can lead to more efficient learning.

6) A2CLearning:
- Orchestrates the training process, including managing the environment, generating episodes, updating the model based on experiences, and handling the replay buffer.
- Utilizes the A2C model to generate actions and learn from the outcomes, employing techniques like Proximal Policy Optimization (PPO) for stable updates.

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


class A2CPlayer(Player):
    name = "A2C Player"

    def __init__(
        self,
        board,
        is_white,
        model_name="a2c_agent",
        is_large=False,
        device="cpu",
        residual_blocks=False,
        shared_feature_extraction=True,
        double_critic=False,
        delayed_critic=False,
        critic_delay_freq=10,
    ):
        # Initialize player class
        super().__init__(board, is_white)

        # Load model
        self.model_name = model_name
        self.color = "white" if is_white else "black"
        cwd = os.getcwd()
        self.model_path = os.path.join(cwd, "assets", "a2c", f"{self.model_name}.pth")
        self.agent = A2C(
            model_path=self.model_path,
            device=device,
            residual_blocks=residual_blocks,
            shared_feature_extraction=shared_feature_extraction,
            double_critic=double_critic,
            delayed_critic=delayed_critic,
            critic_delay_freq=critic_delay_freq,
        )
        self.agent = self.agent.eval()

        # Use environment wrapper for state & action mask functions
        self.env = HexChessEnv(None, None, board=board)

    def get_move(self):
        # Obtain legal moves mask
        action_mask = self.env.get_action_mask(is_white=self.is_white)

        # Evaluate state to get Q-value estimates
        state = self.env.get_state()

        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        state_tensor = state_tensor.permute(
            0, 3, 1, 2
        )  # Adjust dimensions to [batch_size, channels, height, width]

        # Get the action probabilities and value estimate
        if self.agent.double_critic:
            action_probabilities, value_estimate_q1, value_estimate_q2 = self.agent(
                state_tensor
            )
        else:
            action_probabilities, value_estimate = self.agent(state_tensor)

        # apply the action mask
        action_probs = (
            action_probabilities.detach().numpy()
        )  # Convert to NumPy array after detaching
        action_probs = np.reshape(action_probs, action_mask.shape)
        legal_action_probs = action_probs * action_mask

        # Normalize probabilities if not already normalized
        legal_action_probs /= np.sum(legal_action_probs)

        # Sample an action
        action_index = np.argmax(
            legal_action_probs
        )  # Get the index of the maximum probability
        action_tuple = np.unravel_index(action_index, action_mask.shape)

        # Decode action indices to board positions
        index_from, index_to = action_tuple[0], action_tuple[1]
        position_from = self.env.index_to_position(index_from)
        position_to = self.env.index_to_position(index_to)

        # Return move
        return position_from, position_to


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            out_channels
        )  # Consider BatchNorm instead of LayerNorm
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x  # Store the input
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Add the residual connection
        out = self.relu(out)
        return out


####################
# A2C Model
#####################


class Actor(nn.Module):

    def __init__(self, residual_blocks=False):
        super().__init__()

        self.residual_blocks = residual_blocks
        self.initialize_model(residual_blocks)

    def initialize_model(self, residual_blocks):
        # Input Shape - State : [batch_size, 6, 11, 11]
        if residual_blocks:

            self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding="same")
            # Output shape: [batch_size, 16, 11, 11]
            self.layer_norm1 = nn.LayerNorm([16, 11, 11])
            self.residual_block1 = ResidualBlock(16, 16)

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
            # Output shape: [batch_size, 32, 11, 11]
            self.layer_norm2 = nn.LayerNorm([32, 11, 11])
            self.residual_block2 = ResidualBlock(32, 32)

            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
            # Output shape: [batch_size, 64, 11, 11]
            self.layer_norm3 = nn.LayerNorm([64, 11, 11])
            self.residual_block3 = ResidualBlock(64, 64)

            self.flatten = nn.Flatten()
            # Output shape after flatten: [batch_size, 7744] (64*11*11)

        else:

            ######## Shared feature extraction
            self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding="same")
            # Output shape: [batch_size, 16, 11, 11]
            self.layer_norm1 = nn.LayerNorm([16, 11, 11])

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
            # Output shape: [batch_size, 32, 11, 11]
            self.layer_norm2 = nn.LayerNorm([32, 11, 11])

            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
            # Output shape: [batch_size, 64, 11, 11]
            self.layer_norm3 = nn.LayerNorm([64, 11, 11])

            self.flatten = nn.Flatten()
            # Output shape after flatten: [batch_size, 7744] (64*11*11)

        ###### Actor head
        self.actor_fc = nn.Linear(7744, 1024)  # Reduce dimensionality
        self.layer_norm4 = nn.LayerNorm(1024)

        self.action_head = nn.Linear(1024, 8281)  # Expand back to full action space
        # Output shape after action_head: [batch_size, 8281]
        # Action Space (91 * 91)

    def forward(self, X):

        # Shared Feature Extraction
        x = F.relu(self.layer_norm1(self.conv1(X)))
        if self.residual_blocks:
            x = self.residual_block1(x)
        x = F.relu(self.layer_norm2(self.conv2(x)))
        if self.residual_blocks:
            x = self.residual_block2(x)
        x = F.relu(self.layer_norm3(self.conv3(x)))
        if self.residual_blocks:
            x = self.residual_block3(x)
        x = self.flatten(x)

        # Actor
        actor_features = F.relu(self.layer_norm4(self.actor_fc(x)))
        log_action_probs = F.log_softmax(self.action_head(actor_features), dim=-1)

        return log_action_probs


class Critic(nn.Module):

    def __init__(self, residual_blocks=False, double_critic=False):

        super().__init__()
        self.residual_blocks = residual_blocks
        self.double_critic = double_critic
        self.initialize_model1(residual_blocks)
        if double_critic:
            self.initialize_model2(residual_blocks)

    def initialize_model1(self, residual_blocks):
        # Input Shape - State : [batch_size, 6, 11, 11]
        if residual_blocks:

            self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding="same")
            # Output shape: [batch_size, 16, 11, 11]
            self.layer_norm1 = nn.LayerNorm([16, 11, 11])
            self.residual_block1 = ResidualBlock(16, 16)

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
            # Output shape: [batch_size, 32, 11, 11]
            self.layer_norm2 = nn.LayerNorm([32, 11, 11])
            self.residual_block2 = ResidualBlock(32, 32)

            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
            # Output shape: [batch_size, 64, 11, 11]
            self.layer_norm3 = nn.LayerNorm([64, 11, 11])
            self.residual_block3 = ResidualBlock(64, 64)

            self.flatten1 = nn.Flatten()
            # Output shape after flatten: [batch_size, 7744] (64*11*11)

        else:

            ######## Shared feature extraction
            self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding="same")
            # Output shape: [batch_size, 16, 11, 11]
            self.layer_norm1 = nn.LayerNorm([16, 11, 11])

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
            # Output shape: [batch_size, 32, 11, 11]
            self.layer_norm2 = nn.LayerNorm([32, 11, 11])

            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
            # Output shape: [batch_size, 64, 11, 11]
            self.layer_norm3 = nn.LayerNorm([64, 11, 11])

            self.flatten1 = nn.Flatten()
            # Output shape after flatten: [batch_size, 7744] (64*11*11)

        ###### Critic head
        self.critic_fc_q1 = nn.Linear(7744, 1024)  # Reduce dimensionality
        self.layer_norm_q1 = nn.LayerNorm(1024)  # Consider adding LayerNorm here

        self.critic_head_q1 = nn.Linear(1024, 1)  # Expand back to full action space
        # Output shape after value_head: [batch_size, 1]

    def initialize_model2(self, residual_blocks):
        # Input Shape - State : [batch_size, 6, 11, 11]
        if residual_blocks:

            self.conv4 = nn.Conv2d(6, 16, kernel_size=3, padding="same")
            # Output shape: [batch_size, 16, 11, 11]
            self.layer_norm4 = nn.LayerNorm([16, 11, 11])
            self.residual_block4 = ResidualBlock(16, 16)

            self.conv5 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
            # Output shape: [batch_size, 32, 11, 11]
            self.layer_norm5 = nn.LayerNorm([32, 11, 11])
            self.residual_block5 = ResidualBlock(32, 32)

            self.conv6 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
            # Output shape: [batch_size, 64, 11, 11]
            self.layer_norm6 = nn.LayerNorm([64, 11, 11])
            self.residual_block6 = ResidualBlock(64, 64)

            self.flatten2 = nn.Flatten()
            # Output shape after flatten: [batch_size, 7744] (64*11*11)

        else:

            ######## Shared feature extraction
            self.conv4 = nn.Conv2d(6, 16, kernel_size=3, padding="same")
            # Output shape: [batch_size, 16, 11, 11]
            self.layer_norm4 = nn.LayerNorm([16, 11, 11])

            self.conv5 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
            # Output shape: [batch_size, 32, 11, 11]
            self.layer_norm5 = nn.LayerNorm([32, 11, 11])

            self.conv6 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
            # Output shape: [batch_size, 64, 11, 11]
            self.layer_norm6 = nn.LayerNorm([64, 11, 11])

            self.flatten2 = nn.Flatten()
            # Output shape after flatten: [batch_size, 7744] (64*11*11)

        ###### Critic head
        self.critic_fc_q2 = nn.Linear(7744, 1024)  # Reduce dimensionality
        self.layer_norm_q2 = nn.LayerNorm(1024)  # Consider adding LayerNorm here

        self.critic_head_q2 = nn.Linear(1024, 1)  # Expand back to full action space
        # Output shape after value_head: [batch_size, 1]

    def forward(self, X):
        if self.double_critic:

            # Shared Feature Extraction for Q1
            x_q1 = F.relu(self.layer_norm1(self.conv1(X)))
            if self.residual_blocks:
                x_q1 = self.residual_block1(x_q1)
            x_q1 = F.relu(self.layer_norm2(self.conv2(x_q1)))
            if self.residual_blocks:
                x_q1 = self.residual_block2(x_q1)
            x_q1 = F.relu(self.layer_norm3(self.conv3(x_q1)))
            if self.residual_blocks:
                x_q1 = self.residual_block3(x_q1)
            x_q1 = self.flatten1(x_q1)

            # Critic Head Q1
            critic_features_q1 = F.relu(self.layer_norm_q1(self.critic_fc_q1(x_q1)))
            state_value_q1 = self.critic_head_q1(critic_features_q1)

            # Shared Feature Extraction for Q2
            x_q2 = F.relu(self.layer_norm4(self.conv4(X)))
            if self.residual_blocks:
                x_q2 = self.residual_block4(x_q2)
            x_q2 = F.relu(self.layer_norm5(self.conv5(x_q2)))
            if self.residual_blocks:
                x_q2 = self.residual_block5(x_q2)
            x_q2 = F.relu(self.layer_norm6(self.conv6(x_q2)))
            if self.residual_blocks:
                x_q2 = self.residual_block6(x_q2)
            x_q2 = self.flatten2(x_q2)

            # Critic Head Q2
            critic_features_q2 = F.relu(self.layer_norm_q2(self.critic_fc_q2(x_q2)))
            state_value_q2 = self.critic_head_q2(critic_features_q2)

            return state_value_q1, state_value_q2

        else:

            # Shared Feature Extraction for Q1
            x_q1 = F.relu(self.layer_norm1(self.conv1(X)))
            if self.residual_blocks:
                x_q1 = self.residual_block1(x_q1)
            x_q1 = F.relu(self.layer_norm2(self.conv2(x_q1)))
            if self.residual_blocks:
                x_q1 = self.residual_block2(x_q1)
            x_q1 = F.relu(self.layer_norm3(self.conv3(x_q1)))
            if self.residual_blocks:
                x_q1 = self.residual_block3(x_q1)
            x_q1 = self.flatten1(x_q1)

            # Critic Head Q1
            critic_features_q1 = F.relu(self.layer_norm_q1(self.critic_fc_q1(x_q1)))
            state_value_q1 = self.critic_head_q1(critic_features_q1)

            return state_value_q1


class A2C(nn.Module):

    def __init__(
        self,
        model_path=None,
        learning_rate=1e-4,
        device="cpu",
        residual_blocks=False,
        shared_feature_extraction=True,
        double_critic=False,
        delayed_critic=False,
        critic_delay_freq=10,
        PPO=False,
    ):

        super().__init__()

        self.learning_rate = learning_rate
        self.device = device
        self.residual_blocks = residual_blocks
        self.double_critic = double_critic
        self.shared_feature_extraction = shared_feature_extraction
        self.delayed_critic = delayed_critic
        self.PPO = PPO

        if shared_feature_extraction:
            self.initialize_shared_model(residual_blocks)
        else:
            self.actor = Actor(residual_blocks)
            self.actor = self.actor.to(device)
            self.critic = Critic(residual_blocks, double_critic=double_critic)
            self.critic = self.critic.to(device)

        if delayed_critic:
            self.fixed_critic = Critic(residual_blocks, double_critic=double_critic)
            self.fixed_critic = self.fixed_critic.to(device)
            self.fixed_critic.load_state_dict(self.critic.state_dict())
            self.critic_update_counter = 0
            self.critic_delay_freq = critic_delay_freq

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Load model if path is provided
        if model_path:
            self.load_state_dict(torch.load(model_path))
            print("Model loaded from: ", model_path)

    def initialize_shared_model(self, residual_blocks):

        # Input Shape - State : [batch_size, 6, 11, 11]
        if residual_blocks:

            self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding="same")
            # Output shape: [batch_size, 16, 11, 11]
            self.layer_norm1 = nn.LayerNorm([16, 11, 11])
            self.residual_block1 = ResidualBlock(16, 16)

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
            # Output shape: [batch_size, 32, 11, 11]
            self.layer_norm2 = nn.LayerNorm([32, 11, 11])
            self.residual_block2 = ResidualBlock(32, 32)

            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
            # Output shape: [batch_size, 64, 11, 11]
            self.layer_norm3 = nn.LayerNorm([64, 11, 11])
            self.residual_block3 = ResidualBlock(64, 64)

            self.flatten = nn.Flatten()
            # Output shape after flatten: [batch_size, 7744] (64*11*11)

        else:

            ######## Shared feature extraction
            self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding="same")
            # Output shape: [batch_size, 16, 11, 11]
            self.layer_norm1 = nn.LayerNorm([16, 11, 11])

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
            # Output shape: [batch_size, 32, 11, 11]
            self.layer_norm2 = nn.LayerNorm([32, 11, 11])

            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
            # Output shape: [batch_size, 64, 11, 11]
            self.layer_norm3 = nn.LayerNorm([64, 11, 11])

            self.flatten = nn.Flatten()
            # Output shape after flatten: [batch_size, 7744] (64*11*11)

        ###### Actor head
        self.actor_fc = nn.Linear(7744, 1024)  # Reduce dimensionality
        self.layer_norm4 = nn.LayerNorm(1024)

        self.action_head = nn.Linear(1024, 8281)  # Expand back to full action space
        # Output shape after action_head: [batch_size, 8281]
        # Action Space (91 * 91)

        ###### Critic head
        self.critic_fc = nn.Linear(7744, 1024)  # Reduce dimensionality
        self.layer_norm5 = nn.LayerNorm(1024)  # Consider adding LayerNorm here

        self.critic_head = nn.Linear(1024, 1)  # Expand back to full action space
        # Output shape after value_head: [batch_size, 1]

    def forward(self, X, use_fixed_critic=False):
        if (
            self.shared_feature_extraction
        ):  # Check if we are using shared feature extraction
            # Shared Feature Extraction
            x = F.relu(self.layer_norm1(self.conv1(X)))
            if self.residual_blocks:
                x = self.residual_block1(x)
            x = F.relu(self.layer_norm2(self.conv2(x)))
            if self.residual_blocks:
                x = self.residual_block2(x)
            x = F.relu(self.layer_norm3(self.conv3(x)))
            if self.residual_blocks:
                x = self.residual_block3(x)
            x = self.flatten(x)

            # Actor
            actor_features = F.relu(self.layer_norm4(self.actor_fc(x)))
            log_action_probs = F.log_softmax(self.action_head(actor_features), dim=-1)

            # Critic
            critic_features = F.relu(self.layer_norm5(self.critic_fc(x)))
            state_value = self.critic_head(critic_features)

            return log_action_probs, state_value

        else:
            log_action_probs = self.actor(X)

            if use_fixed_critic:
                if self.double_critic:
                    state_value_q1, state_value_q2 = self.fixed_critic(X)
                    return log_action_probs, state_value_q1, state_value_q2
                else:
                    state_value = self.fixed_critic(X)
                    return log_action_probs, state_value

            else:
                if self.double_critic:
                    state_value_q1, state_value_q2 = self.critic(X)
                    return log_action_probs, state_value_q1, state_value_q2
                else:
                    state_value = self.critic(X)
                    return log_action_probs, state_value

    def update(
        self,
        states,
        actions,
        old_log_probs,
        returns,
        dones,
        values,
        advantages,
        next_states,
        importance_weights,
        clip_epsilon=0.2,
        return_td_errors=False,
    ):
        """
        Updates the A2C model parameters based on the sampled minibatch from the prioritized replay buffer.

        Args:
            states (list): Sampled states from the replay buffer.
            actions (list): Sampled actions from the replay buffer.
            log_probs (torch.Tensor): Log probabilities of the sampled actions.
            returns (torch.Tensor): Computed returns for each sampled state-action pair.
            advantages (torch.Tensor): Computed advantages for each sampled state-action pair.
            values (torch.Tensor): Value estimates for the sampled states.
            importance_weights (list): Importance sampling weights for the minibatch.

        Returns:
            float: The total loss after the update.
            float: The policy loss.
            float: The value loss.
            Optional[ndarray]: The TD errors for the minibatch.
        """

        # Convert to tensors
        # returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        state_tensor = torch.FloatTensor(states)
        # advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        state_tensor = state_tensor.permute(
            0, 3, 1, 2
        )  # Adjust dimensions to [batch_size, channels, height, width]
        state_tensor = state_tensor.to(self.device)  # Move to device

        # Normalize advantages (Optional!)   ############## CHECK IF IT HELPS OR NOT!
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ####################
        # Critic Loss
        ####################

        # Check if double critic and Compute the current Q estimates
        if self.double_critic:
            log_action_probs, values_estimate_q1, values_estimate_q2 = self.forward(
                state_tensor
            )
        else:
            log_action_probs, values_estimate = self.forward(state_tensor)

        # Compute Critic Loss
        if self.double_critic:
            td_errors_q1 = returns - values_estimate_q1.squeeze()
            value_loss_q1 = torch.mean(
                td_errors_q1.pow(2)
                * torch.tensor(importance_weights, dtype=torch.float32).to(self.device)
            )

            td_errors_q2 = returns - values_estimate_q2.squeeze()
            value_loss_q2 = torch.mean(
                td_errors_q2.pow(2)
                * torch.tensor(importance_weights, dtype=torch.float32).to(self.device)
            )

            value_loss = value_loss_q1 + value_loss_q2
        else:
            td_errors = returns - values_estimate.squeeze()
            value_loss = torch.mean(
                td_errors.pow(2)
                * torch.tensor(importance_weights, dtype=torch.float32).to(self.device)
            )

        ####################
        # Policy Loss
        ####################

        if self.PPO:

            # Convert Old Log Probs to Tensor
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(
                self.device
            )

            # Calculate Policy Loss
            advantages = advantages.detach()  # Detach advantages before computing loss

            # Get new action log probabilities
            flat_actions_indices = [
                action_from * 91 + action_to for action_from, action_to in actions
            ]  # 91 is the board size (action height and width)
            action_indices_tensor = torch.tensor(
                flat_actions_indices, dtype=torch.long
            ).to(self.device)
            new_action_log_probs = log_action_probs.gather(
                1, action_indices_tensor.unsqueeze(1)
            ).squeeze()

            # Get old action log probabilities
            old_action_log_probs = old_log_probs

            # Calculate the ratio
            ratios = torch.exp(new_action_log_probs / old_action_log_probs)

            # Calculate the surrogate loss
            policy_loss1 = ratios * advantages
            policy_loss2 = (
                torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            )
            policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

        else:

            # Calculate Policy Loss
            advantages = advantages.detach()  # Detach advantages before computing loss

            # Get new action log probabilities
            flat_actions_indices = [
                action_from * 91 + action_to for action_from, action_to in actions
            ]  # 91 is the board size (action height and width)
            action_indices_tensor = torch.tensor(
                flat_actions_indices, dtype=torch.long
            ).to(self.device)
            action_log_probs = log_action_probs.gather(
                1, action_indices_tensor.unsqueeze(1)
            ).squeeze()
            policy_loss = -(action_log_probs * advantages).mean()

        ####################
        # Calculate the Total Loss and Optimize
        ####################

        # Calculate the Total Loss
        total_loss = policy_loss + value_loss

        # Perform backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        ##################
        # Delayed Critic Update
        ##################

        if self.delayed_critic:
            self.critic_update_counter += 1
            if self.critic_update_counter % self.critic_delay_freq == 0:
                self.fixed_critic.load_state_dict(self.critic.state_dict())

        if return_td_errors:
            return (
                total_loss.item(),
                policy_loss.item(),
                value_loss.item(),
                td_errors.detach().cpu().numpy(),
            )
        else:
            return total_loss.item(), policy_loss.item(), value_loss.item()


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, memory_chunk):
        """
        Update the memory with a new experience tuple
        ---
        Args:
        - sars_tuple (tuple): The new experience tuple.
        """

        if len(self.memory) < self.capacity:
            self.memory.append(memory_chunk)
            self.priorities.append(
                max(self.priorities, default=1)
            )  # Use the maximum sampling probability for initialization
            # self.sampling_probs.append(1) #Append 1 for the sampling probability initially
        else:
            self.memory.popleft()
            self.priorities.popleft()
            self.memory.append(memory_chunk)
            self.priorities.append(
                max(self.priorities, default=1)
            )  # Use the maximum sampling probability for initialization
            # self.sampling_probs.append(1) #Append 1 for the sampling probability initially

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

    def sample(
        self,
        batch_size,
        probability_scale=1.0,
        replace_sample=True,
        beta=-1,
        device="mps",
    ):
        """
        Sample a minibatch of samples from the memory
        ---
        Args:
        - batch_size (int): The size of the minibatch.
        - probability_scale (float): The scale for the sampling probabilities. Ranges from 0 to 1, where 0 is uniform sampling and 1 is prioritized sampling.
        - replace_sample (bool): Whether to replace the samples in the memory when sampling.
        - beta (float): The importance sampling weight. We would want it to increase to 1 over time.
        ---
        Returns:
        - minibatch (list): The minibatch of samples.
        - minibatch_indices (list): The indices of the samples in the memory.
        """

        # Actual batch size
        batch_size = min(batch_size, len(self.memory))

        # Get sampling probabilities
        sampling_probs = self.get_probabilities(probability_scale)

        # Get the minibatch indices and minibatch
        minibatch_indices = np.random.choice(
            len(self.memory), batch_size, replace=replace_sample, p=sampling_probs
        )
        minibatch = [self.memory[i] for i in minibatch_indices]

        # Get the importance weights
        importance_weights = self.get_importance(sampling_probs[minibatch_indices])
        importance_weights_tensor = torch.tensor(
            importance_weights, dtype=torch.float32
        ).to(device)

        if beta >= 0:
            importance_weights_tensor = importance_weights_tensor.pow(beta)

        importance_weights = importance_weights_tensor.detach().cpu().numpy()
        return minibatch, minibatch_indices, importance_weights


class A2CLearning:
    def __init__(
        self,
        env,
        actor_critic,
        memory_size=8192,
        device="cpu",
        double_critic=False,
        delayed_critic=False,
        PPO=False,
    ):

        self.actor_critic = actor_critic
        self.actor_critic = self.actor_critic.to(device)
        self.env = env
        self.device = device
        self.double_critic = double_critic
        self.delayed_critic = delayed_critic
        self.PPO = PPO

        # Initialize for Prioritized Memory Replay
        self.memory = PrioritizedReplayBuffer(memory_size)

    def train(
        self,
        n_episodes=1000,
        max_steps=20,
        batch_size=512,
        gamma=0.99,
        sampling_priority_scale=1.0,
        replace_sample=False,
        priority_offset=1e-5,
        update_buffer_after_forward=False,
        importance_sampling_beta_start=-1,
        clip_epsilon=0.2,
    ):
        """
        Train the A2C model

        Args:
            n_episodes (int): The number of episodes to train the model.
            max_steps (int): The maximum number of steps to take in each episode.
            batch_size (int): The size of the minibatch to sample from the memory.
            gamma (float): The discount factor for future rewards.
            sampling_priority_scale (float): The scale for the sampling probabilities. Ranges from 0 to 1, where 0 is uniform sampling and 1 is prioritized sampling.
            replace_sample (bool): Whether to replace the samples in the memory when sampling.
            priority_offset (float): The offset to add to the priorities to avoid zero probabilities.
            update_buffer_after_forward (bool): Whether to update the buffer after the forward pass.
            importance_sampling_beta_start (float): The importance sampling weight. We would want it to increase to 1 over time.

        Returns:
            list: The total rewards for each episode.
            dict: A dictonary of losses containing the total loss, policy loss, and value loss.
        """

        episode_rewards = []
        losses = {}
        losses["total_loss"] = []
        losses["policy_loss"] = []
        losses["value_loss"] = []

        rolling_rewards = deque(maxlen=10)  # Store the last 10 episodes' rewards
        rolling_losses = {
            "total_loss": deque(maxlen=10),
            "policy_loss": deque(maxlen=10),
            "value_loss": deque(maxlen=10),
        }

        if importance_sampling_beta_start >= 0:
            beta = importance_sampling_beta_start
            beta_end = 1
            beta_increment = (beta_end - importance_sampling_beta_start) / n_episodes

        for episode in range(n_episodes):

            # Update the importance sampling beta
            if importance_sampling_beta_start >= 0:
                beta = min(beta_end, beta + beta_increment)

            # Generate a single episode
            ep_rewards = self.generate_episode(max_steps)

            # Sample a minibatch from the memory
            if importance_sampling_beta_start >= 0:
                minibatch, minibatch_indices, importance_weights = self.memory.sample(
                    batch_size,
                    sampling_priority_scale,
                    replace_sample,
                    beta,
                    device=self.device,
                )
            else:
                minibatch, minibatch_indices, importance_weights = self.memory.sample(
                    batch_size,
                    sampling_priority_scale,
                    replace_sample,
                    device=self.device,
                )

            # Unpack the minibatch
            if self.PPO:
                (
                    states,
                    actions,
                    log_action_probs,
                    rewards,
                    next_states,
                    dones,
                    values,
                ) = zip(*minibatch)
            else:
                states, actions, rewards, next_states, dones, values = zip(*minibatch)

            # Compute returns and advantages for the minibatch and update sampling probabilities
            returns, advantages = self.compute_returns(
                rewards, dones, values, gamma, minibatch_indices, priority_offset
            )

            # Update the model
            if update_buffer_after_forward:

                if self.PPO:
                    total_loss, policy_loss, value_loss, td_errors = (
                        self.actor_critic.update(
                            states=states,
                            actions=actions,
                            old_log_probs=log_action_probs,
                            returns=returns,
                            dones=dones,
                            values=values,
                            advantages=advantages,
                            next_states=next_states,
                            importance_weights=importance_weights,
                            clip_epsilon=clip_epsilon,
                            return_td_errors=update_buffer_after_forward,
                        )
                    )
                else:
                    total_loss, policy_loss, value_loss, td_errors = (
                        self.actor_critic.update(
                            states=states,
                            actions=actions,
                            old_log_probs=None,
                            returns=returns,
                            dones=dones,
                            values=values,
                            advantages=advantages,
                            next_states=next_states,
                            importance_weights=importance_weights,
                            clip_epsilon=0,
                            return_td_errors=update_buffer_after_forward,
                        )
                    )

                self.memory.set_priorities(
                    minibatch_indices, td_errors, priority_offset
                )

            else:

                if self.PPO:
                    total_loss, policy_loss, value_loss = self.actor_critic.update(
                        states=states,
                        actions=actions,
                        old_log_probs=log_action_probs,
                        returns=returns,
                        dones=dones,
                        values=values,
                        advantages=advantages,
                        next_states=next_states,
                        importance_weights=importance_weights,
                        clip_epsilon=clip_epsilon,
                        return_td_errors=update_buffer_after_forward,
                    )
                else:
                    total_loss, policy_loss, value_loss = self.actor_critic.update(
                        states=states,
                        actions=actions,
                        old_log_probs=None,
                        returns=returns,
                        dones=dones,
                        values=values,
                        advantages=advantages,
                        next_states=next_states,
                        importance_weights=importance_weights,
                        clip_epsilon=0,
                        return_td_errors=update_buffer_after_forward,
                    )

            # Document the results
            episode_reward = sum(ep_rewards)
            episode_rewards.append(episode_reward)
            losses["total_loss"].append(total_loss)
            losses["policy_loss"].append(policy_loss)
            losses["value_loss"].append(value_loss)

            # Update rolling buffers
            rolling_rewards.append(episode_reward)
            rolling_losses["total_loss"].append(total_loss)
            rolling_losses["policy_loss"].append(policy_loss)
            rolling_losses["value_loss"].append(value_loss)

            # Every 10 episodes, print the average of the collected metrics
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rolling_rewards)
                total_reward = np.sum(rolling_rewards)
                avg_losses = {k: np.mean(v) for k, v in rolling_losses.items()}
                print(
                    f"Episodes {episode-9}-{episode}: Average Reward: {avg_reward:.2f}, Total Episode Reward: {total_reward:.2f},"
                    f"Average Losses: {avg_losses['total_loss']:.4f} -> Policy: {avg_losses['policy_loss']:.4f}, Value: {avg_losses['value_loss']:.4f} "
                )

            # Icrement the beta
            if importance_sampling_beta_start >= 0:
                beta = beta + beta_increment

        return episode_rewards, losses

    def generate_episode(self, max_steps=20):

        # Put model in evaluation mode
        self.actor_critic.eval()

        # Reset the environment
        self.env.reset()

        # Get Current State
        state = self.env.get_state()

        episode_reward = []
        for _ in range(max_steps):

            # Get the legal actions
            action_mask = self.env.get_action_mask()  # Get the action mask

            # Ensure there's at least one legal action
            assert action_mask.any(), "No legal actions available."

            state_tensor = torch.FloatTensor(state).unsqueeze(
                0
            )  # Convert state to tensor
            state_tensor = state_tensor.permute(
                0, 3, 1, 2
            )  # Adjust dimensions to [batch_size, channels, height, width]
            state_tensor = state_tensor.to(self.device)  # Move to device

            # Get the action probabilities and value estimate
            if self.double_critic:
                if self.delayed_critic:
                    log_action_probs, value_estimate_q1, value_estimate_q2 = (
                        self.actor_critic(state_tensor, use_fixed_critic=True)
                    )
                else:
                    log_action_probs, value_estimate_q1, value_estimate_q2 = (
                        self.actor_critic(state_tensor)
                    )
                value_estimate = min(
                    value_estimate_q1, value_estimate_q2
                )  ########### Can Change this to max or average ##########
            else:
                if self.delayed_critic:
                    log_action_probs, value_estimate = self.actor_critic(
                        state_tensor, use_fixed_critic=True
                    )
                else:
                    log_action_probs, value_estimate = self.actor_critic(state_tensor)

            # apply the action mask
            action_probs = (
                log_action_probs.cpu().detach().numpy()
            )  # Convert to NumPy array after detaching
            action_probs = np.reshape(action_probs, action_mask.shape)
            legal_action_probs = action_probs * action_mask

            # Normalize probabilities if not already normalized
            legal_action_probs /= np.sum(legal_action_probs)

            # Sample an action
            legal_action_probs_tensor = torch.tensor(
                legal_action_probs, dtype=torch.float32, requires_grad=False
            )
            legal_action_probs_tensor = legal_action_probs_tensor.to(
                self.device
            )  # Move to device

            action_index = torch.multinomial(
                legal_action_probs_tensor.view(-1), 1
            ).item()
            action_tuple = np.unravel_index(action_index, action_mask.shape)

            # Get Selected Action's Log Probability
            selected_log_prob = log_action_probs.view(-1)[action_index]

            # Decode action indices to board positions
            next_state, reward, done = self.env.step(action_tuple)

            episode_reward.append(reward)

            # Update Memory
            if self.PPO:
                memory_chunk = (
                    state,
                    action_tuple,
                    selected_log_prob,
                    reward,
                    next_state,
                    done,
                    value_estimate.detach().cpu(),
                )
            else:
                memory_chunk = (
                    state,
                    action_tuple,
                    reward,
                    next_state,
                    done,
                    value_estimate.detach().cpu(),
                )

            self.memory.add(memory_chunk)

            # Update the state
            state = next_state

            # Check if the episode is done
            if done:
                break

            # Put the model back in training mode
            self.actor_critic.train()

        return episode_reward

    def compute_returns(
        self, rewards, dones, values, gamma, minibatch_indices, priority_offset
    ):
        """
        Computes the returns for each step in the minibatch, calculates the TD loss,
        and updates the priorities in the replay buffer.

        Args:
            rewards (torch.Tensor): Rewards received from the environment.
            dones (list): List indicating whether each step is a terminal state.
            values (torch.Tensor): Value estimates from the critic.
            gamma (float): Discount factor for future rewards.
            minibatch_indices (list): The indicies of the minibatch in the replay buffer.

        Returns:
            torch.Tensor: The computed returns for each step in the minibatch.
            torch.Tensor: The TD losses for each step in the minibatch.
        """

        # Convert to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)

        next_values = torch.zeros_like(values).to(self.device)

        # Loop through each item and calculate its next value considering terminal states
        for i in range(len(values) - 1):
            if dones[i]:
                next_values[i] = (
                    0.0  # If the current state is terminal, there's no next value
                )
            else:
                next_values[i] = values[i + 1]

        # Calculate TD Target
        td_targets = rewards + (
            gamma
            * next_values
            * (
                1
                - torch.tensor(dones.clone().detach(), dtype=torch.float32).to(
                    self.device
                )
            )
        )
        td_errors = td_targets - values

        # Compute the returns; returns are equivalent to TD Target
        returns = td_targets

        # Update the priorities in the replay buffer
        self.memory.set_priorities(
            minibatch_indices, td_errors.to("cpu"), priority_offset
        )

        # Calculate the advantages; advantages are equivalent to TD Errors
        advantages = td_errors

        return returns, advantages

    def save(self, path=None):
        """Saves the model's state dictionary to the specified path."""
        if path is None:
            print("No path specified. Model will not be saved.")
        torch.save(self.actor_critic.state_dict(), path)
        print(f"Model saved to {path}")
