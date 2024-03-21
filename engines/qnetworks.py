import tensorflow as tf
import tensorflow.keras as krs
import numpy as np
import os
import sys

sys.path.append("..")
from hexchess.players import Player
from .environment import HexChessEnv


class QNetworkPlayer(Player):
    name = "DQN Player"

    def __init__(self, board, is_white, model_name="randomgreedy_pr"):
        # Initialize player class
        super().__init__(board, is_white)

        # Load model
        self.model_name = model_name
        self.color = "white" if is_white else "black"
        self.model_path = os.path.join(
            "assets", "qnetworks", f"{self.model_name}_model_{self.color}"
        )
        self.agent = QNetworkAgent(model_path=self.model_path)

        # Use environment wrapper for state & action mask functions
        self.env = HexChessEnv(None, None, init_game=False)
        self.env.board = board

    def get_move(self):
        # Obtain legal moves mask
        action_mask = self.env.get_action_mask(is_white=self.is_white)

        # Evaluate state to get Q-value estimates
        state = self.env.get_state()
        action_values = self.agent.get_action_values(state, is_fixed=False)
        action_values = np.reshape(action_values, action_mask.shape)

        # Get values for legal moves
        legal_action_values = action_values * action_mask

        # Get action that maximizes Q-function
        action = np.argmax(legal_action_values)
        action = np.unravel_index(action, action_mask.shape)
        if not action_mask[action[0], action[1]]:
            # Make sure our code is working
            assert (
                np.max(legal_action_values) == 0
            ), "Illegal move was chosen while legal moves are available"
            # Pick legal action randomly
            print("WARNING: Making random move since no legal moves seemed interesting")
            legal_actions = np.argwhere(action_mask)
            action_index = np.random.choice(legal_actions.shape[0])
            action = legal_actions[action_index]

        # Decode action indices to board positions
        index_from, index_to = action
        position_from = self.index_to_position(index_from)
        position_to = self.index_to_position(index_to)

        # Perform move
        success, finished = self.board.move(
            position_from, position_to, not self.opponent_is_white
        )
        assert success, "Illegal move was chosen"


class QNetworkAgent:
    def __init__(
        self, model_path=None, discount_factor=0.5, learning_rate=1e-3, verbose=False
    ):
        self.model_path = model_path
        self.gamma = discount_factor
        self.lr = learning_rate
        self.verbose = verbose
        self.init_network()

    def init_network(self):
        if self.model_path is None:
            # Model to map state (11 x 11 x 6) to action space value function (91 * 91 = 8281)
            self.model = krs.models.Sequential(
                [
                    krs.layers.Input(shape=(11, 11, 6)),  # 11 x 11 x 6
                    krs.layers.Conv2D(
                        6, (3, 3), activation="relu", padding="same"
                    ),  # 11 x 11 x 6
                    krs.layers.UpSampling2D(size=(3, 3)),  # 33 x 33 x 6
                    krs.layers.Conv2D(
                        3, (3, 3), activation="relu", padding="same"
                    ),  # 33 x 33 x 3
                    krs.layers.UpSampling2D(size=(3, 3)),  # 99 x 99 x 3
                    krs.layers.Conv2D(
                        1, (5, 5), activation="relu", padding="valid"
                    ),  # 95 x 95 x 1
                    krs.layers.Conv2D(
                        1, (5, 5), activation="linear", padding="valid"
                    ),  # 91 x 91 x 1
                    krs.layers.Flatten(),  # 8281
                ]
            )
        else:
            self.model = krs.models.load_model(self.model_path)

        # Compile the model
        self.compile_model(self.model)

    def compile_model(self, model):
        # Compile with optimizer & loss function
        optimizer = krs.optimizers.SGD(learning_rate=self.lr)
        loss_fn = krs.losses.MeanSquaredError()
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=["mae"])

    def fix_model(self):
        # Create a model clone
        self.model_fixed = krs.models.clone_model(self.model)
        self.model_fixed.set_weights(self.model.get_weights())

        # Compile the model clone
        self.compile_model(self.model_fixed)

    def update_network(self, minibatch):
        """
        Update the Q-network with samples from the minibatch
        ---
        """

        # Parse minibatch content
        states, actions, rewards, new_states = list(zip(*minibatch))
        rewards = np.asarray(rewards)
        actions = np.asarray(actions)
        states = np.asarray(states)
        new_states = np.asarray(new_states)

        # Keep track of TD losses
        td_losses = np.zeros_like(rewards)

        # Q predictions for the new states using fixed model
        q_target = rewards + self.gamma * np.max(
            self.get_action_values(new_states, is_fixed=True), axis=-1
        )  # batch_size

        # Q predictions for initial states using updating model
        q_state = self.get_action_values(states, is_fixed=False)
        q_state = np.reshape(q_state, (-1, 91, 91))  # batch_size x 91 x 91

        # For any move we have actually seen, correct the prediction
        for action_index, action in enumerate(actions):
            # TD loss for the current action
            td_losses[action_index] = (
                q_state[action_index, action[0], action[1]] - q_target[action_index]
            )
            # Update the prediction for the current action
            q_state[action_index, action[0], action[1]] = q_target[action_index]
        q_state = np.reshape(q_state, (-1, 91 * 91))  # batch_size x 8281

        # Minibatch gradient descent step
        self.model.fit(x=states, y=q_state, epochs=1, verbose=0)

        # Return minibatch TD losses
        return td_losses

    def get_action_values(self, state, is_fixed=True):
        """
        Get the action values for a given state
        ---
        Args:
        - state (np.ndarray): The state to get the action values for.
        - is_fixed (bool): Whether to use the fixed model. Default is True.

        Returns:
        - action_values (np.ndarray): The action values for the given state.
        """
        model = self.model_fixed if is_fixed else self.model
        action_values = np.array(model(state))
        return action_values


class QLearning:
    def __init__(self, agent, environment, memory_size=8192):
        self.agent = agent
        self.environment = environment

        self.memory_size = memory_size
        self.memory = []
        self.memory_probs = []

    def learn(
        self,
        n_episodes=100,
        model_fix_episodes=10,
        max_episode_length=250,
        batch_size=1024,
    ):
        # Run through the episodes
        episode_rewards = np.zeros(n_episodes)
        all_step_rewards = []
        for episode_index in range(n_episodes):
            # Update model every few episodes
            if episode_index % model_fix_episodes == 0:
                self.agent.fix_model()
                if episode_index > 0:
                    model_fix_reward = np.mean(
                        episode_rewards[
                            episode_index - model_fix_episodes : episode_index
                        ]
                    )
                    print(
                        f"Episodes {episode_index-model_fix_episodes} - {episode_index}: Mean reward {model_fix_reward}"
                    )

            # Run through an episode
            is_epsilon_greedy = episode_index < n_episodes - 1
            step_rewards = self.run_episode(
                episode_index,
                max_length=max_episode_length,
                batch_size=batch_size,
                is_epsilon_greedy=is_epsilon_greedy,
            )
            all_step_rewards.extend(step_rewards)
            episode_reward = np.sum(step_rewards)
            episode_rewards[episode_index] = episode_reward
            # print(f"Episode {episode_index}: reward = {episode_reward} over {len(step_rewards)} moves")

        return episode_rewards, all_step_rewards

    def run_episode(
        self, episode_index, max_length=250, batch_size=1024, is_epsilon_greedy=True
    ):
        """
        Run a single episode of the game
        ---
        Args:
        - episode_index (int): The index of the episode.
        - max_length (int): The maximum length of the episode. Default is 100.
        - batch_size (int): The size of the minibatch. Default is 10.
        - is_epsilon_greedy (bool): Whether to use epsilon-greedy exploration. Default is True.
        ---
        Returns:
        - step_rewards (list): Immediate reward for each step in the episode
        """

        # Reset the environment
        self.environment.reset()

        # Determine the exploration rate (TODO: finetune)
        eic = 1 / 250  # Slow down the exploration rate decay
        epsilon = max(1 / (episode_index * eic + 1), 0.05) if is_epsilon_greedy else 0.0

        # Get the current state
        state = self.environment.get_state()

        # Run the game
        is_finished = False
        step_index = 0
        step_rewards = []
        while not is_finished:
            # Get the mask of legal actions
            action_mask = self.environment.get_action_mask()

            # Pick move (explore vs exploit)
            is_exploring = np.random.rand() < epsilon
            if is_exploring:
                # Pick legal action randomly
                legal_actions = np.argwhere(action_mask)
                action_index = np.random.choice(legal_actions.shape[0])
                action = legal_actions[action_index]
            else:
                # Get legal action from model
                action_values = self.agent.get_action_values(np.expand_dims(state, 0))
                action_values = np.reshape(action_values, action_mask.shape)
                legal_action_values = action_values * action_mask

                # Get action that maximizes value function
                action = np.argmax(legal_action_values)
                action = np.unravel_index(action, action_mask.shape)
                if not action_mask[action[0], action[1]]:
                    assert (
                        np.max(legal_action_values) == 0
                    ), "Illegal move was chosen while legal moves are available"
                    # Pick legal action randomly
                    legal_actions = np.argwhere(action_mask)
                    action_index = np.random.choice(legal_actions.shape[0])
                    action = legal_actions[action_index]

            # Perform the action
            new_state, reward, is_finished = self.environment.step(action)
            step_rewards.append(reward)

            # Update agent
            sars_tuple = (state, action, reward, new_state)
            self.update_agent(sars_tuple, batch_size, step_index)

            # Keep track of game length
            step_index += 1
            if step_index >= max_length:
                is_finished = True

        return step_rewards

    def update_memory(self, sars_tuple):
        """
        Update the memory with a new experience tuple
        ---
        Args:
        - sars_tuple (tuple): The new experience tuple.
        """

        # Make sure memort content stays under the max size
        while len(self.memory) >= self.memory_size - 1:
            self.memory.pop(0)
            self.memory_probs.pop(0)

        # Append new SARS tuple to memory
        self.memory.append(sars_tuple)
        self.memory_probs.append(1)

    def sample_minibatch(self, batch_size):
        """
        Sample a minibatch of samples from the memory
        ---
        Args:
        - batch_size (int): The size of the minibatch.
        ---
        Returns:
        - minibatch (list): The minibatch of samples.
        - minibatch_indices (list): The indices of the samples in the memory.
        """
        # Actual batch size
        batch_size = min(batch_size, len(self.memory))
        # Obtain random entries from our experience history
        probs = np.array(self.memory_probs)
        minibatch_indices = np.random.choice(
            len(self.memory), batch_size, replace=True, p=probs / np.sum(probs)
        )
        minibatch = [self.memory[i] for i in minibatch_indices]

        return minibatch, minibatch_indices

    def update_agent(self, sars_tuple, batch_size, step_index):
        """
        Update the agent with a new experience tuple
        ---
        Args:
        - sars_tuple (tuple): The new experience tuple.
        - batch_size (int): The size of the minibatch to use for updating the agent.
        """

        # Update agent with experience replay
        # 1) Update memory
        self.update_memory(sars_tuple)

        # 2) Check if enough experiences were acquired
        if len(self.memory) < step_index:
            return

        # 3) Sample a minibatch of samples from the memory
        minibatch, minibatch_indices = self.sample_minibatch(batch_size)

        # 4) Update the agent
        td_losses = self.agent.update_network(minibatch)

        # 5) Update memory sampling probabilities
        for index, td_loss in zip(minibatch_indices, td_losses):
            # Update memory probability to be proportional with TD loss
            # (increase chances of picking harder samples)
            self.memory_probs[index] = np.abs(td_loss)
