from hexchess import HexChessBoard
import numpy as np


class HexChessEnv:
    def __init__(self, opponent_class, opponent_is_white):
        # Save variables
        self.opponent_class = opponent_class
        self.opponent_is_white = opponent_is_white

        # Constants
        self.n_pieces = 6

        # Initialize game
        self.init_game()

    def init_game(self):
        """
        Initialize the board
        """
        # Reset board
        self.reset()

        # Board properties
        self.board_size = len(self.board.board.keys())
        self.coord_min, self.coord_max = self.board.get_coordinate_range()
        self.coord_range = self.coord_max - self.coord_min + 1

        # Action space
        self.action_space = np.zeros((self.board_size, self.board_size))
        # Position to action mapper
        self.mapper = np.cumsum(
            [0]
            + [
                np.sum(np.abs(self.board.get_coordinate_range(q))) + 1
                for q in range(self.coord_min, self.coord_max + 1)
            ]
        )

    def position_to_index(self, position):
        """
        Convert q,r coordinates to action
        ---
        Parameters:
        - position (tuple): The q,r coordinates of the position to convert.

        Returns:
        - action: The action corresponding to the given q,r coordinates.
        """
        q, r = position
        r_min, r_max = self.board.get_coordinate_range(q)
        i, j = q - self.coord_min, r - r_min
        return self.mapper[i] + j

    def index_to_position(self, action):
        """
        Convert action to q,r coordinates
        ---
        Parameters:
        - action (int): The action to convert.

        Returns:
        - position (tuple): The q,r coordinates corresponding to the given action.
        """
        i = np.argwhere(self.mapper > action)[0, 0] - 1
        q = i + self.coord_min
        r_min, r_max = self.board.get_coordinate_range(q)
        r = action - self.mapper[i] + r_min
        return q, r

    def reset(self):
        # Initialize board
        self.board = HexChessBoard()
        self.opponent = self.opponent_class(self.board, self.opponent_is_white)
        self.actions = []
        self.rewards = []

    def get_state(self):
        """
        State representation of the current chess board configuration
        ---
        Returns:
        - state: A 11x11x6 numpy array representing the current state of the board.
        ---
        Notes:
        - The 11x11x6 state representation is a 3D array with the first dimension representing the q coordinate, the second dimension representing the r coordinate, and the third dimension representing the piece type.
        - The piece type is represented by an integer from 0 to 5, where 0 represents a pawn, 1 represents a knight, 2 represents a bishop, 3 represents a rook, 4 represents a queen, and 5 represents a king.
        - The sign of the piece type is represented by a 1 or -1, where 1 represents a white piece and -1 represents a black piece.
        - This square representation of the hexagonal board leaves 30 wasted entries per slice and is thus inefficient.
          However, it does consistently represent the geometrical structure of the q,r coordinate system used for gameplay.
        """

        # Parse board content into state variable
        state = np.zeros((self.coord_range, self.coord_range, self.n_pieces), dtype=int)
        for (q, r), piece in self.board.board.items():
            # Skip empty positions
            if piece is None:
                continue
            # Determine the color of the current piece given the sign
            piece_sign = 1 if piece.is_white else -1
            # Populate board with piece
            r_min, r_max = self.board.get_coordinate_range(q)
            state[q - self.coord_min, r - self.coord_min, piece.index] = piece_sign

        return state

    def get_action_mask(self, is_white=None):
        # Assume the agent is playing as default
        if is_white is None:
            is_white = not self.opponent_is_white

        # Reset action space
        action_mask = np.zeros((self.board_size, self.board_size), dtype=bool)

        # Go through the pieces of the given color
        pieces = self.board.get_pieces(is_white)
        for position_from, piece in pieces.items():
            # Compute move index for piece position
            index_from = self.position_to_index(position_from)

            # Go through the legal moves for the piece
            legal_moves = piece.get_legal_moves(self.board, position_from)
            for position_to in legal_moves:
                # Compute move index for landing position
                index_to = self.position_to_index(position_to)

                # Indicate move legality
                action_mask[index_from, index_to] = True

        return action_mask

    def step(self, action):
        """
        Perform a step in the environment
        ---
        Parameters:
        - action: A tuple containing the action to be performed (position_from, position_to).
        ---
        Returns:
        - state: The state of the board after the action is performed.
        - reward: The reward associated with the action.
        - done: A boolean indicating whether the game is over.
        - info: Additional information about the environment.
        """

        # Parse action
        position_from, position_to = action

        # Compute capture_reward
        capture_reward = 0
        piece_to = self.board.board[position_to]
        if piece_to is not None and piece_to.is_white == self.opponent_is_white:
            capture_reward = piece_to.value

        # Perform action
        success, finished = self.board.move(
            position_from, position_to, not self.opponent_is_white
        )

        # Determine reward
        reward = 0
        if success:
            reward += capture_reward
        else:
            return self.get_state(), 0, finished

        # Perform opponent action if game is not finished
        if not finished:
            # Retry until opponent move is accepted
            opponent_success = False
            while not opponent_success:
                # Get opponent action
                position_from, position_to = self.opponent.get_move()
                # Compute capture_reward
                opponent_capture_reward = 0
                piece_to = self.board.board[position_to]
                if piece_to is not None and piece_to.is_white != self.opponent_is_white:
                    opponent_capture_reward = piece_to.value
                # Perform action
                opponent_success, finished = self.board.move(
                    position_from, position_to, self.opponent_is_white
                )
            # Perform punishment if own piece is captured
            reward -= opponent_capture_reward

        # Return update info
        newstate = self.get_state()
        return newstate, reward, finished

    def render(self):
        print(self.board.to_string())


if __name__ == "__main__":
    from hexchess import RandomPlayer

    # Initialize board
    env = HexChessEnv(RandomPlayer, True)
    print()

    # Render
    env.render()
    print()

    # Test coordinate conversions
    for position in [(-5, 0), (-5, 5), (0, -5), (0, 0), (0, 5), (5, -5), (5, 0)]:
        index = env.position_to_index(position)
        position_prime = env.index_to_position(index)
        print(f"{position} -> {index} -> {position_prime}")
    print()

    # Test state generation
    state = env.get_state()
    vis = np.zeros(state.shape[:2], dtype="str")
    vis[:] = " "
    for i in range(state.shape[2]):
        pieces = ["P", "R", "N", "B", "Q", "K"]
        vis[np.abs(state[:, :, i]) == 1] = pieces[i]
    print(vis)
    print()

    # Test action mask generation
    action_mask = env.get_action_mask()
    for i in range(action_mask.shape[0]):
        for j in range(action_mask.shape[1]):
            if action_mask[i, j]:
                # Print attack
                print(f"{i}-{j}", end=", ")

                # Add departure to visualization
                q, r = env.index_to_position(i)
                vis[q - env.coord_min, r - env.coord_min] = "-"

                # Add landing to visualization
                q, r = env.index_to_position(j)
                vis[q - env.coord_min, r - env.coord_min] = "*"
    print("\n")

    print(vis)
    print()
