from hexchess import HexChessBoard
import numpy as np


class HexChessEnv:
    def __init__(self, opponent_class, opponent_is_white):
        # Save variables
        self.opponent_class = opponent_class
        self.opponent_is_white = opponent_is_white

        # Constants
        self.rewards = {  # Reinfield reward values
            "P": 1,
            "R": 5,
            "N": 3,
            "B": 3,
            "Q": 9,
            "K": 100,  # Additional high value for capturing the king
        }
        self.n_pieces = len(self.rewards.keys())  # Number of pieces

        # Initialize game
        self.init_game()

    def init_game(self):
        """
        Initialize the board
        ---
        """
        # Reset board
        self.reset()

        # Board properties
        self.board_size = len(self.board.board.keys())
        self.coord_min, self.coord_max = self.board.get_coordinate_range()
        self.coord_range = self.coord_max - self.coord_min + 1

        # Action space
        self.action_space = np.zeros((self.board_size, self.board_size))

    def reset(self):
        # Initialize board
        self.board = HexChessBoard()
        self.opponent = self.opponent_class(self.board, self.opponent_is_white)
        self.actions = []
        self.rewards = []

    @property
    def state(self):
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
            state[q - coord_min, r - coord_min, piece.index] = piece_sign

        return state

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
            capture_reward = self.rewards[piece_to.symbol]

        # Perform action
        success, finished = self.board.move(
            position_from, position_to, not self.opponent_is_white
        )

        # Determine reward
        reward = 0
        if success:
            reward = capture_reward

        # Perform opponent action if game is not finished
        if not finished:
            # Retry until opponent move is accepted
            opponent_success = False
            while not opponent_success:
                # Get opponent action
                position_from, position_to = self.opponent.get_move()
                opponent_success, finished = self.board.move(
                    position_from, position_to, self.opponent_is_white
                )

        # Return update info
        newstate = self.state
        return newstate, reward, finished

    def render(self):
        print(self.board.to_string())


if __name__ == "__main__":
    from hexchess import RandomPlayer

    env = HexChessEnv(RandomPlayer, True)
    env.render()
    print(env.board_size)
