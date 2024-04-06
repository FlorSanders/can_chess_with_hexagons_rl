from hexchess.pieces import Pawn, Rook, Knight, Bishop, Queen, King, piece_map
from functools import lru_cache
import json
import numpy as np
import os


class HexChessBoard:
    """
    Implementation of a hexagonal chess board.
    ---
    """

    def __init__(
        self,
        initialize_empty=False,
        initialize_random=False,
        save_dir=os.path.join(os.path.dirname(__file__), "..", "states"),
        save_prob=0.1,
    ):
        """
        Initialize the chess board.
        ---
        Args:
        - initialize_empty (bool): Whether to initiallize the board in an empty state
        - initialize_random (bool): Whether to initialize the board in a random state (from save_dir)
        - save_dir: Directory of saved states
        - save_prob: Probability to save the state any time get_move is called
        """
        # Make sure either save_dir exists or we don't expect random initialization
        assert (
            save_dir is not None or not initialize_random
        ), "Random initialization is only possible if a directory is provided"

        # Save params
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.save_prob = save_prob
        self.reset(initialize_empty, initialize_random)
    
    def reset(
        self,
        initialize_empty=False,
        initialize_random=False,
    ):
        # Initialize game
        self.initialize_board(
            empty_board=initialize_empty,
            random_board=initialize_random,
        )
        self.captures = [
            [],  # black captures
            [],  # white captures
        ]
        self.is_checked = [
            False,  # black
            False,  # white
        ]

    @lru_cache()
    def get_coordinate_range(self, c_prime=0):
        """
        Get the range of coordinates that are allowed on one of the axes.
        ---
        Args:
        - c_prime (int): The current coordinate on the other axis

        Returns:
        - c_min (int): The minimum coordinate allowed
        - c_max (int): The maximum coordinate allowed
        """
        c_min, c_max = -5, 5
        c_min = max(c_min, c_min - c_prime)
        c_max = min(c_max, c_max - c_prime)
        return c_min, c_max

    def convert_coordinate(self, q=None, r=None, s=None):
        """
        Convert coordinates from QRS to QRS.
        ---
        Args:
        - q (int): The Q coordinate
        - r (int): The R coordinate
        - s (int): The S coordinate

        Returns:
        - c (int): The missing coordinate
        """
        if q is None and not (r is None or s is None):
            q = -r - s
            return q
        elif r is None and not (q is None or s is None):
            r = -q - s
            return r
        elif s is None and not (q is None or r is None):
            s = -q - r
            return s
        else:
            raise ValueError("Exactly one missing coordinate is required")

    def move_along_axis(self, position_from, axis, distance):
        """
        Compute the coordinate of the new position after a move along an axis.
        ---
        Args:
        - position_from (tuple): The starting position (q, r)
        - axis (int): Axis along which to move (0, 1, 2)
        - distance (int): Distance to move along the axis

        Returns:
        - position_to (tuple): The new position after the move is performed (q, r). Returns None if resulting coordinate is not on the board.
        """
        # Verify that axis is valid
        assert axis in [
            0,
            1,
            2,
        ], "Axis must be one of [0, 1, 2], representing the (q, r, s) axes"

        # Parse starting position
        q_from, r_from = position_from
        s_from = self.convert_coordinate(q=q_from, r=r_from)
        position_to = [q_from, r_from, s_from]

        # Move along axis
        position_to[axis] += distance

        # Normalize coordinates
        normalize_axis = (axis + 1) % 3
        position_to[normalize_axis] -= distance

        # Verify that resulting coordinate is on the board
        if max(position_to) > 5 or min(position_to) < -5:
            return None

        # Return resulting position
        return tuple(position_to[:2])

    def move_along_diagonal(self, position_from, diagonal, distance):
        """
        Compute the coordinate of the new position after a move along a diagonal.
        ---
        Args:
        - position_from (tuple): The starting position (q, r)
        - diagonal (int): Diagonal along which to move (0, 1, 2) = (qr, rs, sq)
        - distance (int): Distance to move along the axis

        Returns:
        - position_to (tuple): The new position after the move is performed (q, r). Returns None if resulting coordinate is not on the board.
        """
        # Verify that diagonal is valid
        assert diagonal in [
            0,
            1,
            2,
        ], "Axis must be one of [0, 1, 2], representing the (qr, rs, sq) diagonals"

        # Move along first axis
        axis1 = diagonal
        position_temp = self.move_along_axis(position_from, axis1, distance)

        # Return None if resulting coordinate is not on the board
        if position_temp is None:
            return None

        # Move along second axis
        axis2 = (diagonal + 1) % 3
        position_to = self.move_along_axis(position_temp, axis2, -distance)

        return position_to

    def initialize_board(
        self,
        empty_board=False,
        random_board=False,
    ):
        """
        Initialize the chess board and add the pieces in their initial positions.
        ---
        Args:
        - empty_board (bool): Whether to initialize the board in an empty state.

        Returns:
        - board (dict): A dictionary representing the chess board.
        """
        self.board = {}

        # Q axis
        q_min, q_max = self.get_coordinate_range()
        for q in range(q_min, q_max + 1):
            # R axis
            r_min, r_max = self.get_coordinate_range(q)
            for r in range(r_min, r_max + 1):
                self.board[(q, r)] = None

        # Initialize board empty if specified
        if empty_board:
            return self.board

        # Initialize board randomly if specified
        if random_board:
            # Initialize randomly
            success = self.load_state(state_file=None)
            if success:
                return self.board
            else:
                print(
                    "WARNING: Random initialization requested but failed - initializing with standard layout"
                )

        # Add pieces to the board
        for is_white in [True, False]:
            for piece in [Pawn, Rook, Knight, Bishop, King, Queen]:
                initial_positions = piece.initial_positions[is_white]
                for position in initial_positions:
                    self.board[position] = piece(is_white=is_white)

        return self.board

    def save_state(self):
        """
        Save the current state of the chess board.
        ---
        """
        # Build state
        board_state = {}
        for position, piece in self.board.items():
            if piece is not None:
                board_state[f"{position}"] = [piece.symbol, piece.is_white]
            else:
                board_state[f"{position}"] = None

        # Save state
        file_name = f"{len(os.listdir(self.save_dir))}".zfill(5) + ".json"
        with open(os.path.join(self.save_dir, file_name), "w") as f:
            json.dump(board_state, f)

    def load_state(self, state_file=None):
        """
        Load a chess board state from a file.
        ---
        Args:
        - state_file (str): The file to load the board state from. (picks a random file if None)

        Returns:
        - success (bool): True if a state
        """

        # Pick random state file if none is chosen
        if state_file is None:
            state_files = os.listdir(self.save_dir)
            if len(state_files) == 0:
                return False
            state_file = np.random.choice(state_files)

        # Make sure state file exists
        if not os.path.exists(os.path.join(self.save_dir, state_file)):
            return False

        # Load state
        with open(os.path.join(self.save_dir, state_file), "r") as f:
            board_state = json.load(f)

        # Parse board state
        for position, piece in board_state.items():
            # print(isinstance(position, tuple), isinstance(position, str))
            position = tuple(
                [
                    int(i)
                    for i in f"{position}".replace("(", "").replace(")", "").split(",")
                ]
            )
            if piece is None:
                self.board[position] = None
            else:
                symbol, is_white = piece
                self.board[position] = piece_map[symbol](is_white)

        return True

    def get_pieces(self, is_white):
        """
        Get a list of pieces on the chess board.
        ---
        Args:
        - is_white (bool): Whether to get white pieces or black pieces.

        Returns:
        - pieces (dict): The subset of the board that is occupied by the selected pieces.
        """
        pieces = {}
        for position, piece in self.board.items():
            if piece is not None and piece.is_white == is_white:
                pieces[position] = piece
        return pieces

    def get_legal_moves(self, position_from, is_white):
        """
        Get the legal moves for a piece at a given position.At least two valid coordinates must
        ---
        Args:
        - position_from (tuple): The position of the piece to get legal moves for.
        - is_white (bool): True if white is at play, False otherwise.

        Returns:
        - legal_moves (list): A list of legal moves for the piece at the given position.
        """
        piece = self.board[position_from]
        if piece is not None and piece.is_white == is_white:
            legal_moves = piece.get_legal_moves(self, position_from)
            return legal_moves
        else:
            return []

    def move(self, position_from, position_to, is_white):
        """
        Move a piece from one position to another.
        ---
        Args:
        - position_from (tuple): The position of the piece to be moved.
        - position_to (tuple): The position to move the piece to.
        - is_white (bool): True if white is at play, False otherwise.

        Returns:
        - success (bool): True if the move was successful, False otherwise.
        - finished (bool): True if the game is over, False otherwise.
        """

        # Save the state
        if np.random.random() < self.save_prob:
            self.save_state()

        # Keep track of whether the game is finished
        finished = False

        # Check if the given move is legal
        legal_moves = self.get_legal_moves(position_from, is_white)
        if position_to not in legal_moves:
            return False, finished

        # TODO: Implement check and checkmate checks
        # TODO: Check if the new moves alleviates checkmate if performed, else disallow

        # Get the piece to be moved and the content of the board the piece is moving to
        piece_from = self.board[position_from]
        piece_to = self.board[position_to]

        # Check if the move is a capture
        if piece_to is not None:
            # Verify move is no self-capture
            assert (
                piece_to.is_white != piece_from.is_white
            ), "Self-captures are not allowed!"

            # Capture the piece
            piece_to.captured = True
            self.captures[piece_to.is_white].append(piece_to)

            # Check if the game is finished
            if isinstance(piece_to, King):
                finished = True

            # Remove the captured piece from the board
            self.board[position_to] = None

        # Move the piece
        self.board[position_to] = self.board[position_from]
        self.board[position_from] = None
        self.board[position_to].moved = True

        # TODO: Implement en passant
        # TODO: Implement promotion

        return True, finished

    def print_coords(self):
        """
        Print a visualization of the coordinate system.
        ---
        """

        # Initialize coordinate strings
        q_str = ""
        r_str = s_str = "\n"

        # Populate coordinate strings
        r_min, r_max = self.get_coordinate_range()
        for i, r in enumerate(range(r_min, r_max + 1)):
            # Initialize rows
            offset = " " * abs(r)
            row_start = "\n" if i > 0 else ""
            q_row = r_row = s_row = row_start + offset
            # Print coordinates
            q_min, q_max = self.get_coordinate_range(r)
            for q in range(q_min, q_max + 1):
                # Compute s-coord
                s = self.convert_coordinate(q=q, r=r)
                # Build coordinate strings
                q_row += f"{q} ".rjust(3)
                r_row += f"{r} ".rjust(3)
                s_row += f"{s} ".rjust(3)
            # Add rows to strings
            q_str += q_row
            r_str += r_row
            s_str += s_row

        print(q_str)
        print(r_str)
        print(s_str)

    def to_string(self, moves=[]):
        """
        Get a string representation of the chess board.
        ---
        Args:
        - moves (list): Locations to color green on the board, indicating legal moves

        Returns:
        - board_str (str): A string representation of the chess board.
        """
        # Color codes
        white_code = "33"  # yellow
        black_code = "34"  # blue
        move_code = "92"  # green

        board_str = ""
        r_min, r_max = self.get_coordinate_range()
        for i, r in enumerate(range(r_min, r_max + 1)):
            # Initialize row
            offset = " " * abs(r)
            row_start = "\n" if i > 0 else ""
            row = row_start + offset
            # Print pieces
            q_min, q_max = self.get_coordinate_range(r)
            for q in range(q_min, q_max + 1):
                piece = self.board[(q, r)]
                if piece is not None:
                    color_code = white_code if piece.is_white else black_code
                    color_code = move_code if (q, r) in moves else color_code
                    row += f"\033[{color_code}m{piece}\033[0m "
                elif (q, r) in moves:
                    row += f"\033[{move_code}m*\033[0m "
                else:
                    row += "* "
            # Append row to board
            board_str += row
        return board_str

    def __str__(self):
        return self.to_string()


if __name__ == "__main__":
    run_tests = [2]

    # Initialize board & print information
    if 0 in run_tests:
        board = HexChessBoard()
        board.print_coords()
        print()
        print(board.to_string())
        print()

    # Place one piece in the middle & check legal moves
    if 1 in run_tests:
        pieces = [Pawn, Rook, Knight, Bishop, Queen, King]
        for piece in pieces:
            for is_white in [True, False]:
                board = HexChessBoard(empty_board=True)
                board.board[(0, 0)] = piece(is_white=is_white)
                moves = board.get_legal_moves((0, 0), is_white)
                print(board.to_string(moves))
                print()

    # Play random game
    if 2 in run_tests:
        import random

        # Initialize board
        board = HexChessBoard()
        print(board)
        print()

        n_rounds = 3
        for i in range(n_rounds):
            for is_white in [True, False]:
                # Pick random legal move
                move = None
                while move is None:
                    # Pick random piece
                    pieces = board.get_pieces(is_white)
                    piece_keys = list(pieces.keys())
                    piece_key = random.choice(piece_keys)

                    # Get legal moves
                    legal_moves = board.get_legal_moves(piece_key, is_white)
                    # Pick random move
                    if len(legal_moves) > 0:
                        move = random.choice(legal_moves)

                # Perform move
                board.move(piece_key, move, is_white)
                print(board)
                print()
