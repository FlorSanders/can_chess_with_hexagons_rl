from pieces import Pawn, Rook, Knight, Bishop, Queen, King


class HexChessBoard:
    """
    A class to represent a chess board in the hexagonal coordinate system.
    ---
    """

    def __init__(self, empty_board=False):
        """
        Initialize the chess board.
        ---
        Args:
        - empty_board (bool): Whether to initiallize the board in an empty state
        """
        self.board = self.initialize_board(empty_board=empty_board)
        self.captures = [
            [],  # black captures
            [],  # white captures
        ]
        self.check = [False, False]  # black, white

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

    def initialize_board(self, empty_board=False):
        """
        Initialize the chess board and add the pieces in their initial positions.
        ---
        Args:
        - empty_board (bool): Whether to initialize the board in an empty state.

        Returns:
        - board (dict): A dictionary representing the chess board.
        """
        board = {}

        # Q axis
        q_min, q_max = self.get_coordinate_range()
        for q in range(q_min, q_max + 1):
            # R axis
            r_min, r_max = self.get_coordinate_range(q)
            for r in range(r_min, r_max + 1):
                board[(q, r)] = None

        # Add Pieces to the chess board if specified
        if not empty_board:
            for is_white in [True, False]:
                sign = 1 if is_white else -1
                # Pawns
                for q in range(0, 5):
                    board[(sign * q, sign * 1)] = Pawn(is_white=is_white)
                for s in range(0, 4):
                    board[(sign * (-s - 1), sign * (s + 2))] = Pawn(is_white=is_white)
                # Rooks
                board[(sign * 3, sign * 2)] = Rook(is_white=is_white)
                board[(sign * (-3), sign * 5)] = Rook(is_white=is_white)
                # Knights
                board[(sign * 2, sign * 3)] = Knight(is_white=is_white)
                board[(sign * (-2), sign * 5)] = Knight(is_white=is_white)
            # Queen & King
            board[(-1, 5)] = Queen(is_white=True)
            board[(1, 4)] = King(is_white=True)
            board[(-1, -4)] = Queen(is_white=False)
            board[(1, -5)] = King(is_white=False)
            # Three bishops in the center
            for r in range(3, 5 + 1):
                board[(0, r)] = Bishop(is_white=True)
                board[(0, -r)] = Bishop(is_white=False)

        return board

    def get_pieces(self, is_white):
        """
        Get a list of pieces on the chess board.
        ---
        Kwargs:
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
        Get the legal moves for a piece at a given position.
        ---
        Kwargs:
        - position_from (tuple): The position of the piece to get legal moves for.
        - is_white (bool): True if white is at play, False otherwise.
        ---
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

        # Keep track of whether the game is finished
        finished = False

        # Check if the given move is legal
        legal_moves = self.get_legal_moves(position_from, is_white)
        if position_to not in legal_moves:
            return False, finished

        # TODO: Implement check and checkmate checks
        # TODO: Check if the new moves alleviates checkmate if performed, else disallow

        # Check if the move is a capture
        if self.board[position_to] is not None:
            # Capture the piece
            captured_piece = self.board[position_to]
            captured_piece.captured = True
            self.captured_pieces[captured_piece.is_white].append(captured_piece)

            # Check if the game is finished
            if isinstance(captured_piece, King):
                finished = True

            # Remove the captured piece from the board
            self.board[position_to] = None

        # TODO: Implement en passant

        # Move the piece
        self.board[position_to] = self.board[position_from]

        # TODO: Implement castling
        # TODO: Implement promotion

        return True, finished

    def to_string(self, moves=[]):
        """
        Get a string representation of the chess board.
        ---
        Args:
        - moves (list): Locations to color green on the board, indicating legal moves
        Returns:
        - board_str (str): A string representation of the chess board.
        """
        board_str = ""
        r_min, r_max = self.get_coordinate_range()
        for i, r in enumerate(range(r_min, r_max + 1)):
            # Initialize row
            row = "\n" if i > 0 else ""
            offset = " " * (abs(r))
            row += offset
            # Print pieces
            q_min, q_max = self.get_coordinate_range(r)
            for q in range(q_min, q_max + 1):
                piece = self.board[(q, r)]
                if piece is not None:
                    row += f"{piece} "
                elif (q, r) in moves:
                    row += "\033[92m*\033[0m "  # green color code
                else:
                    row += "* "
            # Append row to board
            board_str += row
        return board_str

    def __str__(self):
        return self.to_string()


if __name__ == "__main__":
    # Initialize board & print
    board = HexChessBoard(empty_board=False)
    print(board.to_string([(0, 0)]))
    print()

    # Place one piece in the middle & check legal moves
    pieces = [Pawn, Rook, Knight, Bishop, Queen, King]
    for piece in pieces:
        for is_white in [True, False]:
            board = HexChessBoard(empty_board=True)
            board.board[(0, 0)] = piece(is_white=is_white)
            moves = board.get_legal_moves((0, 0), is_white)
            print(board.to_string(moves))
            print()
