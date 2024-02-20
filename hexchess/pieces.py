class Piece:
    initial_positions = [[], []]  # Black, White
    symbol = "?"
    name = "unknown"
    index = -1
    value = 0

    def __init__(self, is_white=True):
        # Piece properties
        self.is_white = is_white

        # Piece state
        self.captured = False
        self.moved = False

    def __str__(self):
        return self.symbol

    def _get_moves_along_axes(
        self, board, position_from, distance_range=range(1, 10 + 1)
    ):
        """
        Get the moves along axes.
        ---
        Args:
        - board (Board): The chess board.
        - position_from (tuple): The position of the piece to get moves for.
        - distance_range (range): The range of distances to check.

        Returns:
        - moves (list): A list of moves along axes.
        """
        # Initialize moves
        moves = []

        # For any axis
        for axis in range(3):
            # Positive & negative direction
            for sign in [-1, 1]:
                # Move for distance
                for distance in distance_range:
                    # move along axis
                    position_to = board.move_along_axis(
                        position_from, axis, sign * distance
                    )

                    # Break if position is not on the board
                    if position_to is None:
                        break

                    # Break if move is self capture
                    piece_to = board.board[position_to]
                    if piece_to is not None and piece_to.is_white == self.is_white:
                        break

                    # Add move
                    moves.append(position_to)

                    # Break if move is capture
                    if piece_to is not None and piece_to.is_white != self.is_white:
                        break

        return moves

    def _get_moves_along_diagonals(
        self, board, position_from, distance_range=range(1, 5 + 1)
    ):
        """
        Get the moves along diagonals.
        ---
        Args:
        - board (Board): The chess board.
        - position_from (tuple): The position of the piece to get moves for.
        - distance_range (range): The range of distances to check.

        Returns:
        - moves (list): A list of moves along diagonals.
        """
        # Initialize moves
        moves = []

        # For any diagonal
        for diagonal in range(3):
            # Move in positive & negative direction
            for sign in [1, -1]:
                # Move for distance
                for distance in distance_range:
                    # Move along diagonal
                    position_to = board.move_along_diagonal(
                        position_from, diagonal, sign * distance
                    )

                    # Break if position isn't on the board
                    if position_to is None:
                        break

                    # Break if move is self capture
                    piece_to = board.board[position_to]
                    if piece_to is not None and piece_to.is_white == self.is_white:
                        break

                    # Add move
                    moves.append(position_to)

                    # Break if move is capture
                    if piece_to is not None and piece_to.is_white != self.is_white:
                        break

        return moves

    def get_legal_moves(self, board, position_from):
        """
        Get the legal moves for a piece at a given position.
        [source](https://en.wikipedia.org/wiki/Hexagonal_chess)
        ---
        Args:
        - board (Board): The chess board.
        - position_from (tuple): The position of the rook to get legal moves for.

        Returns:
        - legal_moves (list): A list of legal moves for the rook at the given position.
        """
        return []


class Pawn(Piece):
    initial_positions = [
        [
            (4, -5),
            (3, -4),
            (2, -3),
            (1, -2),
            (0, -1),
            (-1, -1),
            (-2, -1),
            (-3, -1),
            (-4, -1),
        ],  # Black
        [
            (-4, 5),
            (-3, 4),
            (-2, 3),
            (-1, 2),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
        ],  # White
    ]
    symbol = "P"
    name = "pawn"
    index = 0
    value = 1

    def get_legal_moves(self, board, position_from):
        """
        The pawn may move one vacant cell vertically forward.
        If it stands on its starting cell or on the starting cell of any other pawn of its colour, then it is also allowed to move two vacant cells vertically forward.
        It may capture one cell orthogonally forward at a 60° angle to the vertical, including capturing en passant. (This capturing move is not diagonal, unlike in orthodox chess.)
        It is promoted when it reaches the end of any file.
        """

        # Initialize legal moves
        legal_moves = []

        # Move forward
        axis = 1
        sign = -1 if self.is_white else 1
        max_distance = (
            2 if position_from in self.initial_positions[self.is_white] else 1
        )
        for distance in range(1, max_distance + 1):
            # Move along axis
            position_to = board.move_along_axis(position_from, axis, sign * distance)

            # Break if position isn't on the board
            if position_to is None:
                break

            # Break if space is occupied
            piece_to = board.board[position_to]
            if piece_to is not None:
                break

            # Add move to legal moves
            legal_moves.append(position_to)

        # Capture 60° angle
        for axis in [0, 2]:
            # Move along axis
            position_to = board.move_along_axis(position_from, axis, -sign)

            # Continue if position isn't on the board
            if position_to is None:
                continue

            # Add move if it results in capture
            piece_to = board.board[position_to]
            if piece_to is not None and piece_to.is_white != self.is_white:
                legal_moves.append(position_to)

        # TODO: Implement en passant
        # TODO: Implement promotion

        return legal_moves


class Rook(Piece):
    initial_positions = [
        [(-3, -2), (3, -5)],  # Black
        [(3, 2), (-3, 5)],  # White
    ]
    symbol = "R"
    name = "rook"
    index = 1
    value = 5

    def get_legal_moves(self, board, position_from):
        """
        The rook may move any number of cells orthogonally, traveling through cell edges.
        """
        axis_moves = self._get_moves_along_axes(board, position_from)
        return axis_moves


class Knight(Piece):
    initial_positions = [
        [(-2, -3), (2, -5)],  # Black
        [(2, 3), (-2, 5)],  # White
    ]
    symbol = "N"
    name = "knight"
    index = 2
    value = 3

    def get_legal_moves(self, board, position_from):
        """
        The knight may move two cells orthogonally in one direction and then one cell orthogonally at a 60° angle, jumping over intervening pieces.
        Equivalently, the knight may move to any nearest cell not on an orthogonal or diagonal line on which it stands.
        """

        # Get legal moves
        legal_moves = []

        # For any primary axis
        for primary_axis in [0, 1, 2]:
            for sign in [1, -1]:
                # Move along primary axis
                position_temp = board.move_along_axis(
                    position_from, primary_axis, sign * 2
                )

                # Skip if position isn't on the board
                if position_temp is None:
                    continue

                # For any secondary axis
                for secondary_axis in [0, 1, 2]:
                    # Different from the primary axis
                    if secondary_axis == primary_axis:
                        continue

                    # Move along secondary axis
                    position_to = board.move_along_axis(
                        position_temp, secondary_axis, -sign * 1
                    )

                    # Skip if position isn't on the board
                    if position_to is None:
                        continue

                    # Break if move is self capture
                    piece_to = board.board[position_to]
                    if piece_to is not None and piece_to.is_white == self.is_white:
                        continue

                    # Add move to legal moves
                    legal_moves.append(position_to)
        return legal_moves


class Bishop(Piece):
    initial_positions = [
        [(0, -3), (0, -4), (0, -5)],  # Black
        [(0, 3), (0, 4), (0, 5)],  # White
    ]
    symbol = "B"
    name = "bishop"
    index = 3
    value = 3

    def get_legal_moves(self, board, position_from):
        """
        The bishop may move any number of cells diagonally, traveling through cell edges.
        """
        diagonal_moves = self._get_moves_along_diagonals(board, position_from)
        return diagonal_moves


class Queen(Piece):
    initial_positions = [
        [(-1, -4)],  # Black
        [(-1, 5)],  # White
    ]
    symbol = "Q"
    name = "queen"
    index = 4
    value = 9

    def get_legal_moves(self, board, position_from):
        """
        The queen may move any number of cells orthogonally or diagonally.
        """
        axis_moves = self._get_moves_along_axes(board, position_from)
        diagonal_moves = self._get_moves_along_diagonals(board, position_from)
        return axis_moves + diagonal_moves


class King(Piece):
    initial_positions = [
        [(1, -5)],  # Black
        [(1, 4)],  # White
    ]
    symbol = "K"
    name = "king"
    index = 5
    value = 44  # One more than the sum of all other piece values on the board

    def get_legal_moves(self, board, position_from):
        """
        The king may move one cell orthogonally or diagonally. There is no castling.
        """
        axis_moves = self._get_moves_along_axes(board, position_from, range(1, 1 + 1))
        diagonal_moves = self._get_moves_along_diagonals(
            board, position_from, range(1, 1 + 1)
        )
        return axis_moves + diagonal_moves
