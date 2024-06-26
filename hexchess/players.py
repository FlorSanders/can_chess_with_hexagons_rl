import random


class Player:
    name = "Undefined Player"

    def __init__(self, board, is_white):
        self.board = board
        self.is_white = is_white

    def get_legal_moves(self):
        legal_moves = []
        pieces = self.board.get_pieces(self.is_white)
        for position_from in pieces.keys():
            piece_moves = self.board.get_legal_moves(position_from, self.is_white)
            for position_to in piece_moves:
                legal_moves.append((position_from, position_to))
        return legal_moves

    def get_move(self):
        pass

    def __str__(self):
        return self.name


class RandomPlayer(Player):
    """
    Player performing random moves.
    ---
    """

    name = "Random Player"

    def get_move(self):
        legal_moves = self.get_legal_moves()
        position_from, position_to = random.choice(legal_moves)
        return position_from, position_to


class GreedyPlayer(Player):
    """
    Players performing greedy moves.
    If a capture is possible, the piece with the maximum value is captured.
    Otherwise, a random move is performed.
    ---
    """

    name = "Greedy Player"

    def get_move(self):
        # Get legal moves
        legal_moves = self.get_legal_moves()

        # Collect moves that capture pieces
        best_catch_move = None
        best_catch_value = 0
        best_from_value = 1000
        for position_from, position_to in legal_moves:
            # Check if move captures a piece
            piece_from = self.board.board[position_from]
            piece_to = self.board.board[position_to]
            if piece_to is None or piece_to.is_white == self.is_white:
                continue
            # Check if move is better than the best catch move
            is_better_catch = piece_to.value >= best_catch_value
            is_safer_move = piece_from.value <= best_from_value
            if is_better_catch and is_safer_move:
                best_catch_value = piece_to.value
                best_from_value = piece_from.value
                best_catch_move = (position_from, position_to)

        # Pick move
        if best_catch_move is not None:
            # Catch piece if possible
            return best_catch_move
        else:
            # Move randomly
            return random.choice(legal_moves)
