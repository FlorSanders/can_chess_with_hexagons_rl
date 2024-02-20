import random


class Player:
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


class RandomPlayer(Player):
    """
    Player performing random moves.
    ---
    """

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

    def get_move(self):
        # Get legal moves
        legal_moves = self.get_legal_moves()

        # Collect moves that capture pieces
        best_catch_move = None
        best_catch_value = 0
        for position_from, position_to in legal_moves:
            # Check if move captures a piece
            piece_to = self.board.board[position_to]
            if piece_to is None or piece_to.is_white == self.is_white:
                continue
            if piece_to.value > best_catch_value:
                best_catch_value = piece_to.value
                best_catch_move = (position_from, position_to)

        # Pick move
        if best_catch_move is not None:
            # Catch piece if possible
            return best_catch_move
        else:
            # Move randomly
            return random.choice(legal_moves)
