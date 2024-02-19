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
    If a capture is possible, it is performed.
    Otherwise, a random move is performed.
    ---
    """

    def get_move(self):
        # Get legal moves
        legal_moves = self.get_legal_moves()

        # Collect moves that capture pieces
        catch_moves = []
        for position_from, position_to in legal_moves:
            # Check if move captures a piece
            piece_to = self.board.board[position_to]
            if piece_to is not None and piece_to.is_white != self.is_white:
                catch_moves.append((position_from, position_to))

        # Pick move
        if len(catch_moves):
            # Catch piece if possible
            return random.choice(catch_moves)
        else:
            # Move randomly
            return random.choice(legal_moves)
