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
    def get_move(self):
        legal_moves = self.get_legal_moves()
        position_from, position_to = random.choice(legal_moves)
        return position_from, position_to
