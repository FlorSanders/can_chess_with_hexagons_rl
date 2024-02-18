class Piece:
    white_code = "33"  # yellow
    black_code = "34"  # blue

    def __init__(self, is_white=True):
        self.is_white = is_white
        self.symbol = "?"
        self.name = "unknown"
        self.captured = False

    def __str__(self):
        color_code = self.white_code if self.is_white else self.black_code
        return f"\033[{color_code}m{self.symbol}\033[0m"

    def get_legal_moves(self, board, position_from):
        return []


class Pawn(Piece):
    def __init__(self, is_white=True):
        super().__init__(is_white=is_white)
        self.symbol = "P"
        self.name = "pawn"


class Rook(Piece):
    def __init__(self, is_white=True):
        super().__init__(is_white=is_white)
        self.symbol = "R"
        self.name = "rook"


class Knight(Piece):
    def __init__(self, is_white=True):
        super().__init__(is_white=is_white)
        self.symbol = "N"
        self.name = "knight"


class Bishop(Piece):
    def __init__(self, is_white=True):
        super().__init__(is_white=is_white)
        self.symbol = "B"
        self.name = "bishop"


class King(Piece):
    def __init__(self, is_white=True):
        super().__init__(is_white=is_white)
        self.symbol = "K"
        self.name = "king"


class Queen(Piece):
    def __init__(self, is_white=True):
        super().__init__(is_white=is_white)
        self.symbol = "Q"
        self.name = "queen"
