import pygame
import math
from board import HexChessBoard
from utils import hexagon_dimensions, hexagon_points, qr_to_xy, xy_to_qr


class Chess:
    colors = {
        0: "#d18b47",  # Dark
        1: "#eead6f",  # Medium
        2: "#ffcf9f",  # Light
        "active": "#c9cbe0",
        "green": "#d0dbb2",
    }

    def __init__(self):
        # Initialize board
        self.board = (
            HexChessBoard()
        )  # TODO: allow starting / stopping / restarting game
        self._setup_game()
        self._load_assets()

    def _setup_game(self):
        # Start PyGame
        pygame.init()
        self.screen_w, self.screen_h = 1200, 900
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("Can Chess With Hexagons?")
        self.timer = pygame.time.Clock()
        self.fps = 15

        # Define fonts
        self.text_font = pygame.font.SysFont("Fira Sans Medium", 35)
        self.title_font = pygame.font.SysFont("Fira Sans Medium", 50)

        # Define element sizes
        self.asset_size = 50
        self.hex_size = 35

        # Game state
        self.is_white = True
        self.selected_piece = None
        self.legal_moves = []
        self.finished = False

    def _load_assets(self):
        self.assets = {}
        pieces = ["pawn", "rook", "knight", "bishop", "king", "queen"]
        colors = ["black", "white"]
        for color in colors:
            for piece in pieces:
                piece_asset = pygame.image.load(f"./assets/{piece}_{color}.svg")
                piece_asset = pygame.transform.scale(
                    piece_asset, (self.asset_size, self.asset_size)
                )
                self.assets[f"{piece}_{color}"] = piece_asset

    def _draw_piece(self, piece, x_c, y_c):
        # Obtain piece asset
        piece_color = "white" if piece.is_white else "black"
        piece_asset = self.assets[f"{piece.name}_{piece_color}"]

        # Draw piece
        self.screen.blit(
            piece_asset, (x_c - self.asset_size / 2, y_c - self.asset_size / 2)
        )

    def _draw_hexagon(self, q, r, color):
        # Compute x & y center from q, r
        # print(color)
        x_c, y_c = qr_to_xy(q, r, self.hex_size)
        x_c += self.screen_w / 2
        y_c += self.screen_h / 2

        # Get hexagon points
        hex_points = [(x + x_c, y + y_c) for x, y in hexagon_points(self.hex_size)]

        # Draw hexagon
        pygame.draw.polygon(self.screen, color, hex_points)
        return x_c, y_c

    def draw_board(self):
        # Draw title above the board
        title = self.title_font.render("Can Chess With Hexagons?", True, "black")
        self.screen.blit(
            title,
            (self.screen_w / 2 - title.get_rect()[2] / 2, 25),
        )
        status_color = "White" if self.is_white else "Black"
        subtitle = self.text_font.render(f"{status_color} to move", True, "black")
        self.screen.blit(
            subtitle,
            (self.screen_w / 2 - subtitle.get_rect()[2] / 2, 75),
        )

        # Draw board hexagons and pieces
        board = self.board
        board_content = board.board
        q_min, q_max = board.get_coordinate_range()
        for i, q in enumerate(range(q_min, q_max + 1)):
            color_index = abs(q) % 3
            r_min, r_max = board.get_coordinate_range(q)
            for j, r in enumerate(range(r_min, r_max + 1)):
                # Draw board hexagon
                hex_color = (
                    (color_index - j) % 3
                    if not self.selected_piece == (q, r)
                    else "active"
                )
                hex_color = hex_color if not (q, r) in self.legal_moves else "green"
                x_c, y_c = self._draw_hexagon(q, r, self.colors[hex_color])

                # Draw piece if present
                piece = board_content[(q, r)]
                if piece is not None:
                    self._draw_piece(piece, x_c, y_c)

        # Draw captured pieces
        white_captures_label = self.text_font.render("White captures", True, "black")
        self.screen.blit(
            white_captures_label,
            (25, self.screen_h - white_captures_label.get_rect()[3] - 75),
        )
        white_captures = self.board.captures[0]
        for i, piece in enumerate(sorted(white_captures, key=lambda x: x.index)):
            x_c = 35 + i * (35 + 5)
            y_c = self.screen_h - 50
            self._draw_piece(piece, x_c, y_c)
        black_captures_label = self.text_font.render("Black captures", True, "black")
        self.screen.blit(
            black_captures_label,
            (
                self.screen_w - black_captures_label.get_rect()[2] - 25,
                self.screen_h - black_captures_label.get_rect()[3] - 75,
            ),
        )
        black_captures = self.board.captures[1]
        for i, piece in enumerate(sorted(black_captures, key=lambda x: x.index)):
            x_c = self.screen_w - 35 - i * (35 + 5)
            y_c = self.screen_h - 50
            self._draw_piece(piece, x_c, y_c)

    def handle_game_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Mouse click at position x,y
            x, y = event.pos

            # Check if coordinate is on the board
            q, r = xy_to_qr(x - self.screen_w / 2, y - self.screen_h / 2, self.hex_size)
            s = -q - r
            if max([q, r, s]) <= 5 and min([q, r, s]) >= -5:
                # Obtain selected piece
                selected_piece = self.board.board[(q, r)]

                if (
                    selected_piece is not None
                    and selected_piece.is_white == self.is_white
                ):
                    # Select a piece to play
                    if self.selected_piece == (q, r):
                        # Unselect piece if selected
                        self.selected_piece = None
                        self.legal_moves = []
                    else:
                        # Piece selected
                        self.selected_piece = (q, r)
                        self.legal_moves = self.board.get_legal_moves(
                            self.selected_piece, self.is_white
                        )
                elif self.selected_piece is not None and (q, r) in self.legal_moves:
                    # Move a piece
                    success, self.finished = self.board.move(
                        self.selected_piece, (q, r), self.is_white
                    )
                    assert success, "Move did not succeed!"
                    # Reset selection
                    self.selected_piece = None
                    self.legal_moves = []
                    # Advance turn
                    self.is_white = not self.is_white

                # Select
                if self.selected_piece is None:
                    self.selected_piece = (q, r)

    def run(self):
        # Game loop
        self.running = True
        while self.running:
            self.timer.tick(self.fps)
            self.screen.fill("white")
            self.draw_board()

            for event in pygame.event.get():
                # Check if the user quite
                self.handle_game_event(event)

            # Remove everything from screen
            pygame.display.flip()

        # Quit game
        pygame.quit()


if __name__ == "__main__":
    Chess().run()
