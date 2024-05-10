import pygame
import math
import random
import os
import logging
from hexchess.board import HexChessBoard
from hexchess.pieces import Pawn, Rook, Knight, Bishop, Queen, King
from hexchess.utils import hexagon_dimensions, hexagon_points, qr_to_xy, xy_to_qr
from hexchess.players import RandomPlayer, GreedyPlayer

logger = logging.getLogger(__name__)


class Game:
    colors = {
        0: "#188EAC",  # Dark
        1: "#79A2B4",  # Medium
        2: "#d4e5ef",  # Light
        "active": "#C6878F",
        "green": "#A2AD59",
    }
    states = [
        "home",
        "menu_1",
        "menu_2",
        "playing",
        "finished",
    ]

    def __init__(self, players=[]):
        # Initialize board
        logger.debug("Initializing Game")
        self.set_state("home")
        self.setup()
        self.load_assets()

        # Opponent classes
        self.players = players

        # Menu state
        self.two_player = None
        self.difficulty = None
        self.player_is_white = None
        self.npc = None

    def set_state(self, state):
        """
        Set the state of the game.
        ---
        Args:
        - state (str): The state to set the game to.
        """

        assert state in self.states, f"Unknown state {state}"
        self.state = state

    def setup(self):
        """
        Set up the pygame environment
        ---
        """

        # Start PyGame
        logger.debug("Setup PyGame")
        pygame.init()
        self.screen_w, self.screen_h = 1200, 900
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("Can Chess With Hexagons?")
        self.timer = pygame.time.Clock()
        self.fps = 15

        # Define fonts
        self.text_font = pygame.font.SysFont("Maven Pro", 30)
        self.title_font = pygame.font.SysFont("Maven Pro", 40)

        # Define element sizes
        self.asset_size = 50
        self.logo_size = 250
        self.hex_size = 35
        self.button_w = 400
        self.button_h = 50
        self.button_padding = 50

        # Define button positions
        ## Even buttons
        self.btn_em1_y = self.screen_h / 2 - 50
        self.btn_ep1_y = self.screen_h / 2 + 50
        ## Odd buttons
        self.btn_om1_y = self.screen_h / 2 - 100
        self.btn_o0_y = self.screen_h / 2
        self.btn_op1_y = self.screen_h / 2 + 100

    def load_assets(self):
        """
        Load assets from disk
        ---
        """
        logger.debug("Loading Assets")
        self.assets = {}
        self.assets_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets"
        )
        pieces = ["pawn", "rook", "knight", "bishop", "king", "queen"]
        colors = ["black", "white"]
        for color in colors:
            for piece in pieces:
                piece_asset = pygame.image.load(
                    os.path.join(self.assets_dir, f"{piece}_{color}.svg")
                )
                piece_asset = pygame.transform.scale(
                    piece_asset, (self.asset_size, self.asset_size)
                )
                self.assets[f"{piece}_{color}"] = piece_asset
        logo_asset = pygame.image.load(os.path.join(self.assets_dir, "logo.png"))
        logo_asset = pygame.transform.scale(
            logo_asset, (self.logo_size, self.logo_size)
        )
        self.assets["logo"] = logo_asset

    def draw_piece(self, piece, x_c, y_c):
        """
        Draw a chess piece
        ---
        Args:
        - piece (Piece): The piece to draw.
        - x_c (int): The x coordinate of the center of the piece.
        - y_c (int): The y coordinate of the center of the piece.
        """

        # Obtain piece asset
        piece_color = "white" if piece.is_white else "black"
        piece_asset = self.assets[f"{piece.name}_{piece_color}"]

        # Draw piece
        self.screen.blit(
            piece_asset, (x_c - self.asset_size / 2, y_c - self.asset_size / 2)
        )

    def draw_hexagon(self, q, r, color):
        """
        Draw board hexagon tile
        ---
        Args:
        - q (int): The q coordinate of the hexagon.
        - r (int): The r coordinate of the hexagon.
        - color (str): The color of the hexagon.

        Returns:
        - x_c (int): The x coordinate of the center of the hexagon.
        - y_c (int): The y coordinate of the center of the hexagon.
        """

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

    def draw_text(
        self, text, x, y, size="text", color="black", center_x=True, center_y=True
    ):
        """
        Draw text on the screen
        ---
        Args:
        - text (str): The text to draw.
        - x (int): The x coordinate of the text.
        - y (int): The y coordinate of the text.
        - size (str): The size of the text.
        - color (str): The color of the text.
        - center_x (bool): Whether to center the text horizontally.
        - center_y (bool): Whether to center the text vertically.
        """

        # Generate text render object
        font = self.text_font if size == "text" else self.title_font
        text_obj = font.render(text, True, color)
        # Get test render object size
        _, _, text_w, text_h = text_obj.get_rect()
        # Center object
        if center_x:
            x -= text_w / 2
        if center_y:
            y -= text_h / 2
        # Draw text
        self.screen.blit(text_obj, (x, y))

    def draw_button(self, text, x, y):
        """
        Draw a button on the screen
        ---
        Args:
        - text (str): The text to display on the button.
        - x (int): The x center coordinate of the button.
        - y (int): The y center coordinate of the button.
        """

        pygame.draw.rect(
            self.screen,
            "black",
            (
                x - self.button_w / 2,
                y - self.button_h / 2,
                self.button_w,
                self.button_h,
            ),
            border_radius=10,
        )
        pygame.draw.rect(
            self.screen,
            "white",
            (
                x - self.button_w / 2 + 5,
                y - self.button_h / 2 + 5,
                self.button_w - 10,
                self.button_h - 10,
            ),
            border_radius=10,
        )
        self.draw_text(text, x, y, size="title")

    def draw_buttons(self, button_labels=[]):
        """
        Draw buttons on the screen
        ---
        """

        # Screen center
        x_center = self.screen_w / 2
        y_center = self.screen_h / 2

        # Button offset
        n_btns = len(button_labels)
        y_range = n_btns * self.button_h + (n_btns - 1) * self.button_padding
        y_offset = -y_range / 2

        # Draw buttons
        for btn_label in button_labels:
            # Draw button
            self.draw_button(btn_label, x_center, y_center + y_offset)
            # Update offset
            y_offset += self.button_h + self.button_padding

    def draw_title(self):
        """
        Draw the title on the screen
        ---
        """
        self.draw_text(
            "Can Chess With Hexagons?",
            self.screen_w / 2,
            25,
            size="title",
            center_y=False,
        )

    def draw_home(self):
        """
        Draw the home screen
        ---
        """

        # Draw title
        self.draw_title()

        # Draw logo
        self.screen.blit(
            self.assets["logo"],
            (
                self.screen_w // 2 - self.logo_size // 2,
                80,
            ),
        )

        # Draw 1 player & 2 player buttons
        self.draw_buttons(["1 Player Game", "2 Player Game"])

    def draw_menu(self, menu=1):
        """
        Draw options menu screen
        ---
        Args:
        - menu (int): The menu to draw. 1 for 1 player, 2 for 2 player. Default 1.
        """

        self.draw_title()
        assert menu in [1, 2], f"Unknown menu {menu}"
        button_labels = (
            [f"{engine.name}" for engine in self.players]
            if menu == 1
            else ["White", "Black", "Random"]
        )
        self.draw_buttons(button_labels)

    def draw_board(self):
        """
        Draw the game board on the screen
        ---
        """

        # Draw title above the board
        self.draw_title()

        # Draw game status message above the board
        status_color = "White" if self.turn_is_white else "Black"
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
                x_c, y_c = self.draw_hexagon(q, r, self.colors[hex_color])

                # Draw piece if present
                piece = board_content[(q, r)]
                if piece is not None:
                    self.draw_piece(piece, x_c, y_c)

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
            self.draw_piece(piece, x_c, y_c)
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
            self.draw_piece(piece, x_c, y_c)

    def draw_finished(self):
        """
        Draw the finished screen
        ---
        """

        # Draw title
        self.draw_title()

        # Draw new game button
        self.draw_buttons(["New Game"])

        # Draw winner message
        winner = "White" if self.turn_is_white else "Black"
        winner_message = f"{winner} wins!"
        self.draw_text(
            winner_message,
            self.screen_w / 2,
            self.screen_h / 3,
            size="title",
        )

    def draw_screen(self):
        """
        Draw the screen according to the current program state
        ---
        """

        if self.state == "home":
            self.draw_home()
        elif self.state == "menu_1":
            self.draw_menu(1)
        elif self.state == "menu_2":
            self.draw_menu(2)
        elif self.state == "playing":
            self.draw_board()
        elif self.state == "finished":
            self.draw_finished()
        else:
            raise ValueError(f"Unknown state {self.state}")

    def start_game(self):
        # Game state
        logger.debug("Starting game")
        self.turn_is_white = True  # White moves first
        self.selected_piece = None
        self.legal_moves = []

        # Initialize board
        self.board = HexChessBoard(initialize_empty=False)

        # Initialize NPC if single player game
        if not self.two_player:
            npc_class = self.players[self.difficulty]
            self.npc = npc_class(self.board, not self.player_is_white)

        # Start playing
        self.set_state("playing")

    def detect_button_click(self, event, n_buttons):
        # Make sure button is clicked
        btn_clicked = None
        if not (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
            return btn_clicked

        # Screen center
        x_center = self.screen_w / 2
        y_center = self.screen_h / 2

        # Make sure mouse click is in the right x-range
        x, y = event.pos
        if not (
            x > (x_center - self.button_w / 2) and x < (x_center + self.button_w / 2)
        ):
            return btn_clicked

        # Check if mouse click is in the right y-range
        y_range = n_buttons * self.button_h + (n_buttons - 1) * self.button_padding
        y_offset = -y_range / 2
        for btn_index in range(n_buttons):
            # Check if button is clicked
            if (
                y > y_center + y_offset - self.button_h / 2
                and y < y_center + y_offset + self.button_h / 2
            ):
                return btn_index

            # Update y offset
            y_offset += self.button_h + self.button_padding

        return btn_clicked

    def handle_home_event(self, event):
        # Check click is in button region
        btn_clicked = self.detect_button_click(event, 2)
        if btn_clicked == 0:
            # Start 1 player game
            logger.info("1 player game")
            self.two_player = False
            self.set_state("menu_1")  # choose difficulty
        elif btn_clicked == 1:
            # Start 2 player game
            logger.info("2 player game")
            self.two_player = True
            self.start_game()

    def handle_menu_event(self, event, menu=1):
        assert menu in [1, 2], f"Unknown menu {menu}"
        n_buttons = len(self.players) if menu == 1 else 3
        btn_clicked = self.detect_button_click(event, n_buttons)

        if btn_clicked is None:
            return

        if menu == 1:
            # Menu = choose player
            logger.info(f"Selected NPC Player: {self.players[btn_clicked].name}")
            self.difficulty = btn_clicked
            self.set_state("menu_2")  # choose color
        else:
            # Menu 2 = choose color
            self.player_is_white = btn_clicked == 0
            if btn_clicked == 2:
                self.player_is_white = random.choice([True, False])
            logger.info(f"Playing as {'white' if self.player_is_white else 'black'}")
            self.start_game()  # Start playing

    def get_player_move(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Mouse click at position x,y
            x, y = event.pos

            # Check if coordinate is on the board
            pos = xy_to_qr(x - self.screen_w / 2, y - self.screen_h / 2, self.hex_size)
            q, r = pos
            s = -q - r
            if max([q, r, s]) <= 5 and min([q, r, s]) >= -5:
                # Piece selected
                selected_piece = self.board.board[pos]
                logger.debug(f"Piece selected: {selected_piece}")
                if (
                    selected_piece is not None
                    and selected_piece.is_white == self.turn_is_white
                ):
                    # Select a piece to play
                    if self.selected_piece == pos:
                        # Unselect piece if selected
                        self.selected_piece = None
                        self.legal_moves = []
                    else:
                        # Piece selected
                        self.selected_piece = pos
                        self.legal_moves = self.board.get_legal_moves(
                            self.selected_piece, self.turn_is_white
                        )
                    return None
                elif self.selected_piece is not None and pos in self.legal_moves:
                    # Reset selected piece & legal moves
                    pos_from = self.selected_piece
                    self.selected_piece = None
                    self.legal_moves = []
                    # Move selected
                    return (pos_from, pos)

    def handle_game_event(self, event):
        # Get next move
        if self.two_player or self.turn_is_white == self.player_is_white:
            # Player to move
            move = self.get_player_move(event)
        else:
            # NPC to move
            move = self.npc.get_move()

        # Perform next move
        if move is not None:
            # Perform move
            position_from, position_to = move
            success, finished = self.board.move(
                position_from, position_to, self.turn_is_white
            )
            assert success, "Move did not succeed!"

            # Check if game is finished
            if finished:
                self.set_state("finished")
            else:
                # Advance turn
                self.turn_is_white = not self.turn_is_white

    def handle_finished_event(self, event):
        btn_clicked = self.detect_button_click(event, 1)
        if btn_clicked == 0:
            # Restart game
            self.set_state("home")

    def handle_event(self, event):
        if self.state == "home":
            self.handle_home_event(event)
        elif self.state == "menu_1":
            self.handle_menu_event(event, 1)
        elif self.state == "menu_2":
            self.handle_menu_event(event, 2)
        elif self.state == "playing":
            self.handle_game_event(event)
        elif self.state == "finished":
            self.handle_finished_event(event)
        else:
            raise ValueError(f"Unknown state {self.state}")

    def run(self):
        # Game loop
        self.running = True
        while self.running:
            self.timer.tick(self.fps)
            self.screen.fill("white")
            self.draw_screen()

            for event in pygame.event.get():
                # Check if the user quit
                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    self.handle_event(event)

            # Remove everything from screen
            pygame.display.flip()

        # Quit game
        pygame.quit()


if __name__ == "__main__":
    Game().run()
