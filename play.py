from hexchess import Game, RandomPlayer, GreedyPlayer
from engines import QNetworkPlayer
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    players = [RandomPlayer, GreedyPlayer, QNetworkPlayer]
    Game(players=players).run()
