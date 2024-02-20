# Can Chess, With Hexagons? A Reinforcement Learning Exploration.

## Introduction

The goal of this project is to explore the use of reinforcement learning techniques to build a chess engine for gameplay on a non-traditional hexagonal board.

## The Game

Hexagonal chess, more specifically the version invented by Władysław Gliński, is played on a non-traditional hexagonal chess board.  
While the board is made up of 91 hexagon tiles, rather than 64 squares, all the familiar pieces are present and their legal movements are heavily inspired by the original game. For a description of the rules, see [Wikipedia](https://en.wikipedia.org/wiki/Hexagonal_chess).

The game was implemented in Python with minimal dependencies, only `pygame==2.5.2` which has been used to develop the user interface.  
In order to play the game, run `pyton play.py` from the main directory.

### Missing features

Some of the more intricate rules of the game are still missing, being:

- [ ] Check & Checkmate checks, the game now concludes only when the king is captured.
- [ ] En passant captures for pawns.
- [ ] Pawn promotion.
- [ ] Restart feature in the GUI.

## Reinforcement Learning

TODO

## Sources

These are some useful references I have used during the development of this project.

- [Can Chess, With Hexagons? - CGP Grey](https://www.youtube.com/watch?v=bgR3yESAEVE)
- [Hexagonal Chess Wikipedia](https://en.wikipedia.org/wiki/Hexagonal_chess)
- [Hexagonal Game Grids](https://www.redblobgames.com/grids/hexagons/)
- [Reference Hex Chess Implementation](https://github.com/AmethystMoon/AmethystMoon.github.io)
  - [Demo](https://amethystmoon.github.io/)
- [Chess Pygame Implementation Tutorial](https://www.youtube.com/watch?v=X-e0jk4I938)
- [Reinforcement Learning for Chess](https://github.com/arjangroen/RLC)
