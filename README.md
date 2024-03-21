# Can Chess, With Hexagons? A Reinforcement Learning Exploration.

## Introduction

The goal of this project is to explore the use of reinforcement learning techniques to build a chess engine for gameplay on a non-traditional hexagonal board.

## Setup

In order to install the dependencies, initiate a conda environment from the `environment.yml` file.

```bash
conda env create -f environment.yml
```

If changes are made, the environment can be exported using the provided script.

```bash
bash export-environment.sh
```

## The Game

Hexagonal chess, more specifically the version invented by Władysław Gliński, is played on a non-traditional hexagonal chess board.  
While the board is made up of 91 hexagon tiles, rather than 64 squares, all the familiar pieces are present and their legal movements are heavily inspired by the original game. For a description of the rules, see [Wikipedia](https://en.wikipedia.org/wiki/Hexagonal_chess).

In order to play the game, run `python play.py` from the main directory.

### Missing features

Some of the more intricate rules of the game are still missing, being:

- [ ] Check & Checkmate checks, the game now concludes only when the king is captured.
- [ ] En passant captures for pawns.
- [ ] Pawn promotion.
- [ ] Restart feature in the GUI.

## Reinforcement Learning

TODO

### Models

- [ ] Q networks
  - [x] Development
  - [ ] Training (TODO train white) -> March 21st
  - [ ] Game integration -> March 21st
- [ ] Advantage Actor Critic (A2C) -> March 31st
  - [ ] Development
  - [ ] Training
  - [ ] Game integration
- [ ] Soft Actor Critic -> April 15th
  - [ ] Development
  - [ ] Training
  - [ ] Game integration
- [ ] Monte Carlo Tree Search
  - [ ] Development
  - [ ] Training
  - [ ] Game integration
- [ ] Model Evaluation Notebook -> March 24th
- [ ] Report -> May 3rd
- [ ] Final Presentation -> May 3rd

## Sources

These are some useful references I have used during the development of this project.

- [Can Chess, With Hexagons? - CGP Grey](https://www.youtube.com/watch?v=bgR3yESAEVE)
- [Hexagonal Chess Wikipedia](https://en.wikipedia.org/wiki/Hexagonal_chess)
- [Hexagonal Game Grids](https://www.redblobgames.com/grids/hexagons/)
- [Reference Hex Chess Implementation](https://github.com/AmethystMoon/AmethystMoon.github.io)
  - [Demo](https://amethystmoon.github.io/)
- [Chess Pygame Implementation Tutorial](https://www.youtube.com/watch?v=X-e0jk4I938)
- [Reinforcement Learning for Chess](https://github.com/arjangroen/RLC)

Advantage Actor Critic (A2C)
https://arxiv.org/abs/1602.01783
https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

Soft Actor Critic (for Discrete Action Spaces)
https://arxiv.org/abs/1910.07207
https://arxiv.org/abs/1801.01290
https://spinningup.openai.com/en/latest/algorithms/sac.html#
https://towardsdatascience.com/adapting-soft-actor-critic-for-discrete-action-spaces-a20614d4a50a

## Potential Avenues for Improvement

- Can we have the model learn the representation for states and actions instead of using the one-hot encoded representation?
-
