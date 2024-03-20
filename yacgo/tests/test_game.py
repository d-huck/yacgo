from yacgo.game import Game
from yacgo.player import RandomPlayer, HumanPlayer
from yacgo.go import govars

black = RandomPlayer(govars.BLACK)
white = RandomPlayer(govars.WHITE)

test_game = Game(3, black, white)

test_game.play_full(print_every=True)
