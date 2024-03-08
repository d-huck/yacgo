from yacgo.game import Game
from yacgo.player import RandomPlayer, HumanPlayer
from yacgo.go import govars

black = HumanPlayer(govars.BLACK)
white = RandomPlayer(govars.WHITE)

test_game = Game(9, black, white)

test_game.play_full(print_every=False)
