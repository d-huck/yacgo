"""Simple Game Class"""

from yacgo.go import game, govars
from yacgo.player import Player


class Game:
    def __init__(self, board_size, black: Player, white: Player, komi=0):
        self.state = game.init_state(board_size)
        self.players = [black, white]
        self.komi = komi
        self.turn = govars.BLACK
        self.done = False
        self.score = 0

    def step(self):
        if not self.done:
            action = self.players[self.turn].sample_action(self.state)
            self.state = game.next_state(self.state, action)
            self.turn = game.turn(self.state)
            self.done = game.game_ended(self.state)
            if self.done:
                self.score = game.winning(self.state, self.komi)

        return self.score

    def play_full(self, print_every=False, print_final=False):
        while not self.done:
            if print_every:
                print(game.state_to_str(self.state))

            self.step()

        if print_every or print_final:
            print(game.state_to_str(self.state))

        return self.score
