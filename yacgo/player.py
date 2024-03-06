import numpy as np
from yacgo.go import game, govars
from sklearn.preprocessing import normalize

class Player:
    def __init__(self, player):
        self.player = player

    def action_probs(self, state):
        raise NotImplementedError("Cannot call method of Player")
    
    def best_action(self, state):
        return np.argmax(govars.INVD_CHNL * self.action_probs(state))
    
    def sample_action(self, state):
        return np.random.choice(np.arange(game.action_size(state)), p=game.valid_moves(state) * self.action_probs(state))
    

class RandomPlayer(Player):
    def action_probs(self, state):
        valid = game.valid_moves(state)
        return valid / np.linalg.norm(valid, 1)
    

class HumanPlayer(Player):
    def action_probs(self, state):
        action = np.zeros(game.action_size(state))
        print(game.str(state))
        success = False
        while not success:
            try:
                action_input = input("Move in the form \"x y\": ")
                action2d = [int(i) for i in action_input.split(' ')]
                action1d = game.action_to_1d(state, action2d)
                print(action1d)
            except Exception as e:
                pass
            else:
                if game.valid_moves(state)[action1d] == 1:
                    success = True

        action[action1d] = 1
        return action