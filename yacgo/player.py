import numpy as np
from yacgo.go import game, govars
from yacgo.algos.mcts import MCTSSearch

class Player:
    def __init__(self, player):
        self.player = player

    def action_probs(self, state):
        raise NotImplementedError("Cannot call method of Player")
    
    def best_action(self, state):
        return np.argmax(game.valid_moves(state) * self.action_probs(state))
    
    def sample_action(self, state):
        dist = game.valid_moves(state) * self.action_probs(state)
        dist = dist / np.sum(dist)
        dist[-1] = max(0, 1 - np.sum(dist[0:-1]))
        return np.random.choice(np.arange(game.action_size(state)), p=dist)
    

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
                action_input = input("Move in the form \"x y\" or \"pass\": ")
                if action_input == "pass":
                    action1d = game.action_size(state) - 1
                else:
                    action2d = [int(i) for i in action_input.split(' ')]
                    action1d = game.action_to_1d(state, action2d)
            except Exception as e:
                pass
            else:
                if game.valid_moves(state)[action1d] == 1:
                    success = True

        action[action1d] = 1
        return action
    

class MCTSPlayer(Player):
    def __init__(self, player, model, sims=400, komi=0):
        super().__init__(player)
        self.search = None
        self.model = model
        self.komi = komi
        self.sims = sims

    def action_probs(self, state):
        if self.search is None:
            self.search = MCTSSearch(state, self.model, root=None, noise=False, komi=self.komi)
        else:
            new_root = None
            for c in self.search.root.children:
                if c is not None:
                    for c_p in c.children:
                        if c_p is not None and np.array_equal(c_p.state, state):
                            new_root = c_p

            if new_root is None:
                # Picked an unexplored action, but we can just reset the search
                self.search = MCTSSearch(state, self.model, root=None, noise=False, komi=self.komi)
            else:
                self.search = MCTSSearch(state, self.model, root=new_root, noise=False, komi=self.komi)
        
        self.search.run_sims(self.sims)
        action_probs = self.search.action_probs()
        return action_probs

            
        
        