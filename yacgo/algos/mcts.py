import numpy as np
from typing import List

from yacgo.go import game, govars

class MCTSSearch:
    def __init__(self, state, model, c_puct=1.1, noise=False, komi=0):
        self.state = state
        self.model = model
        self.c_puct = c_puct
        self.action_dim = game.action_size(state)
        self.noise = noise
        self.komi = komi
        self.sims_run = 0
        self.root: MCTSNode = MCTSNode(state, parent=None, search=self)
        
    def sim(self):
        child = self.root.best_child(self.c_puct)
        child.parent.backup(child.value)

    def run_sims(self, sims=400):
        self.sims_run += sims
        for _ in range(sims):
            self.sim()

    def best_action(self):
        if self.sims_run < self.root.valid_move_count:
            raise ValueError(f"# sims must be at least {self.root.valid_move_count}")
        
        return np.best_child(c_puct=0)
    
    def action_probs(self):
        if self.sims_run < self.root.valid_move_count:
            raise ValueError(f"# sims must be at least {self.root.valid_move_count}")
        
        scores = [c.total_visits / self.root.total_visits if c is not None else 0 for c in self.root.children]
        # print(scores)
        e = np.exp(scores)
        return e / np.sum(e)



class MCTSNode:
    def __init__(self, state, parent, search: MCTSSearch):
        self.search = search
        self.state = state
        self.total_visits = 1
        self.parent: MCTSNode = parent
        self.terminal = False
        if game.game_ended(state):
            self.valid_move_count = 0
            self.value = game.winning(state, self.search.komi)
            self.terminal = True
        else:
            # self.unexplored_moves = game.valid_moves(state)
            self.valid_moves = game.valid_moves(state)
            self.valid_move_count = sum(self.valid_moves)
            self.children: List[MCTSNode] = [None] * search.action_dim
            self.child_visits = [0] * search.action_dim
            self.value, self.policy = search.model.forward_state(state)
            self.policy *= self.valid_moves

    def value_score(self):
        return self.value / self.total_visits
    
    def noisy_policy(self, i):
        if not self.search.noise:
            return self.policy[i]
        else:
            return 0.75 * self.policy[i] + 0.25 * np.random.dirichlet(10.83 / (self.children[i].valid_move_count + 1))

    def best_child(self, c_puct=1.1):
        if self.terminal:
            return self
 
        puct = [c.value_score() * game.turn_pm(self.state) + c_puct * self.noisy_policy(i) * np.sqrt(self.total_visits) / (1 + self.child_visits[i]) 
            if c is not None else (0 if self.valid_moves[i] == 1 else -np.inf) for i, c in enumerate(self.children)]

        # print(puct)
        child_to_expand = np.argmax(puct)
        if self.children[child_to_expand] is None:
            self.children[child_to_expand] = MCTSNode(game.next_state(self.state, child_to_expand), parent=self, search=self.search)
            return self.children[child_to_expand]
        else:
            self.child_visits[child_to_expand] += 1
            return self.children[child_to_expand].best_child()

    def backup(self, value):
        self.value += value
        self.total_visits += 1
        if self.parent is not None:
            self.parent.backup(value)