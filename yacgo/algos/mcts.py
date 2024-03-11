import numpy as np
from typing import List

from yacgo.go import game, govars

class MCTSSearch:
    def __init__(self, state, model, root=None, noise=False, c_puct=1.1, komi=0):
        self.state = state
        self.model = model
        self.c_puct = c_puct
        self.action_dim = game.action_size(state)
        self.noise = noise
        self.komi = komi
        self.sims_run = 0
        if root is None:
            self.root: MCTSNode = MCTSNode(state, parent=None, search=self)
        else:
            assert np.array_equal(root.state, state)
            self.root: MCTSNode = root
            self.sims_run = self.root.total_visits
            self.root.parent = None
        
    def sim(self):
        child = self.root.best_child(self.c_puct)
        child.parent.backup(-child.value)

    def run_sims(self, sims=400):
        self.sims_run += sims
        for _ in range(sims):
            self.sim()

    def best_action(self):
        if self.sims_run < self.root.valid_move_count:
            raise ValueError(f"# sims must be at least {self.root.valid_move_count}")
        
        return np.argmax(self.action_probs())
    
    def action_probs(self):
        if self.sims_run < self.root.valid_move_count:
            raise ValueError(f"# sims must be at least {self.root.valid_move_count}")
        
        scores = [c.total_visits / self.root.total_visits if c is not None else 0 for c in self.root.children]
        
        e = np.exp(scores)
        return e / np.sum(e)

    def action_probs_nodes(self):
        return self.action_probs(), self.root.children


class MCTSNode:
    def __init__(self, state, parent, search: MCTSSearch):
        self.search = search
        self.state = state
        self.total_visits = 1
        self.parent: MCTSNode = parent
        self.terminal = False
        self.valid_moves = game.valid_moves(state)
        self.valid_move_count = sum(self.valid_moves)
        self.children: List[MCTSNode] = [None] * search.action_dim
        if game.game_ended(state):
            self.value = game.winning(state, self.search.komi) * game.turn_pm(state)
            self.terminal = True
        else:
            self.value, self.policy = search.model.forward(state) # TODO: Can we remove invalid moves from channel features?

    def value_score(self):
        return self.value / self.total_visits
    
    def noisy_policy(self, i):
        if not self.search.noise:
            return self.policy[i]
        else:
            return 0.75 * self.policy[i] \
                + 0.25 * np.random.dirichlet(0.03 * game.action_size(self.state) / (self.children[i].valid_move_count + 1))

    def best_child(self, c_puct=1.1):
        if self.terminal: # TODO: we could also return None to prevent an extra backprop if we want
            self.total_visits += 1
            return self
            
        # TODO: refactor for clarity
        puct = [(-c.value_score() if c is not None else 0) + c_puct * self.noisy_policy(i) * np.sqrt(self.total_visits - 1) / (1 + (self.children[i].total_visits if c is not None else 0)) 
            if self.valid_moves[i] == 1 else -np.inf for i, c in enumerate(self.children)]

        # print(puct)
        child_to_expand = np.argmax(puct)
        if self.children[child_to_expand] is None:
            self.children[child_to_expand] = MCTSNode(game.next_state(self.state, child_to_expand), parent=self, search=self.search)
            return self.children[child_to_expand]
        else:
            return self.children[child_to_expand].best_child()

    def backup(self, value):
        self.value += value
        self.total_visits += 1
        if self.parent is not None:
            self.parent.backup(-value)