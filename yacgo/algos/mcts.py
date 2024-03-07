import numpy as np
from typing import List

from yacgo.go import game, govars

class MCTSSearch:
    def __init__(self, state, model, c_puct, noise=False, komi=0):
        self.model = model
        self.c_puct = c_puct
        self.root = MCTSNode(state, parent=None, search=self)
        self.noise = noise
        self.komi = komi

    def sim(self):
        child = self.root.expand()
        child.parent.backup(child.value)

    def best_action(self, sims=400):
        for _ in sims:
            self.sim()

        return np.argmax([c.value_score() for c in self.root.children])

class MCTSNode:
    def __init__(self, state, parent, search: MCTSSearch):
        self.search = search
        self.state = state
        self.total_vists = 1
        self.parent: MCTSNode = parent
        self.terminal = False
        if game.game_ended(state):
            self.valid_move_count = 0
            score_diff = game.winning(state, self.search.komi)
            if score_diff > 0:
                self.value = 1
            elif score_diff == 0:
                self.value = 0
            else:
                self.value = -1
            
            self.terminal = True
        else:
            self.unexplored_moves = game.valid_moves(state)
            self.valid_move_count = sum(self.unexplored_moves)
            self.children: List[MCTSNode] = [None] * len(self.unexplored_moves)
            self.child_visits = [0] * len(self.unexplored_moves)
            self.value, self.policy = search.model(state)
            self.policy *= self.unexplored_moves

    def value_score(self):
        return self.value / self.total_vists
    
    def noisy_policy(self, i):
        if not self.search.noise:
            return self.policy[i]
        else:
            return 0.75 * self.policy[i] + 0.25 * np.random.dirichlet(10.83 / (self.children[i].valid_move_count + 1))

    def expand(self):
        if self.terminal:
            return self
        
        if np.any(self.unexplored_moves):
            # Pick favored unexplored move. Could sample for increased variance?
            options = self.policy * self.unexplored_moves
            best_new_action = np.argmax(options)
            self.unexplored_moves[best_new_action] = 0
            self.children[best_new_action] = MCTSNode(game.next_state(self.state, best_new_action), parent=self, search=self.search)
        
            return self.children[best_new_action]

        else:
            # Use PUCT to choose node to expand
            puct = [c.value_score() * game.turn_pm(self.state) + self.search.c_puct * self.noisy_policy(i) * np.sqrt(self.total_vists) / (1 + self.child_visits[i]) 
                if c is not None else -np.inf for i, c in enumerate(self.children)]

            child_to_expand = np.argmax(puct)
            self.child_visits[child_to_expand] += 1
            return self.children[child_to_expand].expand()

    def backup(self, value):
        self.value += value
        self.total_vists += 1
        self.backup(self.parent)