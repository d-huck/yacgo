"""
Logic for MCTS search
"""

# pylint: disable=missing-function-docstring,missing-class-docstring
import numpy as np

from yacgo.go import game


class MCTSSearch:
    def __init__(self, state, model, args: dict, noise=False, root=None):
        self.state = state
        self.model = model
        self.c_puct = args.c_puct
        self.action_dim = game.action_size(state)
        self.noise = args.mcts_noise and noise  # consider args option for noise
        self.komi = args.komi
        self.pcap_fast = args.pcap_fast
        self.sims_run = 0
        if root is None:
            self.root: MCTSNode = MCTSNode(state, parent=None, search=self)
            self.root.initialize()
        else:
            assert np.array_equal(root.state, state)
            self.root: MCTSNode = root
            self.sims_run = self.root.total_visits
            self.root.parent = None
            if not root.initialized:
                root.initialize()

    def sim(self):
        child = self.root.best_child(self.c_puct)
        child.parent.backup(-child.value)

    def run_sims(self, sims=400):
        self.sims_run += sims
        for _ in range(sims):
            self.sim()

    def best_action(self):
        if self.sims_run < self.pcap_fast:
            raise ValueError(f"# sims must be at least {self.pcap_fast}")

        return np.argmax(self.action_probs())

    def action_probs(self):
        if self.sims_run < self.root.pcap_fast:
            raise ValueError(f"# sims must be at least {self.pcap_fast}")

        scores = [
            c.total_visits / self.root.total_visits if c is not None else -np.inf
            for c in self.root.children
        ]

        e = np.exp(scores)
        return e / np.sum(e)

    def action_probs_nodes(self):
        return self.action_probs(), self.root.children


class MCTSNode:
    def __init__(self, state, parent, search: MCTSSearch):
        self.search = search
        self.state = state
        self.total_visits = 0
        self.initialized = False
        self.parent: MCTSNode = parent
        self.valid_moves = game.valid_moves(state)
        self.valid_move_count = sum(self.valid_moves)
        self.value = 0
        # self.children: List[MCTSNode] = [None] * search.action_dim

    def initialize(self):
        self.initialized = True
        self.total_visits += 1
        if game.game_ended(self.state):
            self.value = game.winning(self.state, self.search.komi) * game.turn_pm(
                self.state
            )
            self.terminal = True
        else:
            self.value, self.policy = self.search.model.forward(
                self.state
            )  # TODO: Can we remove invalid moves from channel features?
            self.terminal = False
            self.children = [
                (
                    (
                        MCTSNode(
                            game.next_state(self.state, a),
                            parent=self,
                            search=self.search,
                        )
                    )
                    if self.valid_moves[a] == 1
                    else None
                )
                for a in range(game.action_size(self.state))
            ]

    def value_score(self):
        if self.value == 0:
            return 0
        return self.value / self.total_visits

    def noisy_policy(self):
        if not self.search.noise:
            return self.policy
        else:
            return 0.75 * self.policy + 0.25 * np.random.dirichlet(
                [
                    (
                        0.03 * game.action_size(self.state) / (c.valid_move_count + 1)
                        if c is not None
                        else np.nextafter(0, 1)
                    )
                    for c in self.children
                ]
            )

    def best_child(self, c_puct=1.1):
        if (
            self.terminal
        ):  # TODO: we could also return None to prevent an extra backprop if we want
            self.total_visits += 1
            return self

        # TODO: refactor for clarity
        noisy_policy = self.noisy_policy()

        puct = [
            (
                -c.value_score()
                + c_puct
                * noisy_policy[i]
                * np.sqrt(self.total_visits - 1)
                / (1 + self.children[i].total_visits)
                if self.valid_moves[i] == 1
                else -np.inf
            )
            for i, c in enumerate(self.children)
        ]

        # print(puct)
        child_to_expand = np.argmax(puct)
        if not self.children[child_to_expand].initialized:
            self.children[child_to_expand].initialize()
            return self.children[child_to_expand]
        else:
            return self.children[child_to_expand].best_child()

    def backup(self, value):
        self.value += value
        self.total_visits += 1
        if self.parent is not None:
            self.parent.backup(-value)
