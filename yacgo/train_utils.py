"""
Utils for training and evaluating models. 
"""

# pylint: disable=missing-function-docstring,missing-class-docstring

import gc
from typing import List

import numpy as np

from yacgo.algos.mcts import MCTSSearch
from yacgo.data import TrainState
from yacgo.go import game
from yacgo.data import DataGameClientMixin
from yacgo.models import Model


class GameGenerator(DataGameClientMixin):
    def __init__(self, model: Model, args: dict, display: bool = False):
        super().__init__(args)

        self.board_size = args.board_size
        self.model = model
        self.komi = args.komi
        self.pcap_train = args.pcap_train
        self.pcap_fast = args.pcap_fast
        self.pcap_prob = args.pcap_prob
        self.max_turns = self.board_size * self.board_size * 2
        self.n_turns = 0
        self.args = args
        self.display = display

    def sim_game(self):
        try:
            data: List[TrainState] = []
            state = game.init_state(self.board_size)
            if self.display:
                print(game.state_to_str(state))
            mcts = MCTSSearch(state, self.model, self.args, noise=True, root=None)
            while not game.game_ended(state):
                if self.n_turns >= self.max_turns:
                    break
                train = np.random.random() < self.pcap_prob
                mcts.run_sims(self.pcap_train if train else self.pcap_fast)
                action_probs, nodes = mcts.action_probs_nodes()
                if train:
                    data.append(TrainState(state, np.float32(0.0), action_probs))

                action = np.random.choice(
                    np.arange(game.action_size(state)), p=action_probs
                )
                state = game.next_state(state, action)
                if self.display:
                    print(game.state_to_str(state))
                mcts = MCTSSearch(
                    state, self.model, self.args, root=nodes[action], noise=True
                )
                self.n_turns += 1

            for d in data:
                self.deposit(d)
            gc.collect()

        except KeyboardInterrupt:
            print("Quitting game generation, closing sockets...")
            self.destroy()

        return data

    def sim_games(self, num_games=1):
        data: List[TrainState] = []
        for _ in range(num_games):
            data.extend(self.sim_game())
        return data

    # Run until we have some number of training examples
    def sim_data(self, min_data):
        data: List[TrainState] = []
        while len(data) < min_data:
            data.extend(self.sim_game())
        return data
