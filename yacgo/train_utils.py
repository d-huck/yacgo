"""
Utils for training and evaluating models. 
TODO: Go through and ensure all numpy is yacgo.data.DATA_DTYPE
"""

# pylint: disable=missing-function-docstring,missing-class-docstring

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from tqdm.auto import tqdm
from yacgo.algos.mcts import MCTSSearch
from yacgo.data import TrainState
from yacgo.game import Game
from yacgo.go import game, govars
from yacgo.data import DataGameClientMixin, DATA_DTYPE
from yacgo.player import MCTSPlayer, RandomPlayer
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
                train = np.random.random() < self.pcap_prob
                mcts.run_sims(
                    self.pcap_train
                    if train
                    else np.random.randint(self.pcap_fast, self.pcap_train // 2)
                )
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

            winner = game.winning(state)
            for d in data:
                # TODO: ensure DATA_DTYPE all the way through
                d.value = DATA_DTYPE(winner)  # * game.turn_pm(d.state))
                self.deposit(d)

        except KeyboardInterrupt:
            print("Quitting game generation, closing sockets...")
            self.destroy()

        return data

    def sim_games(self, num_games=1):
        data: List[TrainState] = []
        for _ in num_games:
            data.extend(self.sim_game())
        return data

    # Run until we have some number of training examples
    def sim_data(self, min_data):
        data: List[TrainState] = []
        while len(data) < min_data:
            data.extend(self.sim_game())
        return data
