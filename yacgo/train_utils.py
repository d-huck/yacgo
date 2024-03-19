"""
Utils for training and evaluating models. 
"""

# pylint: disable=missing-function-docstring,missing-class-docstring

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from yacgo.algos.mcts import MCTSSearch
from yacgo.data import TrainState
from yacgo.game import Game
from yacgo.go import game, govars
from yacgo.data import DataGameClientMixin
from yacgo.player import MCTSPlayer, RandomPlayer
from yacgo.models import Model


class GameGenerator(DataGameClientMixin):
    def __init__(
        self,
        model: Model,
        args: dict,
    ):
        super().__init__(args)
        self.board_size = args.board_size
        self.model = model
        self.komi = args.komi
        self.pcap_train = args.pcap_train
        self.pcap_fast = args.pcap_fast
        self.pcap_prob = args.pcap_prob
        self.args = args

    def sim_game(self):
        data: List[TrainState] = []
        state = game.init_state(self.board_size)
        mcts = MCTSSearch(state, self.model, self.args, noise=True, root=None)
        while not game.game_ended(state):
            train = np.random.random() < self.pcap_prob
            mcts.run_sims(self.pcap_train if train else self.pcap_fast)
            action_probs, nodes = mcts.action_probs_nodes()
            if train:
                data.append(TrainState(state, 0, action_probs))

            action = np.random.choice(
                np.arange(game.action_size(state)), p=action_probs
            )
            state = game.next_state(state, action)
            mcts = MCTSSearch(
                state, self.model, self.args, root=nodes[action], noise=True
            )

        winner = game.winning(state)
        for d in data:
            d.value = winner * game.turn_pm(d.state)
            self.deposit(d)

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


@dataclass
class CompetitionResult:
    score: int  # sum of results (+ means model1 performed better)
    probs: Tuple[int]  # (m1 win%, m2 win%)
    games: np.ndarray  # [-1, 1, 1, -1, 0, ...] (+ means model1 win)
    bw_wins: Tuple[int]  # (%black win, %white win) aggregate - just for analysis
    raw_bw_games: np.ndarray
    raw_wb_games: np.ndarray


class ModelCompetition:
    def __init__(self, board_size, model1, model2, args):
        self.board_size = board_size

        self.model1 = model1
        self.model2 = model2
        self.sims = args.n_simulations
        self.komi = args.komi
        self.args = args

    def compete(self, num_games=1) -> CompetitionResult:
        bw_games = num_games // 2
        wb_games = num_games - bw_games

        raw_bw_results = []
        for _ in range(bw_games):
            if self.model1 is None:
                b_player = RandomPlayer(govars.BLACK)
            else:
                b_player = MCTSPlayer(govars.BLACK, self.model1, self.args)
            if self.model2 is None:
                w_player = RandomPlayer(govars.WHITE)
            else:
                w_player = MCTSPlayer(govars.WHITE, self.model2, self.args)

            g = Game(self.board_size, b_player, w_player, self.komi)
            result = g.play_full()
            raw_bw_results.append(result)

        raw_wb_results = []
        for _ in range(wb_games):
            if self.model2 is None:
                b_player = RandomPlayer(govars.BLACK)
            else:
                b_player = MCTSPlayer(govars.BLACK, self.model2, self.args)
            if self.model1 is None:
                w_player = RandomPlayer(govars.WHITE)
            else:
                w_player = MCTSPlayer(govars.WHITE, self.model1, self.args)
            g = Game(self.board_size, b_player, w_player, self.komi)
            result = g.play_full()
            raw_wb_results.append(result)

        raw_bw_results = np.array(raw_bw_results)
        raw_wb_results = np.array(raw_wb_results)
        score = np.sum(raw_bw_results) - np.sum(raw_wb_results)
        games = np.concatenate((raw_bw_results, raw_wb_results * -1))
        probs = [
            (np.count_nonzero(games == 1) / num_games),
            (np.count_nonzero(games == -1) / num_games),
        ]
        bw_wins = [
            (
                np.count_nonzero(raw_bw_results == 1)
                + np.count_nonzero(raw_wb_results == 1)
            )
            / num_games,
            (
                np.count_nonzero(raw_bw_results == -1)
                + np.count_nonzero(raw_wb_results == -1)
            )
            / num_games,
        ]

        return CompetitionResult(
            score, probs, games, bw_wins, raw_bw_results, raw_wb_results
        )
