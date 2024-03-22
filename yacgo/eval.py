""""
Code for evaluating networks.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from tqdm.auto import tqdm

from yacgo.game import Game
from yacgo.go import game, govars
from yacgo.models import Model
from yacgo.player import MCTSPlayer, RandomPlayer


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
        self.pbar = tqdm(total=1, unit="game", postfix={"score": 0.0})

    def compete(self, num_games=1) -> CompetitionResult:
        bw_games = num_games // 2
        wb_games = num_games - bw_games
        scores = []
        self.pbar.reset(total=num_games)

        def avg_score():
            return f"{sum(scores) / len(scores):04.4f}"

        raw_bw_results = []
        self.pbar.set_description("Black vs White")
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
            scores.append(result)
            self.pbar.set_postfix({"score": sum(scores)})
            self.pbar.update(1)
            raw_bw_results.append(result)

        raw_wb_results = []
        self.pbar.set_description("White vs Black")
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
            scores.append(result)
            self.pbar.set_postfix({"score": sum(scores)})
            self.pbar.update(1)
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
