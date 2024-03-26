""""
Code for evaluating networks.
"""

from dataclasses import dataclass
from multiprocessing import Lock, Pool
import signal
from typing import Tuple

import numpy as np
import wandb
from tqdm.auto import tqdm

from yacgo.game import Game
from yacgo.go import govars
from yacgo.models import InferenceClient
from yacgo.player import MCTSPlayer, RandomPlayer

# mp.set_start_method("fork")
BW_GAME = 0
WB_GAME = 1


def play_game(game_args):
    """Plays a single game between two models. Since this is multithreaded,
    it can only take a single argument and cannot be a class method.

    Args:
        game_args (_type_): list of arguments for the game (model1, model2,
            args, komi). modelX is expected to be the port on which an
            InferenceClient can connect to the server for the model. If
            modelX is None, then RandomPlayer will be used

    Returns:
        _type_: _description_
    """
    try:
        model1, model2, args, komi, order = game_args

        if model1 is None:
            p1 = RandomPlayer(govars.BLACK)
        else:
            model = InferenceClient([model1])
            p1 = MCTSPlayer(govars.BLACK, model, args)
        if model2 is None:
            p2 = RandomPlayer(govars.WHITE)
        else:
            model = InferenceClient([model2])
            p2 = MCTSPlayer(govars.WHITE, model, args)

        g = Game(args.board_size, p1, p2, komi)

        result = g.play_full()
        return order, result
    except KeyboardInterrupt:
        return order, 0


@dataclass
class CompetitionResult:
    """Dataclass for holding competition results."""

    score: int  # sum of results (+ means model1 performed better)
    probs: Tuple[int]  # (m1 win%, m2 win%)
    games: np.ndarray  # [-1, 1, 1, -1, 0, ...] (+ means model1 win)
    bw_wins: Tuple[int]  # (%black win, %white win) aggregate - just for analysis
    raw_bw_games: np.ndarray
    raw_wb_games: np.ndarray


class ModelCompetition:
    """
    Class for handling model competitions between two models.
    If either model is None, a random player will be used.
    """

    def __init__(self, model1: int, model2: int, args):
        self.board_size = args.board_size
        self.model1 = model1
        self.model2 = model2
        self.sims = args.n_simulations
        self.komi = args.komi
        self.args = args
        self.pbar = tqdm(
            total=1, unit="game", postfix={"score": 0.0}, smoothing=0.001, leave=False
        )
        self.lock = Lock()
        self.n_workers = args.num_games
        self.scores = []
        self.wandb = args.wandb

    def compete(self, num_games=1) -> CompetitionResult:
        """Run competition between two models.

        Args:
            num_games (int, optional): Number of games in competition. Defaults to 1.

        Returns:
            CompetitionResult: results of the competition
        """
        bw_games = num_games // 2
        wb_games = num_games - bw_games

        self.pbar.set_description("Running Competition")
        self.pbar.reset(total=num_games)
        n_workers = min(self.n_workers, num_games)
        raw_bw_results = []
        raw_wb_results = []
        comp_args = [
            (self.model1, self.model2, self.args, self.komi, "bw")
            for _ in range(bw_games)
        ]
        comp_args += [
            (self.model2, self.model1, self.args, self.komi, "wb")
            for _ in range(wb_games)
        ]
        with Pool(n_workers) as p:
            for order, result in p.imap_unordered(play_game, comp_args):
                if order == "bw":
                    raw_bw_results.append(result)
                    self.scores.append(result)
                elif order == "wb":
                    raw_wb_results.append(result)
                    self.scores.append(result * -1)
                self.pbar.update(1)
                self.pbar.set_postfix({"score": sum(self.scores)})
        self.pbar.close()

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


if __name__ == "__main__":
    from yacgo.utils import make_args

    args = make_args()

    comp = ModelCompetition(None, None, args)
    result = comp.compete(num_games=1024)
    print(result)
