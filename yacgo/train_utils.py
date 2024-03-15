from yacgo.player import MCTSPlayer, RandomPlayer
from yacgo.algos.mcts import MCTSSearch, MCTSNode
from yacgo.go import game, govars
from yacgo.game import Game
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple

@dataclass
class TrainState:
    state: np.ndarray
    value: np.float32
    policy: np.ndarray

class GameGenerator:
    def __init__(self, board_size, model, komi=0, pcap_train=400, pcap_fast=100, pcap_prob=0.25):
        self.board_size = board_size
        self.model = model
        self.komi = komi
        self.pcap_train = pcap_train
        self.pcap_fast = pcap_fast
        self.pcap_prob = pcap_prob
        
    def sim_game(self):
        data: List[TrainState] = []
        state = game.init_state(self.board_size)
        mcts = MCTSSearch(state, self.model, root=None, noise=True)
        while not game.game_ended(state):
            train = np.random.random() < self.pcap_prob
            mcts.run_sims(self.pcap_train if train else self.pcap_fast)
            action_probs, nodes = mcts.action_probs_nodes()
            if train:
                data.append(TrainState(state, 0, action_probs))
            
            action = np.random.choice(np.arange(game.action_size(state)), p=action_probs)
            state = game.next_state(state, action)
            mcts = MCTSSearch(state, self.model, root=nodes[action], noise=True)

        winner = game.winning(state)
        for d in data:
            d.value = winner * game.turn_pm(d.state)

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
    score: int # sum of results (+ means model1 performed better)
    probs: Tuple[int] # (m1 win%, m2 win%)
    games: np.ndarray # [-1, 1, 1, -1, 0, ...] (+ means model1 win)
    bw_wins: Tuple[int] # (%black win, %white win) aggregate - just for analysis
    raw_bw_games: np.ndarray
    raw_wb_games: np.ndarray


class ModelCompetition:
    def __init__(self, board_size, model1, model2, sims=400, komi=0):
        self.board_size = board_size
        self.model1 = model1
        self.model2 = model2
        self.sims = sims
        self.komi = komi

    def compete(self, num_games=1) -> CompetitionResult:
        bw_games = num_games // 2
        wb_games = num_games - bw_games

        raw_bw_results = []
        for _ in range(bw_games):
            if self.model1 is None:
                b_player = RandomPlayer(govars.BLACK)
            else:
                b_player = MCTSPlayer(govars.BLACK, self.model1, self.sims, self.komi)
            if self.model2 is None:
                w_player = RandomPlayer(govars.WHITE)
            else:
                w_player = MCTSPlayer(govars.WHITE, self.model2, self.sims, self.komi)

            g = Game(self.board_size, b_player, w_player, self.komi)
            result = g.play_full()
            raw_bw_results.append(result)
        
        raw_wb_results = []
        for _ in range(wb_games):
            if self.model2 is None:
                b_player = RandomPlayer(govars.BLACK)
            else:
                b_player = MCTSPlayer(govars.BLACK, self.model2, self.sims, self.komi)
            if self.model1 is None:
                w_player = RandomPlayer(govars.WHITE)
            else:
                w_player = MCTSPlayer(govars.WHITE, self.model1, self.sims, self.komi)
            g = Game(self.board_size, b_player, w_player, self.komi)
            result = g.play_full()
            raw_wb_results.append(result)

        raw_bw_results = np.array(raw_bw_results)
        raw_wb_results = np.array(raw_wb_results)
        score = np.sum(raw_bw_results) - np.sum(raw_wb_results)
        games = np.concatenate((raw_bw_results, raw_wb_results * -1))
        probs = [(np.count_nonzero(games == 1) / num_games),
                 (np.count_nonzero(games == -1) / num_games)]
        bw_wins = [(np.count_nonzero(raw_bw_results == 1) + np.count_nonzero(raw_wb_results == 1)) / num_games,
                   (np.count_nonzero(raw_bw_results == -1) + np.count_nonzero(raw_wb_results == -1)) / num_games]

        return CompetitionResult(score, probs, games, bw_wins, raw_bw_results, raw_wb_results)
    