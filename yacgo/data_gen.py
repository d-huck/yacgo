from yacgo.player import Player
from yacgo.algos.mcts import MCTSSearch, MCTSNode
from yacgo.go import game, govars
from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class TrainState:
    state: np.ndarray
    value: np.float32
    policy: np.ndarray

class GameGenerator():
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
            
            action = np.random.choice(np.arange(game.action_size(state), p=action_probs))
            state = game.next_state(state, action)
            mcts = MCTSSearch(state, self.model, root=nodes[action], noise=True)

        winner = game.winning(state)
        for d in data:
            d.value = winner * game.turn_pm(d.state)