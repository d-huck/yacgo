from yacgo.go import game, govars
from yacgo.player import RandomPlayer
from yacgo.algos.mcts import MCTSSearch
from yacgo.models import InferenceRandom
from yacgo.utils import make_args, set_args

import time

def run_next_state_test(state, next_action):
    for _ in range(10000):
        next_state = game.next_state(state, next_action)


def run_mcts_test(state):
    args = set_args(n_simulations=10000)
    search = MCTSSearch(state, InferenceRandom(), args, noise=True)
    search.run_sims(2000)
    print(game.next_state_called)

state = game.init_state(9)
p1 = RandomPlayer(govars.BLACK)
p2 = RandomPlayer(govars.WHITE)

for _ in range(10):
    state = game.next_state(state, p1.best_action(state))
    state = game.next_state(state, p2.best_action(state))


# next_action = p1.best_action(state)
start = time.time()

run_mcts_test(state)

total_time = time.time() - start
print(total_time)