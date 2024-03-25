from yacgo.data import DataBroker
from yacgo.go import game, govars
from yacgo.player import MCTSPlayer
from yacgo.models import InferenceRandom
from yacgo.utils import make_args

import numpy as np

black_test_state = np.array([
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ],
    [
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
    ],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
])

p1 = MCTSPlayer(govars.BLACK, InferenceRandom(), make_args())
policy = p1.action_probs(black_test_state)

print(policy)

new_state, new_val, new_pol = DataBroker.random_symmetry(black_test_state, 1, policy)

print(game.state_to_str(new_state))
print(new_state)
print(new_val)
print(new_pol)
