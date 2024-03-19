"""
Test model competition and game generation
"""

import numpy as np

from yacgo.models import InferenceRandom
from yacgo.train_utils import GameGenerator, ModelCompetition
from yacgo.utils import make_args

args = make_args()


def test_game_gen():
    gen = GameGenerator(model=InferenceRandom(), args=args)
    data = gen.sim_game()

    for d in data:
        print(d.state)
        print(d.value)
        print(d.policy)


def test_compete():
    comp = ModelCompetition(5, model1=None, model2=None, args=args)
    results = comp.compete(100)
    print(results.score)
    print(results.probs)
    print(results.games)
    print(results.bw_wins)
    print(results.raw_bw_games)
    print(results.raw_wb_games)


test_compete()
