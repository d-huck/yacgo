from yacgo.train_utils import GameGenerator, ModelCompetition
from yacgo.models import InferenceRandom
import numpy as np

def test_game_gen():
    gen = GameGenerator(5, model=InferenceRandom())
    data = gen.sim_game()

    for d in data:
        print(d.state)
        print(d.value)
        print(d.policy)


def test_compete():
    comp = ModelCompetition(5, model1=None, model2=None, sims=400, komi=0)
    results = comp.compete(100)
    print(results.score)
    print(results.probs)
    print(results.games)
    print(results.bw_wins)
    print(results.raw_bw_games)
    print(results.raw_wb_games)

test_compete()