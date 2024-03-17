from yacgo.go import game, govars
from yacgo.game import Game
from yacgo.player import Player, RandomPlayer, HumanPlayer, MCTSPlayer
from yacgo.algos.mcts import MCTSSearch
from yacgo.models import InferenceRandom
import numpy as np
    
model = InferenceRandom()

def game_test():
    black = MCTSPlayer(govars.BLACK, model, sims=400, komi=1)
    white = RandomPlayer(govars.WHITE)
    test_game = Game(5, black, white)
    test_game.play_full(print_every=True)

def custom_state_test():
    black_test_state = [
        [[0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
        ],
        [[0, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
        ],
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
        ],
        [[0, 1, 0],
        [1, 1, 1],
        [0, 0, 0]
        ],
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
        ],
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
        ]
    ]

    white_test_state = [
        [[0, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
        ],
        [[0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
        ],
        [[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
        ],
        [[0, 1, 0],
        [1, 1, 1],
        [0, 0, 0]
        ],
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
        ],
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
        ]
    ]

    test_state = np.array(black_test_state)
    print(game.str(test_state))
    mcts = MCTSSearch(test_state, model, komi=1)
    mcts.run_sims(10000)
    print("action probs")
    print(mcts.action_probs())

    print("values")
    print([-c.value_score() if c is not None else 0 for c in mcts.root.children])


custom_state_test()