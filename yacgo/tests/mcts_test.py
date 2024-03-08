from yacgo.go import game, govars
from yacgo.game import Game
from yacgo.player import Player, RandomPlayer, HumanPlayer, MCTSPlayer
from yacgo.algos.mcts import MCTSSearch
from yacgo.models import InferenceRandom

# class MCTSTestPlayer(Player):
#     def __init__(self, player, model):
#         super().__init__(player)
#         self.model = model
    
#     def action_probs(self, state):
#         search = MCTSSearch(state, self.model)
#         search.run_sims(1000)
#         return search.action_probs()

# model = EfficientFormerV2(
#     depths=[2, 2, 6, 4],
#     in_chans=6, # num of game state channels
#     img_size=5,
#     embed_dims=(64, 64, 64, 128),
#     downsamples=(False, False, False, True),
#     num_vit=2,
#     mlp_ratios=(4, 4, (4, 3, 3, 3, 4, 4), (4, 3, 3, 4)),
#     num_classes=5**2 + 1  
# )
    
model = InferenceRandom()

black = MCTSPlayer(govars.BLACK, model, sims=1000)
white = RandomPlayer(govars.WHITE)
test_game = Game(5, black, white)
test_game.play_full(print_every=True)