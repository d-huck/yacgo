from yacgo.game import Game
from yacgo.models import InferenceLocal
from yacgo.player import MCTSPlayer, HumanPlayer
from yacgo.go import govars
from yacgo.utils import make_args


def main():
    args = make_args()

    model = InferenceLocal(args)
    black = MCTSPlayer(govars.BLACK, model, args)
    white = HumanPlayer(govars.WHITE)
    test_game = Game(args.board_size, black, white)
    test_game.play_full(print_every=True)


if __name__ == "__main__":
    main()
