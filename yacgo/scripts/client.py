"""
Runs n GamePlay workers.
"""

from multiprocessing import Process
from yacgo.utils import make_args
from yacgo.train_utils import GameGenerator
from yacgo.models import InferenceClient
from yacgo.databroker import DataGameClient


def gameplay_worker(ports, args):
    """Wrapper around a simple gameplay worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    model = InferenceClient(ports)
    data = DataGameClient(args)
    game_gen = GameGenerator(
        args.board_size,
        model,
    )
    print("Starting Game Generation...")
    while True:
        data = game_gen.sim_game()

        for d in data:
            data.deposit(d)


def main():
    """
    Main Process
    """
    args = make_args()
    try:
        games = []
        ports = list(
            range(
                args.inference_server_port,
                args.inference_server_port + args.num_servers,
            )
        )
        for _ in range(args.num_games):
            games.append(
                Process(target=gameplay_worker, args=(ports, args), daemon=True)
            )

        for game in games:
            game.start()

        for game in games:
            game.join()
    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")
