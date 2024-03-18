"""
Runs n GamePlay workers.
"""

from multiprocessing import Process
from yacgo.utils import make_args
from yacgo.train_utils import GameGenerator
from yacgo.models import InferenceClient
from yacgo.data import DataGameClientMixin


def gameplay_worker(ports, args):
    """Wrapper around a simple gameplay worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    model = InferenceClient(ports)
    data_client = DataGameClientMixin(args)
    game_gen = GameGenerator(args.board_size, model, args)
    print("Starting Game Generation...")
    data = game_gen.sim_data(1024)
    for d in data:
        data_client.deposit(d)
    while True:
        data = game_gen.sim_game()

        for d in data:
            data_client.deposit(d)


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
        print("Starting games...")
        for game in games:
            game.start()

        for game in games:
            game.join()
    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")


if __name__ == "__main__":
    main()
