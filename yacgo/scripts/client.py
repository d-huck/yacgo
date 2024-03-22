"""
Runs n GamePlay workers.
"""

import multiprocessing as mp
import time
from multiprocessing import Process

from yacgo.models import InferenceClient
from yacgo.train_utils import GameGenerator
from yacgo.utils import make_args


def gameplay_worker(ports, i, args):
    """Wrapper around a simple gameplay worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    model = InferenceClient(ports)
    game_gen = GameGenerator(model, args)
    print(f"{i:03d}: Starting Game Generation...")
    # data = game_gen.sim_data(1024)
    # for d in data:
    #     data_client.deposit(d)
    try:
        while True:
            game_gen.sim_game()
            print(f"{i:03d}: Game finished!")
    except KeyboardInterrupt:
        print("Quitting game generation, closing sockets...")
        game_gen.destroy()


def main():
    """
    Main Process
    """
    args = make_args()

    # mp.set_start_method("forkserver")
    try:
        games = []
        ports = list(
            range(
                args.inference_server_port,
                args.inference_server_port + args.num_servers,
            )
        )
        for i in range(args.num_games):
            games.append(
                Process(target=gameplay_worker, args=(ports, i, args), daemon=True)
            )
        print("Starting games...")
        for i, game in enumerate(games):
            # time.sleep(1)  # be nice to your cpu
            game.start()

        for game in games:
            game.join()
    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")


if __name__ == "__main__":
    main()
