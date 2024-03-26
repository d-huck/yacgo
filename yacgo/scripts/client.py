"""
Runs n GamePlay workers.
"""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import time
from multiprocessing import Process

from yacgo.models import InferenceClient
from yacgo.train_utils import GameGenerator
from yacgo.utils import make_args


def gameplay_worker(ports, i, display, args):
    """Wrapper around a simple gameplay worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """

    def game_play_thread():
        model = InferenceClient(ports, args.inference_server_address)
        game_gen = GameGenerator(model, args, display=display)
        # data = game_gen.sim_data(1024)
        # for d in data:
        #     data_client.deposit(d)
        try:
            while True:
                game_gen.sim_game()
        except KeyboardInterrupt:
            print("Quitting game generation, closing sockets...")
            game_gen.destroy()

    n_threads = args.num_games // args.num_game_processes
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for _ in range(n_threads):
            executor.submit(game_play_thread)


def main():
    """
    Main Process
    """
    args = make_args()

    mp.set_start_method("spawn")
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
        for i, game in enumerate(args.num_game_processes):
            time.sleep(1)  # be nice to your cpu
            game.start()

        for game in games:
            game.join()
    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")


if __name__ == "__main__":
    main()
