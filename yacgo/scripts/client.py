"""
Runs n GamePlay workers.
"""

import gc
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
import threading
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

    def game_play_thread(display):
        model = InferenceClient(ports, args.inference_server_address)
        game_gen = GameGenerator(model, args, display=display)

        try:
            while True:
                game_gen.sim_game()
                del game_gen
                gc.collect()
                game_gen = GameGenerator(model, args, display=display)
        except KeyboardInterrupt:
            print("Quitting game generation, closing sockets...")
            game_gen.destroy()

    n_threads = args.num_games // args.num_game_processes
    for _ in range(n_threads):
        threading.Thread(target=game_play_thread, args=(display,))
        # display = False
    threads = [
        threading.Thread(target=game_play_thread, args=(display,))
        for _ in range(n_threads)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


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
        display = True
        for i in range(args.num_game_processes):
            games.append(
                Process(
                    target=gameplay_worker, args=(ports, i, display, args), daemon=True
                )
            )
            display = False
        print("Starting games...")
        for i, game in enumerate(games):
            game.start()
            time.sleep(2)  # stagger start the workers

        for game in games:
            game.join()
    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")


if __name__ == "__main__":
    main()
