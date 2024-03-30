"""
Simple script to generate a bunch of data for training the initial model iteration.
Since this only uses a random inferencer, it is much quicker than generating data from
a newly initialized, untrained model.
"""

# from concurrent.futures import ThreadPoolExecutor
import gc
import multiprocessing as mp
import time

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

from yacgo.models import InferenceRandom
from yacgo.train_utils import GameGenerator
from yacgo.utils import make_args


def random_gameplay(args):
    display = True
    model = InferenceRandom(args)

    def game_play_thread():
        try:
            game_gen = GameGenerator(model, args, display=display)
            while True:
                game_gen.sim_game()
                del game_gen
                gc.collect()
                game_gen = GameGenerator(model, args, display=display)
        except KeyboardInterrupt:
            print("Quitting game generation, closing sockets...")

    game_play_thread()


def main():
    """
    Main Process
    """
    args = make_args()

    mp.set_start_method("spawn")
    try:
        games = []
        for i in range(args.num_game_processes):
            games.append(Process(target=random_gameplay, args=(args,), daemon=True))
        print("Starting games...")
        for i, game in enumerate(games):
            game.start()
            time.sleep(1)  # stagger start the workers

        for game in games:
            game.join()
    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")


if __name__ == "__main__":
    main()
