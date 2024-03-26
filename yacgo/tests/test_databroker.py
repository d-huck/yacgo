"""
Simple script to test databroker flow
"""

import atexit
import multiprocessing as mp
import time
from multiprocessing import Process, Pool

from tqdm.auto import tqdm
from yacgo.data import DataBroker, KataGoDataClient
from yacgo.models import InferenceRandom, Trainer
from yacgo.train_utils import GameGenerator
from yacgo.utils import make_args


def gameplay_worker(i, args):
    """Wrapper around a simple gameplay worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    model = InferenceRandom()
    game_gen = GameGenerator(model, args, display=False)
    try:
        while True:
            game_gen.sim_game()
            # print(f"{i:03d}: Game finished!")
    except KeyboardInterrupt:
        print("Quitting game generation, closing sockets...")
        game_gen.destroy()


def databroker_worker(args):
    """Wrapper around a simple databroker worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    broker = DataBroker(args)
    print("Starting databroker...")
    broker.run()


def trainer_worker(args):
    """Wrapper around a simple trainer worker.

    Args:
        args (dict): args dict.
    """
    trainer = Trainer(args)
    print("Starting trainer...")
    trainer.run()


@atexit.register
def cleanup_wait():
    """
    Cleanup function
    """
    print("Cleaning up...")
    time.sleep(5)


def main():
    """
    Main Process
    """
    mp.set_start_method("spawn")
    args = make_args()

    databroker = Process(target=databroker_worker, args=(args,), daemon=True)
    databroker.start()
    # trainer = Process(target=trainer_worker, args=(args,), daemon=True)
    # trainer.start()
    model = InferenceRandom()
    # time.sleep(5)
    # game_gen = GameGenerator(model, args, display=True)
    games = [(i, args) for i in range(128)]
    pbar = tqdm(total=len(games))
    with Pool(processes=16) as p:
        for _ in p.starmap(gameplay_worker, games):
            pbar.update
    # data = KataGoDataClient(args)
    # data.run("data/kata1-b40c256x2-s5095420928-d1229425124")
    print("Done making data... Terminating processes.")
    # trainer.terminate()
    # trainer.join()
    databroker.terminate()
    databroker.join()
    print("Exiting...")


if __name__ == "__main__":
    main()
