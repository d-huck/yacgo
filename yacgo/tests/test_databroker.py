"""
Simple script to test databroker flow
"""

import atexit
import multiprocessing as mp
import time
from multiprocessing import Pool, Process

import numpy as np
from tqdm.auto import tqdm

from yacgo.data import (
    DataBroker,
    DataGameClientMixin,
    KataGoDataClient,
    TrainState,
    DATA_DTYPE,
)
from yacgo.models import InferenceRandom, Trainer
from yacgo.train_utils import GameGenerator
from yacgo.utils import make_args


def gameplay_worker(i, args):
    """Wrapper around a simple gameplay worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    # model = InferenceRandom()
    # game_gen = GameGenerator(model, args, display=False)
    client = DataGameClientMixin(args)
    try:
        for _ in range(1024):
            data = TrainState(
                state=DATA_DTYPE(
                    np.random.random(
                        (
                            1,
                            args.num_feature_channels,
                            args.board_size,
                            args.board_size,
                        )
                    )
                ),
                value=DATA_DTYPE(np.random.random()),
                policy=DATA_DTYPE(np.random.random((1, args.board_size**2 + 1))),
            )
            client.deposit(data)

            # print(f"{i:03d}: Game finished!")
    except KeyboardInterrupt:
        print("Quitting game generation, closing sockets...")
        # game_gen.destroy()


def databroker_worker(args):
    """Wrapper around a simple databroker worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    print("Starting databroker...")
    broker = DataBroker(args)
    broker.run()


def trainer_worker(args):
    """Wrapper around a simple trainer worker.

    Args:
        args (dict): args dict.
    """
    trainer = Trainer(args)
    print("Starting trainer...")
    trainer.run()


# @atexit.register
def cleanup_wait():
    """
    Cleanup function
    """
    print("Cleaning up...")
    time.sleep(30)


def main():
    """
    Main Process
    """
    mp.set_start_method("spawn")
    args = make_args()
    args.wandb = False

    databroker = Process(target=databroker_worker, args=(args,), daemon=True)
    databroker.start()

    model = InferenceRandom()
    games = [(i, args) for i in range(16)]
    pbar = tqdm(total=len(games))
    with Pool(processes=16) as p:
        for _ in p.starmap(gameplay_worker, games):
            pbar.update()
    pbar.close()
    print("Done making data... Terminating processes.")
    databroker.terminate()
    databroker.join()
    print("Exiting...")


if __name__ == "__main__":
    main()
