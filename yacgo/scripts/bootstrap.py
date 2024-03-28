"""
Simple script to generate a bunch of data for training the initial model iteration.
Since this only uses a random inferencer, it is much quicker than generating data from
a newly initialized, untrained model.
"""

# from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Pool
import signal
from tqdm.auto import tqdm

from yacgo.data import DataBroker
from yacgo.models import InferenceRandom
from yacgo.train_utils import GameGenerator
from yacgo.utils import make_args
import wandb


def random_gameplay(args):
    """Wrapper around a simple gameplay worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """

    def game_play_thread():
        model = InferenceRandom()
        game_gen = GameGenerator(model, args, display=False)
        _ = game_gen.sim_games(64)
        # finally:
        game_gen.destroy()
        return True

    return game_play_thread()


def databroker_worker(args):
    """Wrapper around a simple databroker worker.

    Args:
        port (int): Port server is listening on.
        args (dict): args dict.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            job_type="replay_buffer",
            config=args,
        )

    broker = DataBroker(args)
    broker.run()


def main():
    args = make_args()
    # args.wandb = False

    databroker = Process(target=databroker_worker, args=(args,), daemon=True)
    databroker.start()
    try:
        games = [args for i in range(32000 // 64)]
        pbar = tqdm(total=len(games))
        with Pool(processes=64) as pool:
            for _ in pool.map(random_gameplay, games):
                pbar.update(1)
    except KeyboardInterrupt:
        pass
    databroker.terminate()
    databroker.join()


if __name__ == "__main__":
    main()
